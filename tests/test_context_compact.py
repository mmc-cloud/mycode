from collections.abc import Iterator
from datetime import datetime, timezone
import json

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.context_budget import (
    ContextBudget,
    TokenEstimator,
    TokenUsage,
    format_model_context_stats,
)
from mycode.context_compact import (
    COMPACT_SUMMARY_MARKER,
    CompactBoundary,
    CompactPolicy,
    CompactState,
    CompactSummary,
    ConversationCompactor,
)
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.runner import AgentRunner
from mycode.session import ChatSession
from mycode.tools import ToolRegistry


def context_budget(max_input_tokens: int) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )


def summary_json(objective: str) -> str:
    return json.dumps(
        {
            "objective": objective,
            "progress": ["progress"],
            "decisions": ["decision"],
            "constraints": ["constraint"],
            "open_items": ["next"],
            "references": ["artifact_path: C:/state/artifact.txt"],
        }
    )


def conversation_with_tool_turn() -> Conversation:
    tool_call = AgentToolCall(
        id="call-read",
        name="read_file",
        arguments={"path": "README.md"},
    )
    return Conversation.from_messages(
        [
            Message(role="user", content="first request " * 40),
            Message(role="assistant", content="first reply " * 40),
            Message(role="user", content="second request"),
            Message(
                role="assistant",
                content="",
                tool_calls=(tool_call,),
                reasoning_content="private synthetic reasoning",
            ),
            Message(
                role="tool",
                content="OK\nartifact reference",
                tool_call_id="call-read",
            ),
            Message(role="assistant", content="second reply"),
            Message(role="user", content="current request"),
        ]
    )


def test_compactor_summarizes_old_turns_and_keeps_recent_protocol_groups() -> None:
    client = RecordingSummaryClient(
        responses=[summary_json("continue the task")],
        token_usages=[
            TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150)
        ],
    )
    observed_states: list[CompactState] = []
    compactor = ConversationCompactor(
        llm_client=client,
        policy=CompactPolicy(
            trigger_ratio=0.01,
            recent_turns_to_keep=2,
        ),
        on_state_changed=observed_states.append,
    )
    conversation = conversation_with_tool_turn()

    prepared = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )

    assert prepared.stats.status == "compacted"
    assert prepared.stats.compacted_message_count == 2
    assert prepared.stats.covered_turn_count == 1
    assert prepared.attempt_token_usage == TokenUsage(
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
    )
    visible = prepared.conversation.get_messages()
    assert visible[0].role == "system"
    assert COMPACT_SUMMARY_MARKER in visible[0].content
    assert visible[1:] == conversation.get_messages()[2:]
    assert visible[2].tool_calls[0].id == "call-read"
    assert visible[3].tool_call_id == "call-read"
    assert observed_states == [compactor.state]

    boundary = compactor.state.boundary
    assert boundary is not None
    assert boundary.summary_prompt_tokens == 120
    assert boundary.summary_completion_tokens == 30
    prompt_payload = json.loads(client.seen_conversations[0][1]["content"])
    assert [message["role"] for message in prompt_payload["new_messages"]] == [
        "user",
        "assistant",
    ]


def test_compactor_incrementally_merges_previous_summary_without_splitting_tools() -> None:
    client = RecordingSummaryClient(
        responses=[
            summary_json("first compact"),
            summary_json("second compact"),
        ]
    )
    compactor = ConversationCompactor(
        llm_client=client,
        policy=CompactPolicy(
            trigger_ratio=0.01,
            recent_turns_to_keep=2,
        ),
    )
    conversation = conversation_with_tool_turn()
    first = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )
    assert first.stats.compacted_message_count == 2

    conversation.add_assistant_message("current reply")
    conversation.add_user_message("fourth request")
    second = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )

    assert second.stats.status == "compacted"
    assert second.stats.compacted_message_count == 6
    assert second.stats.covered_turn_count == 2
    prompt_payload = json.loads(client.seen_conversations[1][1]["content"])
    assert prompt_payload["previous_summary"]["objective"] == "first compact"
    new_messages = prompt_payload["new_messages"]
    assert [message["role"] for message in new_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert new_messages[1]["tool_calls"][0]["id"] == "call-read"
    assert "reasoning_content" not in new_messages[1]
    assert new_messages[2]["tool_call_id"] == "call-read"


def test_compactor_failure_uses_cooldown_and_keeps_full_history_for_fallback() -> None:
    client = RecordingSummaryClient(responses=["not-json"])
    compactor = ConversationCompactor(
        llm_client=client,
        policy=CompactPolicy(
            trigger_ratio=0.01,
            recent_turns_to_keep=1,
            failure_cooldown_messages=5,
        ),
    )
    conversation = conversation_with_tool_turn()

    failed = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )
    cooled_down = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )

    assert failed.stats.status == "failed"
    assert failed.conversation.get_messages() == conversation.get_messages()
    assert failed.stats.consecutive_failure_count == 1
    assert failed.stats.retry_after_message_count == 12
    assert compactor.state.last_failure_reason == "compact_failure:ValidationError"
    assert "not-json" not in (compactor.state.last_failure_reason or "")
    assert cooled_down.stats.status == "cooldown"
    assert len(client.seen_conversations) == 1


def test_compactor_restores_persisted_cooldown_without_retrying_immediately() -> None:
    saved_states: list[CompactState] = []
    first_client = RecordingSummaryClient(responses=["invalid"])
    first_compactor = ConversationCompactor(
        llm_client=first_client,
        policy=CompactPolicy(
            trigger_ratio=0.01,
            recent_turns_to_keep=1,
            failure_cooldown_messages=5,
        ),
        on_state_changed=saved_states.append,
    )
    conversation = conversation_with_tool_turn()
    first_compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )
    restored_client = RecordingSummaryClient(responses=[])
    restored = ConversationCompactor(
        llm_client=restored_client,
        policy=first_compactor.policy,
        state=saved_states[-1],
    )

    prepared = restored.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )

    assert prepared.stats.status == "cooldown"
    assert prepared.stats.consecutive_failure_count == 1
    assert restored_client.seen_conversations == []


def test_compactor_opens_circuit_after_repeated_failures() -> None:
    client = RecordingSummaryClient(responses=["bad-json", "still-bad"])
    compactor = ConversationCompactor(
        llm_client=client,
        policy=CompactPolicy(
            trigger_ratio=0.01,
            recent_turns_to_keep=1,
            failure_cooldown_messages=1,
            breaker_failure_threshold=2,
            breaker_cooldown_messages=10,
        ),
    )
    conversation = conversation_with_tool_turn()

    first = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )
    assert first.stats.status == "failed"
    conversation.add_assistant_message("complete current turn")
    second = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )
    blocked = compactor.prepare(
        conversation,
        context_budget(2000),
        token_estimator=TokenEstimator(),
    )

    assert second.stats.status == "failed"
    assert second.stats.consecutive_failure_count == 2
    assert blocked.stats.status == "circuit_open"
    assert blocked.stats.circuit_open is True
    assert len(client.seen_conversations) == 2


def test_compactor_ignores_invalid_boundary_instead_of_splitting_history() -> None:
    invalid_boundary = CompactBoundary(
        boundary_id="invalid",
        covered_message_count=1,
        covered_turn_count=1,
        summary=CompactSummary(
            objective="objective",
            progress=(),
            decisions=(),
            constraints=(),
            open_items=(),
            references=(),
        ),
        source_estimated_tokens=100,
        created_at=datetime.now(timezone.utc),
    )
    compactor = ConversationCompactor(
        llm_client=RecordingSummaryClient(responses=[]),
        state=CompactState(boundary=invalid_boundary),
    )
    conversation = conversation_with_tool_turn()

    prepared = compactor.prepare(
        conversation,
        context_budget(10000),
        token_estimator=TokenEstimator(),
    )

    assert prepared.stats.status == "invalid_boundary"
    assert prepared.stats.boundary_id is None
    assert prepared.conversation.get_messages() == conversation.get_messages()
    assert compactor.state.boundary is None
    assert compactor.state.consecutive_failure_count == 1


def test_agent_runner_sends_summary_plus_tail_and_accounts_summary_usage() -> None:
    summary_client = RecordingSummaryClient(
        responses=[summary_json("runner compact")],
        token_usages=[
            TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        ],
    )
    main_client = RecordingAgentClient(
        usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
    )
    runner = AgentRunner(
        llm_client=main_client,
        tool_registry=ToolRegistry(),
        conversation=conversation_with_tool_turn(),
        context_budget=context_budget(2000),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            policy=CompactPolicy(
                trigger_ratio=0.01,
                recent_turns_to_keep=2,
            ),
        ),
    )

    events = list(runner.run("fourth request"))

    assert [event.content for event in events if event.type == "text_delta"] == ["done"]
    assert COMPACT_SUMMARY_MARKER in main_client.seen_conversations[0][0]["content"]
    assert all(
        "first request" not in str(message.get("content"))
        for message in main_client.seen_conversations[0]
    )
    assert runner.run_token_usage == TokenUsage(
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
    )
    assert runner.last_model_context is not None
    assert runner.last_model_context.compact_stats is not None
    assert runner.last_model_context.compact_stats.status == "compacted"


def test_agent_runner_falls_back_to_deterministic_trim_when_summary_is_invalid() -> None:
    summary_client = RecordingSummaryClient(responses=["invalid summary"])
    main_client = RecordingAgentClient(
        usage=TokenUsage(prompt_tokens=30, completion_tokens=5, total_tokens=35)
    )
    runner = AgentRunner(
        llm_client=main_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request"),
                Message(role="assistant", content="old reply"),
                Message(role="user", content="large current " * 500),
                Message(role="assistant", content="large reply " * 500),
            ]
        ),
        context_budget=context_budget(800),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            policy=CompactPolicy(
                trigger_ratio=0.01,
                recent_turns_to_keep=2,
            ),
        ),
    )

    events = list(runner.run("latest request"))

    assert [event.content for event in events if event.type == "text_delta"] == ["done"]
    assert main_client.seen_conversations == [
        [{"role": "user", "content": "latest request"}]
    ]
    assert runner.last_model_context is not None
    assert runner.last_model_context.trimmed is True
    assert runner.last_model_context.compact_stats is not None
    assert runner.last_model_context.compact_stats.status == "failed"
    assert len(summary_client.seen_conversations) == 1


def test_agent_runner_retries_canonical_history_when_active_summary_overflows() -> None:
    summary_client = RecordingSummaryClient(
        responses=[summary_json("unused")],
    )
    main_client = RecordingAgentClient(
        usage=TokenUsage(prompt_tokens=30, completion_tokens=5, total_tokens=35)
    )
    boundary = CompactBoundary(
        boundary_id="oversized-summary",
        covered_message_count=2,
        covered_turn_count=1,
        summary=CompactSummary(
            objective="S" * 1800,
            progress=(),
            decisions=(),
            constraints=(),
            open_items=(),
            references=(),
        ),
        source_estimated_tokens=1000,
        created_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )
    runner = AgentRunner(
        llm_client=main_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="system", content="system prompt"),
                Message(role="user", content="old request " * 120),
                Message(role="assistant", content="old reply " * 120),
                Message(role="user", content="current request"),
            ]
        ),
        context_budget=context_budget(400),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            state=CompactState(boundary=boundary),
        ),
    )

    events = list(runner.run("latest request"))

    assert [event.content for event in events if event.type == "text_delta"] == ["done"]
    sent = main_client.seen_conversations[0]
    assert all(COMPACT_SUMMARY_MARKER not in str(item.get("content")) for item in sent)
    assert all("old request" not in str(item.get("content")) for item in sent)
    assert any(item.get("content") == "latest request" for item in sent)
    assert runner.last_model_context is not None
    assert runner.last_model_context.estimate.over_budget is False
    assert runner.last_model_context.compact_stats is not None
    assert (
        runner.last_model_context.compact_stats.status
        == "canonical_fallback"
    )
    assert runner.last_model_context.compact_stats.summary_visible is False
    assert runner.last_model_context.source_message_count == 5
    context_stats = format_model_context_stats(runner.last_model_context)
    assert "messages=3/5" in context_stats
    assert "compact_summary_visible=false" in context_stats
    assert summary_client.seen_conversations == []


def test_chat_session_retries_canonical_history_when_active_summary_overflows() -> None:
    summary_client = RecordingSummaryClient(
        responses=[summary_json("unused")],
    )
    main_client = RecordingChatClient()
    boundary = CompactBoundary(
        boundary_id="oversized-chat-summary",
        covered_message_count=2,
        covered_turn_count=1,
        summary=CompactSummary(
            objective="S" * 1800,
            progress=(),
            decisions=(),
            constraints=(),
            open_items=(),
            references=(),
        ),
        source_estimated_tokens=1000,
        created_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )
    session = ChatSession(
        llm_client=main_client,
        conversation=Conversation.from_messages(
            [
                Message(role="system", content="system prompt"),
                Message(role="user", content="old request " * 120),
                Message(role="assistant", content="old reply " * 120),
                Message(role="user", content="current request"),
            ]
        ),
        context_budget=context_budget(400),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            state=CompactState(boundary=boundary),
        ),
    )

    reply = session.send_user_message("latest request")

    assert reply.content == "chat done"
    sent = main_client.seen_conversations[0]
    assert all(COMPACT_SUMMARY_MARKER not in str(item.get("content")) for item in sent)
    assert all("old request" not in str(item.get("content")) for item in sent)
    assert any(item.get("content") == "latest request" for item in sent)
    assert session.last_model_context is not None
    assert session.last_model_context.estimate.over_budget is False
    assert session.last_model_context.compact_stats is not None
    assert (
        session.last_model_context.compact_stats.status
        == "canonical_fallback"
    )
    assert session.last_model_context.compact_stats.summary_visible is False
    assert session.last_model_context.source_message_count == 5
    assert summary_client.seen_conversations == []


def test_chat_session_uses_same_compact_summary_plus_recent_tail_model() -> None:
    summary_client = RecordingSummaryClient(
        responses=[summary_json("chat compact")]
    )
    main_client = RecordingChatClient()
    history = Conversation.from_messages(
        [
            Message(role="user", content="old request " * 50),
            Message(role="assistant", content="old reply " * 50),
            Message(role="user", content="recent request"),
            Message(role="assistant", content="recent reply"),
        ]
    )
    session = ChatSession(
        llm_client=main_client,
        conversation=history,
        context_budget=context_budget(2000),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            policy=CompactPolicy(
                trigger_ratio=0.01,
                recent_turns_to_keep=2,
            ),
        ),
    )

    reply = session.send_user_message("current request")

    assert reply.content == "chat done"
    assert COMPACT_SUMMARY_MARKER in main_client.seen_conversations[0][0]["content"]
    assert history.get_messages()[0].content.startswith("old request")
    assert history.get_messages()[-1] == Message(
        role="assistant",
        content="chat done",
    )


class RecordingSummaryClient:
    def __init__(
        self,
        *,
        responses: list[str],
        token_usages: list[TokenUsage | None] | None = None,
    ) -> None:
        self.responses = responses
        self.token_usages = [] if token_usages is None else token_usages
        self.last_token_usage: TokenUsage | None = None
        self.seen_conversations: list[list[dict[str, object]]] = []

    def complete(self, conversation: Conversation) -> Message:
        self.seen_conversations.append(conversation.to_model_messages())
        self.last_token_usage = (
            self.token_usages.pop(0) if self.token_usages else None
        )
        if not self.responses:
            raise RuntimeError("No summary response")
        return Message(role="assistant", content=self.responses.pop(0))

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        raise NotImplementedError


class RecordingAgentClient:
    def __init__(self, *, usage: TokenUsage) -> None:
        self.usage = usage
        self.last_token_usage: TokenUsage | None = None
        self.seen_conversations: list[list[dict[str, object]]] = []

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        self.seen_conversations.append(conversation.to_model_messages())
        self.last_token_usage = self.usage
        yield AgentEvent(type="text_delta", content="done")


class RecordingChatClient:
    def __init__(self) -> None:
        self.last_token_usage = None
        self.seen_conversations: list[list[dict[str, object]]] = []

    def complete(self, conversation: Conversation) -> Message:
        self.seen_conversations.append(conversation.to_model_messages())
        return Message(role="assistant", content="chat done")
