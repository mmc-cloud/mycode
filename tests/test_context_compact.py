from collections.abc import Iterator
from datetime import datetime, timezone
import json

import pytest

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.context_budget import (
    ContextBudget,
    ContextBudgetExceededError,
    MemoryContextStats,
    TokenEstimator,
    TokenUsage,
    estimate_conversation,
    format_model_context_stats,
)
from mycode.context_builder import ContextBuilder
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


def test_agent_runner_preserves_summary_and_stops_when_active_summary_overflows() -> None:
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

    assert events[-1].stop_reason == "context_overflow"
    assert main_client.seen_conversations == []
    context = runner.last_model_context
    assert context is not None
    assert context.estimate.over_budget is True
    assert context.compact_stats is not None
    assert context.compact_stats.status == "insufficient_history"
    assert context.compact_stats.summary_visible is True
    assert any(COMPACT_SUMMARY_MARKER in message.content for message in context.messages)
    assert context.source_message_count == 5
    assert "over_budget=True" in format_model_context_stats(context)
    assert runner.compactor.state.boundary == boundary
    assert summary_client.seen_conversations == []


def test_chat_session_preserves_summary_and_raises_when_active_summary_overflows() -> None:
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

    with pytest.raises(ContextBudgetExceededError) as error:
        session.send_user_message("latest request")

    assert main_client.seen_conversations == []
    context = session.last_model_context
    assert context is error.value.context
    assert context.estimate.over_budget is True
    assert context.compact_stats is not None
    assert context.compact_stats.status == "insufficient_history"
    assert context.compact_stats.summary_visible is True
    assert any(COMPACT_SUMMARY_MARKER in message.content for message in context.messages)
    assert context.source_message_count == 5
    assert session.compactor.state.boundary == boundary
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


@pytest.mark.parametrize("entry", ["agent", "chat", "chat_stream"])
@pytest.mark.parametrize(
    "status", ["not_needed", "compacted", "active", "failed", "cooldown", "circuit_open"]
)
def test_shared_pipeline_preserves_compact_states(entry, status, monkeypatch, tmp_path) -> None:
    import mycode.context_builder as builder_module

    summary_client = RecordingSummaryClient(
        responses=["invalid" if status == "failed" else summary_json("continue")],
        token_usages=[TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)],
    )
    boundary = CompactBoundary(
        boundary_id="restored-boundary",
        covered_message_count=2,
        covered_turn_count=1,
        summary=CompactSummary.model_validate_json(summary_json("restored")),
        source_estimated_tokens=1000,
        created_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )
    state = CompactState()
    if status == "active":
        state = CompactState(boundary=boundary)
    elif status in {"cooldown", "circuit_open"}:
        state = CompactState(
            boundary=boundary,
            consecutive_failure_count=1 if status == "cooldown" else 3,
            retry_after_message_count=100,
        )
    compactor = ConversationCompactor(
        llm_client=summary_client,
        policy=CompactPolicy(
            trigger_ratio=1 if status in {"not_needed", "active"} else 0.01,
            recent_turns_to_keep=2,
        ),
        state=state,
    )
    stages = []
    prepared_results = []
    real_prepare = compactor.prepare
    real_budget = builder_module.budget_model_context

    def prepare(*args, **kwargs):
        stages.append("compact")
        prepared = real_prepare(*args, **kwargs)
        prepared_results.append(prepared)
        return prepared

    def budget(*args, **kwargs):
        stages.append("budget")
        return real_budget(*args, **kwargs)

    monkeypatch.setattr(compactor, "prepare", prepare)
    monkeypatch.setattr(builder_module, "budget_model_context", budget)
    history = conversation_with_tool_turn()
    original = history.get_messages()
    if entry == "agent":
        client = RecordingAgentClient(
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        )
        from mycode.artifacts import ToolResultArtifactStore
        owner = AgentRunner(
            llm_client=client, tool_registry=ToolRegistry(), conversation=history,
            compactor=compactor, context_budget=context_budget(2000),
            tool_result_artifact_store=ToolResultArtifactStore(tmp_path / "a", 10),
        )
        assert list(owner.run("next"))[-1].stop_reason == "final_answer"
    else:
        client = RecordingChatClient()
        owner = ChatSession(
            llm_client=client, conversation=history,
            compactor=compactor, context_budget=context_budget(2000),
        )
        if entry == "chat_stream":
            assert "".join(owner.stream_user_message("next")) == "chat done"
        else:
            assert owner.send_user_message("next").content == "chat done"

    context = owner.last_model_context
    assert stages == ["compact", "budget"]
    assert context.compact_stats is prepared_results[0].stats
    assert context.compact_stats.status == status
    assert context.estimate.estimated_input_tokens <= context.estimate.max_input_tokens
    assert client.seen_conversations == [
        Conversation.from_messages(list(context.messages)).to_model_messages()
    ]
    has_summary = any(COMPACT_SUMMARY_MARKER in m.content for m in context.messages)
    assert has_summary == (status in {"compacted", "active", "cooldown", "circuit_open"})
    assert history.get_messages()[:len(original)] == original
    assert len(summary_client.seen_conversations) == (1 if status in {"compacted", "failed"} else 0)
    if entry != "agent":
        assert owner.last_compact_token_usage == prepared_results[0].attempt_token_usage


@pytest.mark.parametrize("omit_memory", [False, True])
def test_builder_assembles_memory_guidance_and_tools_before_final_budget(
    omit_memory, monkeypatch,
) -> None:
    import mycode.context_builder as builder_module

    history = conversation_with_tool_turn()
    memory = Message(role="system", content="memory " * (1000 if omit_memory else 1))
    guidance = ("Keep the current runtime target.", "Use existing evidence.")
    tools = [{"name": "read_file", "parameters": {"type": "object"}}]
    compactor = ConversationCompactor(
        llm_client=RecordingSummaryClient(responses=[summary_json("compact with memory")]),
        policy=CompactPolicy(trigger_ratio=0.01, recent_turns_to_keep=2),
    )
    real_prepare = compactor.prepare
    real_budget = builder_module.budget_model_context
    stages = []

    def prepare(conversation, *args, **kwargs):
        stages.append("compact")
        assert conversation is history
        assert kwargs["memory_message"] is memory
        assert kwargs["tools"] is tools
        assert not any(guidance[0] in m.content for m in conversation.get_messages())
        return real_prepare(conversation, *args, **kwargs)

    def budget(messages, *args, **kwargs):
        stages.append("budget")
        assert messages[-2] is memory
        assert messages[-1] == Message(role="system", content="\n\n".join(guidance))
        assert any(COMPACT_SUMMARY_MARKER in m.content for m in messages)
        assert kwargs["tools"] is tools
        return real_budget(messages, *args, **kwargs)

    monkeypatch.setattr(compactor, "prepare", prepare)
    monkeypatch.setattr(builder_module, "budget_model_context", budget)
    result = ContextBuilder(context_budget(2000), TokenEstimator(), compactor).build(
        history, tools=tools, memory_message=memory,
        memory_stats=MemoryContextStats(selected_entry_count=1), guidance=guidance,
    )
    context = result.context
    assert stages == ["compact", "budget"]
    assert context.compact_stats.status == "compacted"
    assert context.compact_stats.summary_visible is True
    assert any(COMPACT_SUMMARY_MARKER in m.content for m in context.messages)
    assert context.memory_stats.included_entry_count == (0 if omit_memory else 1)
    assert (memory in context.messages) is not omit_memory
    assert any(m.content == "\n\n".join(guidance) for m in context.messages)
    assert context.estimate.tool_schema_chars > 0
    assert not context.estimate.over_budget


def test_compact_summary_survives_guidance_driven_history_trimming() -> None:
    history = conversation_with_tool_turn()
    policy = CompactPolicy(trigger_ratio=0.01, recent_turns_to_keep=2)
    # Size the test budget using a real summary view, then build with a fresh
    # compactor so the tested request actually performs successful Compact.
    preview = ConversationCompactor(
        RecordingSummaryClient(responses=[summary_json("continue")]), policy,
    ).prepare(history, context_budget(2000), token_estimator=TokenEstimator())
    guidance = "runtime guidance " * 300
    minimal = Conversation.from_messages([
        preview.conversation.get_messages()[0],
        Message(role="system", content=guidance),
        history.get_messages()[-1],
    ])
    budget = context_budget(estimate_conversation(minimal).estimated_input_tokens + 10)
    compactor = ConversationCompactor(
        RecordingSummaryClient(responses=[summary_json("continue")]), policy,
    )

    context = ContextBuilder(budget, TokenEstimator(), compactor).build(
        history, guidance=(guidance,),
    ).context

    assert context.compact_stats.status == "compacted"
    assert context.compact_stats.summary_visible is True
    assert COMPACT_SUMMARY_MARKER in context.messages[0].content
    assert context.messages[1].content == guidance
    assert context.messages[-1] == history.get_messages()[-1]
    assert context.trimmed_message_count > 0
    assert not context.estimate.over_budget


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

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        self.seen_conversations.append(conversation.to_model_messages())
        yield "chat done"


@pytest.mark.parametrize("reuse", [False, True])
def test_retention_precedes_compact_and_budget_does_not_repeat_it(tmp_path, monkeypatch, reuse):
    from dataclasses import replace
    from mycode.artifacts import ToolResultArtifactStore, EXTERNALIZED_TOOL_RESULT_MARKER
    from mycode.tool_result_retention import ToolResultRetentionPolicy, TurnLocalFullGroup

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    messages = conversation_with_tool_turn().get_messages()
    full = "OK\n" + "large evidence " * 4000
    reference = store.externalize(tool_name="read_file", tool_call_id="call-read", content=full)
    messages[4] = replace(messages[4], content=reference)
    history = Conversation.from_messages(messages)
    handoff = TurnLocalFullGroup(messages[3], {"call-read": (reference, full)})
    observed_states = []
    client = RecordingSummaryClient(responses=[summary_json("retention compatibility")])
    compactor = ConversationCompactor(
        client, CompactPolicy(trigger_ratio=0.01, recent_turns_to_keep=2),
        on_state_changed=observed_states.append,
    )
    real_prepare = compactor.prepare
    prepared = []

    def prepare(conversation, *args, **kwargs):
        assert conversation.get_messages()[4].content == full
        result = real_prepare(conversation, *args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(compactor, "prepare", prepare)
    budget = context_budget(3000)
    context = ContextBuilder(
        budget, TokenEstimator(), compactor, ToolResultRetentionPolicy(budget, store),
    ).build(history, guidance=("protected runtime guidance",),
            turn_local_full_group=handoff if reuse else None).context
    assert len(prepared) == len(observed_states) == len(client.seen_conversations) == 1
    assert context.compact_stats is prepared[0].stats
    assert context.compact_stats.status == "compacted"
    assert context.compact_stats.compacted_message_count == 2
    assert context.compact_stats.covered_turn_count == 1
    assert compactor.state.boundary == observed_states[0].boundary
    assert context.compact_stats.summary_visible
    assert any(COMPACT_SUMMARY_MARKER in m.content for m in context.messages)
    assert any(m.content == "current request" for m in context.messages)
    assert any(m.content == "protected runtime guidance" for m in context.messages)
    assert all(EXTERNALIZED_TOOL_RESULT_MARKER in m.content for m in context.messages if m.role == "tool")
    assert context.retention_stats.artifact_groups == 1
    assert context.retention_stats.budget_downgraded_groups == 1
    assert history.get_messages() == messages
    assert not context.estimate.over_budget
