import json

import pytest

from mycode.agent import AgentToolCall
from mycode.context_budget import (
    ContextBudget,
    MemoryContextStats,
    TokenEstimator,
    TokenUsage,
    build_model_context as _build_model_context,
    estimate_conversation as _estimate_conversation,
    estimate_message,
    format_model_context_stats,
)
from mycode.conversation import Conversation
from mycode.messages import Message


UNIT_TOKEN_ESTIMATOR = TokenEstimator(
    ascii_tokens_per_char=1.0,
    mixed_tokens_per_char=1.0,
    non_ascii_tokens_per_char=1.0,
    calibration_safety_factor=1.0,
)


def context_budget(max_input_tokens: int, **kwargs: int) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        **kwargs,
    )


def estimate_conversation(conversation, budget=None, tools=None):
    return _estimate_conversation(
        conversation,
        budget,
        tools,
        token_estimator=UNIT_TOKEN_ESTIMATOR,
    )


def build_model_context(conversation, budget=None, tools=None, **kwargs):
    return _build_model_context(
        conversation,
        budget,
        tools,
        token_estimator=UNIT_TOKEN_ESTIMATOR,
        **kwargs,
    )


def test_context_budget_defaults_to_128k_with_output_and_safety_reserves() -> None:
    budget = ContextBudget()

    assert budget.context_window_tokens == 128000
    assert budget.reserved_output_tokens == 8192
    assert budget.safety_margin_tokens == 4096
    assert budget.max_input_tokens == 115712


def test_context_budget_requires_positive_available_input_tokens() -> None:
    with pytest.raises(ValueError, match="context_window_tokens"):
        ContextBudget(context_window_tokens=0)

    with pytest.raises(ValueError, match="leave at least 1 input token"):
        ContextBudget(
            context_window_tokens=100,
            reserved_output_tokens=60,
            safety_margin_tokens=40,
        )


def test_context_budget_validates_tool_result_compression_settings() -> None:
    with pytest.raises(ValueError, match="tool_result_compression_threshold_chars"):
        ContextBudget(tool_result_compression_threshold_chars=0)

    with pytest.raises(ValueError, match="recent_tool_result_groups_to_keep"):
        ContextBudget(recent_tool_result_groups_to_keep=-1)


def test_estimate_message_counts_role_and_content() -> None:
    message = Message(role="user", content="hello")
    estimate = estimate_message(message)

    assert estimate.role == "user"
    assert estimate.content_chars == 5
    assert estimate.tool_call_chars == 0
    assert estimate.tool_call_id_chars == 0
    assert estimate.serialization_overhead_chars > 0
    assert estimate.total_chars == len(
        json.dumps(message.to_model_dict(), ensure_ascii=False)
    )


def test_estimate_message_counts_tool_call_arguments() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=(
            AgentToolCall(
                id="call_123",
                name="read_file",
                arguments={"path": "README.md", "max_lines": 20},
            ),
        ),
    )

    estimate = estimate_message(message)

    assert estimate.role == "assistant"
    assert estimate.content_chars == 0
    assert estimate.tool_call_chars == len(
        json.dumps(message.to_model_dict()["tool_calls"], ensure_ascii=False)
    )
    assert estimate.total_chars == len(
        json.dumps(message.to_model_dict(), ensure_ascii=False)
    )


def test_estimate_message_counts_reasoning_content() -> None:
    tool_call = AgentToolCall(
        id="call_reasoning",
        name="read_file",
        arguments={"path": "README.md"},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(tool_call,),
        reasoning_content="private synthetic reasoning",
    )

    estimate = estimate_message(message)

    assert estimate.reasoning_chars == len("private synthetic reasoning")
    assert estimate.total_chars == len(
        json.dumps(message.to_model_dict(), ensure_ascii=False)
    )


def test_estimate_message_counts_present_empty_reasoning_as_zero_chars() -> None:
    tool_call = AgentToolCall(
        id="call_empty_reasoning",
        name="read_file",
        arguments={"path": "README.md"},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(tool_call,),
        reasoning_state="present_empty",
    )

    estimate = estimate_message(message)

    assert estimate.reasoning_chars == 0
    assert estimate.total_chars == len(
        json.dumps(message.to_model_dict(), ensure_ascii=False)
    )


def test_estimate_message_counts_unicode_characters_not_bytes() -> None:
    message = Message(
        role="assistant",
        content="你好",
        tool_calls=(
            AgentToolCall(
                id="call_unicode",
                name="grep",
                arguments={"query": "阶段八", "path_pattern": "docs/**/*.md"},
            ),
        ),
    )

    estimate = estimate_message(message)

    assert estimate.content_chars == len("你好")
    assert len("你好") == 2
    assert len("你好".encode("utf-8")) == 6
    assert estimate.tool_call_chars == len(
        json.dumps(message.to_model_dict()["tool_calls"], ensure_ascii=False)
    )


def test_default_token_estimator_accounts_for_chinese_character_density() -> None:
    estimator = TokenEstimator()
    budget = ContextBudget()
    ascii_conversation = Conversation.from_messages(
        [Message(role="user", content="a" * 1000)]
    )
    chinese_conversation = Conversation.from_messages(
        [Message(role="user", content="中" * 1000)]
    )

    ascii_estimate = _estimate_conversation(
        ascii_conversation,
        budget,
        token_estimator=estimator,
    )
    chinese_estimate = _estimate_conversation(
        chinese_conversation,
        budget,
        token_estimator=estimator,
    )

    assert ascii_estimate.token_profile == "ascii"
    assert chinese_estimate.token_profile == "non_ascii"
    assert chinese_estimate.estimated_input_tokens > ascii_estimate.estimated_input_tokens


def test_token_estimator_calibrates_from_real_prompt_usage() -> None:
    estimator = TokenEstimator(calibration_safety_factor=1.1)
    conversation = Conversation.from_messages(
        [Message(role="user", content="a" * 100)]
    )
    initial = _estimate_conversation(
        conversation,
        ContextBudget(),
        token_estimator=estimator,
    )

    estimator.observe(
        initial,
        TokenUsage(
            prompt_tokens=initial.total_chars,
            completion_tokens=10,
            total_tokens=initial.total_chars + 10,
        ),
    )
    calibrated = _estimate_conversation(
        conversation,
        ContextBudget(),
        token_estimator=estimator,
    )

    assert initial.token_estimate_source == "default"
    assert calibrated.token_estimate_source == "calibrated"
    assert calibrated.tokens_per_char == pytest.approx(1.1)
    assert calibrated.estimated_input_tokens == pytest.approx(
        calibrated.total_chars * 1.1,
        abs=1,
    )


def test_estimate_message_counts_tool_result_id() -> None:
    message = Message(
        role="tool",
        content="OK\n1 | hello",
        tool_call_id="call_123",
    )
    estimate = estimate_message(message)

    assert estimate.content_chars == len("OK\n1 | hello")
    assert estimate.tool_call_id_chars == len("call_123")
    assert estimate.total_chars == len(
        json.dumps(message.to_model_dict(), ensure_ascii=False)
    )


def test_estimate_conversation_sums_message_estimates() -> None:
    conversation = Conversation()
    conversation.add_system_message("system rules")
    conversation.add_user_message("hello")
    conversation.add_assistant_message("hi")

    estimate = estimate_conversation(conversation, context_budget(200))

    assert estimate.message_count == 3
    assert estimate.max_input_tokens == 200
    assert estimate.estimated_input_tokens == estimate.total_chars
    serialized_message_chars = sum(
        message_estimate.total_chars
        for message_estimate in estimate.message_estimates
    )
    assert estimate.message_list_overhead_chars > 0
    assert estimate.message_chars == (
        serialized_message_chars + estimate.message_list_overhead_chars
    )
    assert estimate.message_chars == len(
        json.dumps(conversation.to_model_messages(), ensure_ascii=False)
    )
    assert estimate.tool_schema_chars == 0
    assert estimate.total_chars == estimate.message_chars
    assert estimate.over_budget is False


def test_estimate_conversation_counts_tool_schema_overhead() -> None:
    conversation = Conversation.from_messages([Message(role="user", content="read")])
    tools = [
        {
            "name": "read_file",
            "description": "Read a text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]

    estimate = estimate_conversation(
        conversation,
        context_budget(100),
        tools=tools,
    )

    assert estimate.message_chars == len(
        json.dumps(conversation.to_model_messages(), ensure_ascii=False)
    )
    assert estimate.tool_schema_chars == len(
        json.dumps(
            [{"type": "function", "function": dict(tools[0])}],
            ensure_ascii=False,
        )
    )
    assert estimate.total_chars == estimate.message_chars + estimate.tool_schema_chars


def test_estimate_conversation_marks_over_budget() -> None:
    conversation = Conversation.from_messages(
        [
            Message(role="user", content="12345"),
            Message(role="assistant", content="67890"),
        ]
    )

    estimate = estimate_conversation(conversation, context_budget(10))

    assert estimate.total_chars > 10
    assert estimate.over_budget is True


def test_estimate_conversation_can_be_over_budget_from_tools_only() -> None:
    conversation = Conversation.from_messages([Message(role="user", content="hi")])

    estimate = estimate_conversation(
        conversation,
        context_budget(40),
        tools=[
            {
                "name": "tool_with_large_schema",
                "description": "x" * 100,
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    assert estimate.message_chars <= 40
    assert estimate.tool_schema_chars > 40
    assert estimate.over_budget is True


def test_build_model_context_keeps_all_messages_when_under_budget() -> None:
    conversation = Conversation()
    conversation.add_system_message("rules")
    conversation.add_user_message("hello")

    context = build_model_context(conversation, context_budget(100))

    assert context.messages == tuple(conversation.get_messages())
    assert context.trimmed is False
    assert context.trimmed_message_count == 0


def test_build_model_context_trims_old_messages_and_keeps_system() -> None:
    conversation = Conversation.from_messages(
        [
            Message(role="system", content="rules"),
            Message(role="user", content="old request " * 20),
            Message(role="assistant", content="old reply " * 20),
            Message(role="user", content="current"),
        ]
    )

    context = build_model_context(conversation, context_budget(100))

    assert context.messages == (
        Message(role="system", content="rules"),
        Message(role="user", content="current"),
    )
    assert context.trimmed is True
    assert context.trimmed_message_count == 2


def test_build_model_context_keeps_tool_call_and_result_together() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    tool_result = Message(
        role="tool",
        content="OK\n1 | hello",
        tool_call_id="call_123",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="user", content="old request " * 20),
            Message(
                role="assistant",
                content="",
                tool_calls=(tool_call,),
                reasoning_content="private synthetic reasoning",
            ),
            tool_result,
        ]
    )

    context = build_model_context(conversation, context_budget(10))

    assert context.messages == (
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
        tool_result,
    )
    assert context.estimate.over_budget is True


def test_build_model_context_keeps_multiple_tool_calls_and_results_together() -> None:
    first_call = AgentToolCall(
        id="call_first",
        name="read_file",
        arguments={"path": "first.py"},
    )
    second_call = AgentToolCall(
        id="call_second",
        name="read_file",
        arguments={"path": "second.py"},
    )
    assistant = Message(
        role="assistant",
        content="",
        tool_calls=(first_call, second_call),
    )
    first_result = Message(
        role="tool",
        content="OK\nfirst",
        tool_call_id="call_first",
    )
    second_result = Message(
        role="tool",
        content="OK\nsecond",
        tool_call_id="call_second",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="user", content="old request " * 20),
            assistant,
            first_result,
            second_result,
        ]
    )

    context = build_model_context(conversation, context_budget(10))

    assert context.messages == (assistant, first_result, second_result)
    assert context.estimate.over_budget is True


def test_build_model_context_drops_incomplete_tool_call_group() -> None:
    first_call = AgentToolCall(
        id="call_first",
        name="read_file",
        arguments={"path": "first.py"},
    )
    second_call = AgentToolCall(
        id="call_second",
        name="read_file",
        arguments={"path": "second.py"},
    )
    conversation = Conversation.from_messages(
        [
            Message(role="user", content="before"),
            Message(
                role="assistant",
                content="",
                tool_calls=(first_call, second_call),
            ),
            Message(role="tool", content="OK\nfirst", tool_call_id="call_first"),
            Message(role="user", content="after"),
        ]
    )

    context = build_model_context(conversation, context_budget(1000))

    assert context.messages == (
        Message(role="user", content="before"),
        Message(role="user", content="after"),
    )
    assert context.trimmed_message_count == 2


def test_build_model_context_drops_orphan_tool_result() -> None:
    conversation = Conversation.from_messages(
        [
            Message(role="user", content="current"),
            Message(role="tool", content="OK\norphan", tool_call_id="missing"),
        ]
    )

    context = build_model_context(conversation, context_budget(1000))

    assert context.messages == (Message(role="user", content="current"),)
    assert context.trimmed_message_count == 1


def test_build_model_context_does_not_join_tool_chain_across_system_message() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            Message(role="system", content="new rules"),
            Message(role="tool", content="OK\nresult", tool_call_id="call_123"),
            Message(role="user", content="current"),
        ]
    )

    context = build_model_context(conversation, context_budget(1000))

    assert context.messages == (
        Message(role="system", content="new rules"),
        Message(role="user", content="current"),
    )
    assert context.trimmed_message_count == 2


def test_build_model_context_reports_oversized_latest_chinese_message() -> None:
    conversation = Conversation.from_messages(
        [Message(role="user", content="中" * 100)]
    )

    context = build_model_context(conversation, context_budget(60))

    assert context.messages == tuple(conversation.get_messages())
    assert context.estimate.estimated_input_tokens > context.estimate.max_input_tokens
    assert context.estimate.over_budget is True


def test_build_model_context_compresses_old_long_tool_result() -> None:
    old_tool_call = AgentToolCall(
        id="call_old",
        name="run_command",
        arguments={"command": ["python", "-m", "pytest"]},
    )
    recent_tool_call = AgentToolCall(
        id="call_recent",
        name="read_file",
        arguments={"path": "README.md"},
    )
    old_result = Message(
        role="tool",
        content=(
            "OK\n"
            "large output\n"
            + ("x" * 300)
            + '\n\nMETADATA\n{"exit_code": 0, "stdout": "'
            + ("y" * 300)
            + '", "stdout_truncated": true}'
        ),
        tool_call_id="call_old",
    )
    recent_result = Message(
        role="tool",
        content="OK\n1 | current\n\nMETADATA\n{\"path\": \"README.md\"}",
        tool_call_id="call_recent",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(old_tool_call,)),
            old_result,
            Message(role="assistant", content="", tool_calls=(recent_tool_call,)),
            recent_result,
        ]
    )

    context = build_model_context(
        conversation,
        context_budget(
            1200,
            tool_result_compression_threshold_chars=120,
        ),
    )

    assert context.compressed_tool_result_count == 1
    compressed_result = context.messages[1]
    assert compressed_result.role == "tool"
    assert compressed_result.tool_call_id == "call_old"
    assert "[tool result compressed]" in compressed_result.content
    assert "tool_name: run_command" in compressed_result.content
    assert "stdout_omitted" in compressed_result.content
    assert '"stdout_truncated": true' in compressed_result.content
    assert "y" * 80 not in compressed_result.content
    assert context.messages[-1] == recent_result


def test_build_model_context_can_compress_all_tool_results_when_configured() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    tool_result = Message(
        role="tool",
        content="OK\n" + ("x" * 200) + '\n\nMETADATA\n{"path": "README.md"}',
        tool_call_id="call_123",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            tool_result,
        ]
    )

    context = build_model_context(
        conversation,
        context_budget(
            50,
            tool_result_compression_threshold_chars=20,
            recent_tool_result_groups_to_keep=0,
        ),
    )

    assert context.compressed_tool_result_count == 1
    assert context.messages[1].tool_call_id == "call_123"
    assert "[tool result compressed]" in context.messages[1].content


def test_build_model_context_preserves_failure_preview_when_compressing() -> None:
    tool_call = AgentToolCall(
        id="call_failed",
        name="run_command",
        arguments={"command": ["python", "missing.py"]},
    )
    failure_reason = "IMPORTANT_FAILURE_REASON " + ("x" * 300)
    tool_result = Message(
        role="tool",
        content=(
            f"ERROR\n{failure_reason}"
            '\n\nMETADATA\n{"exit_code": 1, "stderr": "full stderr"}'
        ),
        tool_call_id="call_failed",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            tool_result,
        ]
    )

    context = build_model_context(
        conversation,
        context_budget(
            100,
            tool_result_compression_threshold_chars=20,
            recent_tool_result_groups_to_keep=0,
        ),
    )

    compressed_result = context.messages[1]
    assert '"error_preview": "IMPORTANT_FAILURE_REASON ' in compressed_result.content
    assert "stderr_omitted" in compressed_result.content
    assert "x" * 250 not in compressed_result.content


def test_build_model_context_does_not_compress_when_under_budget() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    tool_result = Message(
        role="tool",
        content="OK\n" + ("x" * 200) + '\n\nMETADATA\n{"path": "README.md"}',
        tool_call_id="call_123",
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            tool_result,
        ]
    )

    context = build_model_context(
        conversation,
        context_budget(
            1000,
            tool_result_compression_threshold_chars=20,
            recent_tool_result_groups_to_keep=0,
        ),
    )

    assert context.compressed_tool_result_count == 0
    assert context.messages[1] == tool_result


def test_build_model_context_injects_memory_without_mutating_conversation() -> None:
    conversation = Conversation.from_messages(
        [
            Message(role="system", content="rules"),
            Message(role="user", content="current request"),
        ]
    )
    memory_message = Message(role="system", content="untrusted memory data")
    memory_stats = MemoryContextStats(
        safe_entry_count=2,
        relevant_entry_count=1,
        selected_entry_count=1,
        estimated_tokens=24,
        irrelevant_entry_count=1,
        scopes=("project",),
    )

    context = build_model_context(
        conversation,
        context_budget(1000),
        memory_message=memory_message,
        memory_stats=memory_stats,
    )

    assert context.messages == (
        Message(role="system", content="rules"),
        memory_message,
        Message(role="user", content="current request"),
    )
    assert conversation.get_messages() == [
        Message(role="system", content="rules"),
        Message(role="user", content="current request"),
    ]
    assert context.memory_stats is not None
    assert context.memory_stats.included_entry_count == 1


def test_build_model_context_drops_memory_before_latest_request_on_overflow() -> None:
    conversation = Conversation.from_messages(
        [
            Message(role="system", content="rules"),
            Message(role="user", content="current request"),
        ]
    )
    memory_message = Message(role="system", content="memory " * 40)
    memory_stats = MemoryContextStats(
        safe_entry_count=1,
        relevant_entry_count=1,
        selected_entry_count=1,
        estimated_tokens=300,
        scopes=("project",),
    )

    context = build_model_context(
        conversation,
        context_budget(100),
        memory_message=memory_message,
        memory_stats=memory_stats,
    )

    assert context.messages == tuple(conversation.get_messages())
    assert context.estimate.over_budget is False
    assert context.memory_stats is not None
    assert context.memory_stats.included_entry_count == 0
    assert context.memory_stats.selected_entry_count == 1


def test_format_model_context_stats_reports_counts_without_memory_content() -> None:
    conversation = Conversation.from_messages(
        [Message(role="user", content="current")]
    )
    context = build_model_context(
        conversation,
        context_budget(1000),
        memory_message=Message(role="system", content="private memory text"),
        memory_stats=MemoryContextStats(
            safe_entry_count=4,
            relevant_entry_count=3,
            selected_entry_count=2,
            estimated_tokens=80,
            irrelevant_entry_count=1,
            conflict_count=1,
            budget_omitted_count=1,
            issue_count=1,
            scopes=("user", "project"),
        ),
    )

    summary = format_model_context_stats(context)

    assert "memory=2 injected/2 selected/3 relevant/4 safe" in summary
    assert "memory_scopes=user+project" in summary
    assert "memory_conflicts=1" in summary
    assert "memory_budget_omitted=1" in summary
    assert "memory_warnings=1" in summary
    assert "private memory text" not in summary


def test_build_model_context_rejects_non_system_memory_message() -> None:
    with pytest.raises(ValueError, match="system role"):
        build_model_context(
            Conversation(),
            context_budget(100),
            memory_message=Message(role="user", content="memory"),
        )
