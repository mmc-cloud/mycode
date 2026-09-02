from collections.abc import Iterator
import threading
import time

import pytest

from mycode.agent import (
    AgentEvent,
    AgentModelResponse,
    AgentProgressSnapshot,
    AgentToolCall,
)
from mycode.context_budget import (
    ContextBudget, MemoryContextStats, TokenUsage, estimate_conversation,
)
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.memory import MemoryStore
from mycode.memory_context import MemoryContextSelector, MemoryRecall
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.runner import (
    AgentRunner,
    DEFAULT_MAX_CONCURRENT_SAFE_TOOLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
    EMPTY_RESPONSE_RETRY_PROMPT,
    ToolBatchExecution,
    ToolBatchContractError,
    ToolCallExecution,
    execute_tool_batch,
    format_tool_result,
)
from mycode.run_progress import MAIN_NEAR_LIMIT_PROMPT, MAX_TURNS_FINALIZATION_PROMPT
from mycode.permissions import ConfirmationResult, PermissionDecision
from mycode.tools import (
    BaseTool,
    ReadFileTool,
    ToolArgs,
    ToolRegistry,
    ToolResult,
    Workspace,
)
from mycode.tools.read_file import MAX_LINES_LIMIT


def context_budget(max_input_tokens: int) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )


def _run_response(runner: AgentRunner, user_message: str) -> AgentModelResponse:
    current_turn_content: list[str] = []
    for event in runner.run(user_message):
        if event.type in {"turn", "model_start"}:
            current_turn_content = []
        elif event.type == "text_delta":
            current_turn_content.append(event.content)
        elif event.type == "error":
            current_turn_content.append(event.error or "")
        elif event.type == "stop":
            return AgentModelResponse(
                content="".join(current_turn_content) or event.content,
                stop_reason=event.stop_reason or "model_error",
            )
    raise AssertionError("AgentRunner.run() ended without a stop event")


def test_runner_returns_final_answer_without_tools() -> None:
    observations: list[dict[str, object]] = []
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="final answer")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        observability_sink=observations.append,
    )

    response = _run_response(runner, "hello")

    assert response == AgentModelResponse(content="final answer")
    assert runner.conversation.get_messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="final answer"),
    ]
    assert [record["event_type"] for record in observations] == [
        "context_snapshot",
        "model_response",
    ]
    assert all(record["schema_version"] == 1 for record in observations)
    model_response = observations[1]
    assert model_response["empty_response"] is False
    assert model_response["content_chars"] == len("final answer")


def test_runner_executes_tool_calls_when_content_is_empty() -> None:
    tool_call = AgentToolCall(
        id="call_tool",
        name="fake_tool",
        arguments={"text": "inspect"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls"),
            AgentModelResponse(content="done"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=_successful_batch,
    )

    assert _run_response(runner, "inspect").content == "done"
    assert len(llm_client.seen_conversations) == 2
    assert runner.last_runtime_state is not None
    assert runner.last_runtime_state.last_tool_observation is not None


@pytest.mark.parametrize("empty_content", ["", None, "   \t"])
def test_runner_retries_once_after_empty_response(empty_content) -> None:
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(content=empty_content),
            AgentModelResponse(content="recovered final"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        max_turns=1,
        near_limit_remaining_turns=None,
        near_limit_prompt=None,
    )

    events = list(runner.run("continue"))

    assert events[-1] == AgentEvent(type="stop", stop_reason="final_answer")
    assert [event.type for event in events].count("turn") == 1
    assert [event.type for event in events].count("model_start") == 2
    assert len(llm_client.seen_conversations) == 2
    retry_system_messages = [
        str(message["content"])
        for message in llm_client.seen_conversations[1]
        if message["role"] == "system"
    ]
    assert EMPTY_RESPONSE_RETRY_PROMPT in retry_system_messages
    assert runner.conversation.get_messages() == [
        Message(role="user", content="continue"),
        Message(role="assistant", content="recovered final"),
    ]


def test_runner_retries_empty_response_to_tool_call_in_last_turn() -> None:
    tool_call = AgentToolCall(
        id="call_tool",
        name="fake_tool",
        arguments={"text": "inspect"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(content=""),
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            ),
        ],
        plain_responses=["bounded final"],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=1,
        near_limit_remaining_turns=None,
        near_limit_prompt=None,
    )

    events = list(runner.run("inspect"))

    assert [event.type for event in events].count("turn") == 1
    assert len(llm_client.seen_conversations) == 2
    assert runner.last_runtime_state is not None
    assert runner.last_runtime_state.last_tool_observation is not None
    assert events[-1].stop_reason == "max_turns"


def test_runner_stops_with_empty_response_error_after_second_empty() -> None:
    observations: list[dict[str, object]] = []
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(content=""),
            AgentModelResponse(content="   "),
            AgentModelResponse(content="must not be requested"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        observability_sink=observations.append,
        max_turns=1,
        near_limit_remaining_turns=None,
        near_limit_prompt=None,
    )

    events = list(runner.run("continue"))

    assert events[-1].type == "stop"
    assert events[-1].stop_reason == "model_error"
    assert any(
        event.type == "error" and "empty_response" in (event.error or "")
        for event in events
    )
    assert len(llm_client.seen_conversations) == 2
    assert len(llm_client.responses) == 1
    assert [event.type for event in events].count("turn") == 1
    assert [event.type for event in events].count("model_start") == 2
    assert runner.conversation.get_messages() == [
        Message(role="user", content="continue")
    ]
    model_responses = [
        record for record in observations if record["event_type"] == "model_response"
    ]
    assert len(model_responses) == 2
    assert all(record["empty_response"] is True for record in model_responses)
    assert model_responses[-1]["error_type"] == "empty_response"


def test_runner_reasoning_without_content_or_tools_is_empty_response() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="final after retry")],
        stream_events=[
            [
                AgentEvent(
                    type="reasoning_delta",
                    reasoning_content="private reasoning",
                )
            ]
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    assert _run_response(runner, "continue").content == "final after retry"
    assert len(llm_client.seen_conversations) == 2


def test_runner_accepts_final_immediately_after_tracked_mutation() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(content="first final"),
            AgentModelResponse(content="must not be requested"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool(), FakeTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    response = _run_response(runner, "make a change")

    assert response.content == "first final"
    assert len(llm_client.seen_conversations) == 2
    assert len(llm_client.responses) == 1
    assert runner.last_runtime_state is not None
    assert runner.last_runtime_state.last_tool_observation is not None
    assert runner.last_runtime_state.last_tool_observation.mutation == "yes"
    assert runner.last_runtime_state.last_reason == "final_answer"
    assert runner.conversation.get_messages()[-1] == Message(
        role="assistant",
        content="first final",
    )
    assert all(
        {schema["name"] for schema in tools}
        == {"write_file", "run_validation", "fake_tool"}
        for tools in llm_client.seen_tools
    )
def test_runner_convergence_guidance_never_removes_investigation_tools() -> None:
    calls = [
        AgentToolCall(id="first", name="fake_tool", arguments={"text": "a"}),
        AgentToolCall(id="second", name="fake_tool", arguments={"text": "b"}),
        AgentToolCall(id="third", name="fake_tool", arguments={"text": "c"}),
    ]
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[calls[0]], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[calls[1]], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[calls[2]], stop_reason="tool_calls"),
            AgentModelResponse(content="replanned"),
        ]
    )

    def same_result_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=call,
                    result=ToolResult.success("same result"),
                )
                for call in tool_calls
            )
        )

    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=same_result_batch,
    )

    assert _run_response(runner, "investigate").content == "replanned"
    assert all(
        [schema["name"] for schema in tools] == ["fake_tool"]
        for tools in llm_client.seen_tools
    )
    fourth_system_messages = [
        str(message["content"])
        for message in llm_client.seen_conversations[3]
        if message["role"] == "system"
    ]
    assert any("重复主导的停滞特征" in message for message in fourth_system_messages)


def test_execute_tool_batch_runs_adjacent_safe_tools_concurrently_in_order() -> None:
    probe = ActivityProbe()
    first = InstrumentedTool("first", probe)
    second = InstrumentedTool("second", probe)
    registry = ToolRegistry.from_tools([first, second])
    tool_calls = [
        AgentToolCall(id="call-second", name="second", arguments={"text": "2"}),
        AgentToolCall(id="call-first", name="first", arguments={"text": "1"}),
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert [execution.tool_call.id for execution in batch.executions] == [
        "call-second",
        "call-first",
    ]
    assert [execution.result.content for execution in batch.executions] == [
        "second:2",
        "first:1",
    ]
    assert probe.max_active == 2


def test_execute_tool_batch_caps_safe_concurrency_per_chunk() -> None:
    probe = ActivityProbe()
    tools = [
        InstrumentedTool(f"safe_{index}", probe)
        for index in range(DEFAULT_MAX_CONCURRENT_SAFE_TOOLS * 3 + 1)
    ]
    registry = ToolRegistry.from_tools(tools)
    tool_calls = [
        AgentToolCall(
            id=f"call-{index}",
            name=f"safe_{index}",
            arguments={"text": str(index)},
        )
        for index in range(len(tools))
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert len(batch.executions) == len(tool_calls)
    assert probe.max_active == DEFAULT_MAX_CONCURRENT_SAFE_TOOLS


def test_execute_tool_batch_preserves_order_across_safe_chunks() -> None:
    probe = ActivityProbe()
    tools = [InstrumentedTool(f"ordered_{index}", probe) for index in range(9)]
    registry = ToolRegistry.from_tools(tools)
    tool_calls = [
        AgentToolCall(
            id=f"call-{index}",
            name=f"ordered_{index}",
            arguments={"text": str(index)},
        )
        for index in range(len(tools))
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert [execution.tool_call.id for execution in batch.executions] == [
        call.id for call in tool_calls
    ]
    assert [execution.result.content for execution in batch.executions] == [
        f"ordered_{index}:{index}" for index in range(len(tools))
    ]


def test_execute_tool_batch_keeps_unsafe_tools_as_serial_barriers() -> None:
    probe = ActivityProbe()
    before_one = InstrumentedTool("before_one", probe)
    before_two = InstrumentedTool("before_two", probe)
    serial = InstrumentedTool("serial", probe, concurrency_safe=False)
    after_one = InstrumentedTool("after_one", probe)
    after_two = InstrumentedTool("after_two", probe)
    registry = ToolRegistry.from_tools(
        [before_one, before_two, serial, after_one, after_two]
    )
    tool_calls = [
        AgentToolCall(id="call-before-1", name="before_one", arguments={"text": "1"}),
        AgentToolCall(id="call-before-2", name="before_two", arguments={"text": "2"}),
        AgentToolCall(id="call-serial", name="serial", arguments={"text": "3"}),
        AgentToolCall(id="call-after-1", name="after_one", arguments={"text": "4"}),
        AgentToolCall(id="call-after-2", name="after_two", arguments={"text": "5"}),
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert [execution.tool_call.id for execution in batch.executions] == [
        call.id for call in tool_calls
    ]
    assert probe.serial_overlap is False
    assert probe.safe_started_during_serial is False
    assert probe.max_active == 2


def test_execute_tool_batch_does_not_cancel_sibling_after_tool_failure() -> None:
    probe = ActivityProbe()
    failing = InstrumentedTool("failing", probe, fail=True)
    succeeding = InstrumentedTool("succeeding", probe)
    registry = ToolRegistry.from_tools([failing, succeeding])
    tool_calls = [
        AgentToolCall(id="call-failing", name="failing", arguments={"text": "x"}),
        AgentToolCall(id="call-succeeding", name="succeeding", arguments={"text": "y"}),
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert batch.executions[0].result.ok is False
    assert batch.executions[0].result.error == "synthetic failure"
    assert batch.executions[1].result == ToolResult.success("succeeding:y")
    assert set(probe.calls) == {"failing", "succeeding"}


def test_execute_tool_batch_converts_real_async_exception_without_canceling_sibling() -> None:
    probe = ActivityProbe()
    raising = InstrumentedTool("raising", probe)
    succeeding = InstrumentedTool("succeeding_async", probe)
    registry = RaisingAsyncRegistry.from_tools([raising, succeeding])

    batch = execute_tool_batch(
        registry,
        [
            AgentToolCall(id="call-raising", name="raising", arguments={"text": "x"}),
            AgentToolCall(
                id="call-succeeding",
                name="succeeding_async",
                arguments={"text": "y"},
            ),
        ],
    )

    assert batch.executions[0].result == ToolResult.failure(
        error="Tool execution failed: synthetic async crash",
        metadata={"exception_type": "RuntimeError"},
    )
    assert batch.executions[1].result == ToolResult.success("succeeding_async:y")
    assert probe.calls == ["succeeding_async"]


def test_execute_tool_batch_returns_failures_for_all_calls_over_total_limit() -> None:
    probe = ActivityProbe()
    tools = [
        InstrumentedTool(f"limited_{index}", probe)
        for index in range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE + 5)
    ]
    registry = ToolRegistry.from_tools(tools)
    tool_calls = [
        AgentToolCall(
            id=f"call-{index}",
            name=f"limited_{index}",
            arguments={"text": str(index)},
        )
        for index in range(len(tools))
    ]

    batch = execute_tool_batch(registry, tool_calls)

    assert [execution.tool_call.id for execution in batch.executions] == [
        call.id for call in tool_calls
    ]
    assert all(
        execution.result.ok
        for execution in batch.executions[:DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE]
    )
    overflow = batch.executions[DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE:]
    assert len(overflow) == 5
    assert all(not execution.result.ok for execution in overflow)
    assert [
        execution.result.metadata["tool_call_index"]
        for execution in overflow
    ] == list(range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE, len(tool_calls)))
    assert len(probe.calls) == DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE


def test_concurrent_permission_interaction_is_serialized() -> None:
    probe = ActivityProbe()
    checker = AskingPermissionChecker()
    confirmer = SerialConfirmer()
    registry = ToolRegistry.from_tools(
        [InstrumentedTool("first", probe), InstrumentedTool("second", probe)],
        permission_checker=checker,
        confirmer=confirmer,
    )

    batch = execute_tool_batch(
        registry,
        [
            AgentToolCall(id="call-first", name="first", arguments={"text": "1"}),
            AgentToolCall(id="call-second", name="second", arguments={"text": "2"}),
        ],
    )

    assert all(execution.result.ok for execution in batch.executions)
    assert checker.max_active == 1
    assert confirmer.max_active == 1


def test_runner_executes_tool_and_returns_second_model_answer() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
                reasoning_content="private synthetic reasoning",
            ),
            AgentModelResponse(content="final answer"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    response = _run_response(runner, "use a tool")

    assert response == AgentModelResponse(content="final answer")
    assert runner.conversation.get_messages() == [
        Message(role="user", content="use a tool"),
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
        Message(
            role="tool",
            content='OK\nfake result: hello\n\nMETADATA\n{"text": "hello"}',
            tool_call_id="call_123",
        ),
        Message(role="assistant", content="final answer"),
    ]
    assert llm_client.seen_conversations[1][1]["reasoning_content"] == (
        "private synthetic reasoning"
    )


def test_runner_accumulates_token_usage_across_internal_tool_turns() -> None:
    tool_call = AgentToolCall(
        id="call_usage",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls"),
            AgentModelResponse(content="done"),
            AgentModelResponse(content="next run"),
        ],
        token_usages=[
            TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            TokenUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        ],
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    _run_response(runner, "measure usage")

    assert runner.last_token_usage == TokenUsage(
        prompt_tokens=20,
        completion_tokens=3,
        total_tokens=23,
    )
    assert runner.run_token_usage == TokenUsage(
        prompt_tokens=30,
        completion_tokens=5,
        total_tokens=35,
    )

    _run_response(runner, "measure the next run")

    assert runner.run_token_usage == TokenUsage(
        prompt_tokens=5,
        completion_tokens=1,
        total_tokens=6,
    )


def test_runner_honors_custom_tool_batch_stop_response() -> None:
    tool_call = AgentToolCall(
        id="call_control",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls")]
    )

    def stop_after_batch(registry, tool_calls):
        assert registry.require("fake_tool").name == "fake_tool"
        assert tool_calls == [tool_call]
        return ToolBatchExecution(
            executions=(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success(content="controlled"),
                ),
            ),
            stop_response=AgentModelResponse(
                content="control complete",
                stop_reason="control_tool",
            ),
        )

    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=stop_after_batch,
    )

    response = _run_response(runner, "use control")

    assert response.stop_reason == "control_tool"
    assert runner.conversation.get_messages()[-1] == Message(
        role="tool",
        content="OK\ncontrolled\n\nMETADATA\n{}",
        tool_call_id="call_control",
    )


def test_runner_rejects_incomplete_custom_tool_batch() -> None:
    tool_call = AgentToolCall(
        id="call_missing_result",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    runner = AgentRunner(
        llm_client=RecordingLLMClient(
            responses=[
                AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls")
            ]
        ),
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=lambda registry, tool_calls: ToolBatchExecution(
            executions=()
        ),
    )

    with pytest.raises(ToolBatchContractError, match="exactly one ordered result"):
        _run_response(runner, "use control")


def test_runner_passes_tool_schemas_to_llm() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="final answer")]
    )
    registry = ToolRegistry.from_tools([FakeTool()])
    runner = AgentRunner(llm_client=llm_client, tool_registry=registry)

    _run_response(runner, "hello")

    assert llm_client.seen_tools == [
        [
            {
                "name": "fake_tool",
                "description": "Fake tool.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                    "type": "object",
                },
            }
        ]
    ]


def test_runner_uses_model_context_without_trimming_history() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="final answer")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 20),
                Message(role="assistant", content="old reply " * 20),
            ]
        ),
        context_budget=context_budget(100),
    )

    _run_response(runner, "current")

    assert llm_client.seen_conversations[0] == [
        {"role": "user", "content": "current"}
    ]
    assert runner.last_model_context is not None
    assert runner.last_model_context.trimmed is True
    assert runner.last_model_context.trimmed_message_count == 2
    assert runner.conversation.get_messages() == [
        Message(role="user", content="old request " * 20),
        Message(role="assistant", content="old reply " * 20),
        Message(role="user", content="current"),
        Message(role="assistant", content="final answer"),
    ]


def test_runner_does_not_call_model_when_context_is_over_budget() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="unused")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        context_budget=context_budget(60),
    )

    response = _run_response(runner, "中" * 100)

    assert response.stop_reason == "context_overflow"
    assert "was not sent" in response.content
    assert llm_client.seen_conversations == []
    assert runner.last_model_context is not None
    assert runner.last_model_context.estimate.over_budget is True


def test_runner_does_not_call_model_when_tool_schema_exceeds_budget() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="unused")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        context_budget=context_budget(50),
    )

    response = _run_response(runner, "hi")

    assert response.stop_reason == "context_overflow"
    assert llm_client.seen_conversations == []
    assert runner.last_model_context is not None
    assert runner.last_model_context.estimate.tool_schema_chars > 100


def test_runner_writes_missing_tool_failure_back_to_conversation() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="missing",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls"),
            AgentModelResponse(content="saw the error"),
        ]
    )
    runner = AgentRunner(llm_client=llm_client, tool_registry=ToolRegistry())

    response = _run_response(runner, "use a missing tool")

    assert response == AgentModelResponse(content="saw the error")
    assert runner.conversation.get_messages()[2] == Message(
        role="tool",
        content='ERROR\nTool not found: missing\n\nMETADATA\n{"tool_name": "missing"}',
        tool_call_id="call_123",
    )


def test_runner_returns_model_error_without_executing_tools() -> None:
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(
                content="bad tool arguments",
                stop_reason="model_error",
            )
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    response = _run_response(runner, "hello")

    assert response == AgentModelResponse(
        content="bad tool arguments",
        stop_reason="model_error",
    )
    assert runner.conversation.get_messages() == [
        Message(role="user", content="hello"),
    ]


def test_runner_returns_model_error_when_model_request_raises() -> None:
    runner = AgentRunner(
        llm_client=FailingLLMClient(error=RuntimeError("network down")),
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    response = _run_response(runner, "hello")

    assert response == AgentModelResponse(
        content="模型流式请求失败：network down",
        stop_reason="model_error",
    )


def test_runner_tracked_mutations_do_not_extend_strict_max_turns() -> None:
    first_tool_call = AgentToolCall(
        id="call_123",
        name="write_file",
        arguments={"text": "first"},
    )
    second_tool_call = AgentToolCall(
        id="call_456",
        name="write_file",
        arguments={"text": "second"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[first_tool_call], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[second_tool_call], stop_reason="tool_calls"),
        ],
        plain_responses=["同步阶段性结论。"],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        max_turns=2,
        near_limit_remaining_turns=None,
        near_limit_prompt=None,
    )

    response = _run_response(runner, "loop")

    assert response == AgentModelResponse(
        content="同步阶段性结论。",
        stop_reason="max_turns",
    )
    assert len(llm_client.seen_conversations) == 2
    assert len(llm_client.seen_plain_conversations) == 1
    assert runner.conversation.get_messages()[-1].content.endswith("同步阶段性结论。")
    assert runner.last_runtime_state is not None


def test_runner_default_max_turns_allows_bounded_project_exploration() -> None:
    assert DEFAULT_MAX_TURNS == 50
    assert AgentRunner(
        llm_client=FakeLLMClient(responses=[]),
        tool_registry=ToolRegistry(),
    ).max_turns == 50


@pytest.mark.parametrize(
    "options",
    [
        {"near_limit_remaining_turns": None},
        {"near_limit_prompt": None},
        {"near_limit_remaining_turns": 0},
        {"near_limit_remaining_turns": 50},
        {"near_limit_prompt": "   "},
    ],
)
def test_runner_rejects_invalid_near_limit_configuration(options) -> None:
    with pytest.raises(ValueError, match="near_limit_"):
        AgentRunner(
            llm_client=FakeLLMClient(responses=[]),
            tool_registry=ToolRegistry(),
            **options,
        )


def test_runner_stops_on_repeated_tool_call() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "same"},
    )
    repeated_tool_call = AgentToolCall(
        id="call_456",
        name="fake_tool",
        arguments={"text": "same"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls"),
            AgentModelResponse(
                tool_calls=[repeated_tool_call],
                stop_reason="tool_calls",
            ),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        repeated_tool_call_limit=2,
    )

    response = _run_response(runner, "repeat")

    assert response.stop_reason == "repeated_tool_call"
    assert "repeated the same tool call" in response.content


def test_runner_reports_near_limit_guidance_once_at_exactly_five_turns() -> None:
    tool_calls = [
        AgentToolCall(
            id=f"call_{index}",
            name="fake_tool",
            arguments={"text": f"inspect-{index}"},
        )
        for index in range(1, 6)
    ]
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            *(
                [AgentEvent(type="tool_call", tool_call=tool_call)]
                for tool_call in tool_calls
            ),
            [AgentEvent(type="text_delta", content="done")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        max_turns=6,
    )

    events = list(runner.run("inspect"))

    turn_events = [event for event in events if event.type == "turn"]
    assert len(turn_events) == 6
    assert [event.content for event in turn_events] == [
        "",
        "距离运行轮次上限还有 5 轮，已发送一次接近上限提醒",
        "",
        "",
        "",
        "",
    ]
    prompt_counts = [
        sum(
            message["role"] == "system"
            and MAIN_NEAR_LIMIT_PROMPT in str(message["content"])
            for message in context
        )
        for context in llm_client.seen_conversations
    ]
    assert prompt_counts == [0, 1, 0, 0, 0, 0]


def test_empty_response_retry_does_not_recalculate_near_limit_guidance() -> None:
    tool_call = AgentToolCall(
        id="call_1",
        name="fake_tool",
        arguments={"text": "inspect"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content=""),
            AgentModelResponse(content="recovered final"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=2,
        near_limit_remaining_turns=1,
        near_limit_prompt=MAIN_NEAR_LIMIT_PROMPT,
    )

    events = list(runner.run("inspect"))

    turn_events = [event for event in events if event.type == "turn"]
    assert len(turn_events) == 2
    assert sum(bool(event.content) for event in turn_events) == 1
    assert [event.type for event in events].count("model_start") == 3
    prompt_counts = [
        sum(
            message["role"] == "system"
            and MAIN_NEAR_LIMIT_PROMPT in str(message["content"])
            for message in context
        )
        for context in llm_client.seen_conversations
    ]
    assert prompt_counts == [0, 1, 1]


def _successful_batch(registry, tool_calls):
    return ToolBatchExecution(
        executions=tuple(
            ToolCallExecution(
                tool_call=tool_call,
                result=ToolResult.success(f"completed {tool_call.name}"),
            )
            for tool_call in tool_calls
        )
    )


def test_runner_stream_continues_from_persisted_max_turns_checkpoint() -> None:
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[[AgentEvent(type="text_delta", content="continued")]],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="original task"),
                Message(
                    role="assistant",
                    content="## 续跑检查点\n\n剩余动作：修改并测试。",
                ),
            ]
        ),
    )

    events = list(runner.run("继续"))

    first_turn = next(event for event in events if event.type == "turn")
    assert "已载入上次检查点" in first_turn.content
    assert any(
        message["role"] == "system" and "直接使用上一条“续跑检查点”" in str(message["content"])
        for message in llm_client.seen_conversations[0]
    )
    assert runner.conversation.get_messages()[-2:] == [
        Message(role="user", content="继续"),
        Message(role="assistant", content="continued"),
    ]


def test_runner_continues_from_persisted_max_turns_checkpoint() -> None:
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="continued")],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="original task"),
                Message(
                    role="assistant",
                    content="## 续跑检查点\n\n剩余动作：修改并测试。",
                ),
            ]
        ),
    )

    response = _run_response(runner, "继续")

    assert response == AgentModelResponse(content="continued")
    assert any(
        message["role"] == "system" and "直接使用上一条“续跑检查点”" in str(message["content"])
        for message in llm_client.seen_conversations[0]
    )


def test_runner_streams_final_answer_text_and_stop_event() -> None:
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(type="text_delta", content="hello"),
                AgentEvent(type="text_delta", content=" world"),
            ],
        ],
    )
    runner = AgentRunner(llm_client=llm_client, tool_registry=ToolRegistry())

    events = list(runner.run("hello"))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert "tokens" in events[1].content
    assert "conservative estimate" in events[1].content
    assert events[2:] == [
        AgentEvent(type="model_start"),
        AgentEvent(type="text_delta", content="hello"),
        AgentEvent(type="text_delta", content=" world"),
        AgentEvent(
            type="progress",
            progress=AgentProgressSnapshot(
                reason="final_answer",
            ),
        ),
        AgentEvent(type="stop", stop_reason="final_answer"),
    ]
    assert runner.conversation.get_messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hello world"),
    ]


def test_runner_stream_uses_model_context_without_trimming_history() -> None:
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(type="text_delta", content="final answer"),
            ],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 20),
                Message(role="assistant", content="old reply " * 20),
            ]
        ),
        context_budget=context_budget(100),
    )

    events = list(runner.run("current"))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert "messages=1/3" in events[1].content
    assert "trimmed=2" in events[1].content
    assert "old request" not in events[1].content
    assert "old reply" not in events[1].content
    assert llm_client.seen_conversations[0] == [
        {"role": "user", "content": "current"}
    ]
    assert runner.last_model_context is not None
    assert runner.last_model_context.trimmed is True
    assert runner.conversation.get_messages() == [
        Message(role="user", content="old request " * 20),
        Message(role="assistant", content="old reply " * 20),
        Message(role="user", content="current"),
        Message(role="assistant", content="final answer"),
    ]


def test_runner_stream_does_not_call_model_when_context_is_over_budget() -> None:
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[[AgentEvent(type="text_delta", content="unused")]],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        context_budget=context_budget(60),
    )

    events = list(runner.run("中" * 100))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert "over_budget=True" in events[1].content
    assert events[2].type == "error"
    assert "was not sent" in (events[2].error or "")
    assert events[3] == AgentEvent(type="stop", stop_reason="context_overflow")
    assert llm_client.seen_conversations == []


def test_runner_streams_tool_call_tool_result_and_second_model_answer() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    tool_call_event = AgentEvent(type="tool_call", tool_call=tool_call)
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(
                    type="reasoning_delta",
                    reasoning_content="private synthetic reasoning",
                ),
                tool_call_event,
            ],
            [AgentEvent(type="text_delta", content="final answer")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    events = list(runner.run("use a tool"))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert events[2:] == [
        AgentEvent(type="model_start"),
        tool_call_event,
        AgentEvent(
            type="tool_result",
            tool_result=ToolResult.success(
                content="fake result: hello",
                metadata={"text": "hello"},
            ),
        ),
        AgentEvent(
            type="progress",
            progress=AgentProgressSnapshot(
                same_tool_repeat=1,
                same_result_repeat=1,
                reason="run_started",
            ),
        ),
        AgentEvent(type="turn", turn_number=2, max_turns=50),
        AgentEvent(type="model_start"),
        AgentEvent(type="text_delta", content="final answer"),
        AgentEvent(
            type="progress",
            progress=AgentProgressSnapshot(
                same_tool_repeat=1,
                same_result_repeat=1,
                reason="final_answer",
            ),
        ),
        AgentEvent(type="stop", stop_reason="final_answer"),
    ]
    assert runner.conversation.get_messages() == [
        Message(role="user", content="use a tool"),
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
        Message(
            role="tool",
            content='OK\nfake result: hello\n\nMETADATA\n{"text": "hello"}',
            tool_call_id="call_123",
        ),
        Message(role="assistant", content="final answer"),
    ]
    assert all(event.type != "reasoning_delta" for event in events)
    assert llm_client.seen_conversations[1][1]["reasoning_content"] == (
        "private synthetic reasoning"
    )


def test_runner_stream_observes_normalized_args_but_replays_provider_call(
    tmp_path,
) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    raw_tool_call = AgentToolCall(
        id="call_bounded_read",
        name="read_file",
        arguments={
            "path": "sample.txt",
            "max_lines": MAX_LINES_LIMIT + 500,
        },
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=raw_tool_call)],
            [AgentEvent(type="text_delta", content="done")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [ReadFileTool(Workspace(tmp_path))]
        ),
    )

    events = list(runner.run("read the file"))

    observable_call = next(
        event.tool_call
        for event in events
        if event.type == "tool_call" and event.tool_call is not None
    )
    result = next(
        event.tool_result
        for event in events
        if event.type == "tool_result" and event.tool_result is not None
    )
    assistant_call = runner.conversation.get_messages()[1].tool_calls[0]
    assert observable_call.arguments["max_lines"] == MAX_LINES_LIMIT
    assert result.metadata["start_line"] == 1
    assert result.metadata["end_line"] == 1
    assert result.metadata["has_more"] is False
    assert assistant_call == raw_tool_call


def test_runner_stream_replays_present_empty_reasoning_as_null() -> None:
    tool_call = AgentToolCall(
        id="call_empty_reasoning",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(
                    type="reasoning_state",
                    reasoning_state="present_empty",
                ),
                AgentEvent(type="tool_call", tool_call=tool_call),
            ],
            [AgentEvent(type="text_delta", content="done")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    events = list(runner.run("use a tool"))

    assert events[-1] == AgentEvent(type="stop", stop_reason="final_answer")
    tool_message = runner.conversation.get_messages()[1]
    assert tool_message.reasoning_state == "present_empty"
    assert tool_message.reasoning_content is None
    assert llm_client.seen_conversations[1][1]["reasoning_content"] is None


def test_runner_stream_supports_fifty_present_empty_reasoning_tool_turns() -> None:
    tool_calls = [
        AgentToolCall(
            id=f"call-empty-{index}",
            name="fake_tool",
            arguments={"text": str(index)},
        )
        for index in range(50)
    ]
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(
                    type="reasoning_state",
                    reasoning_state="present_empty",
                ),
                AgentEvent(type="tool_call", tool_call=tool_call),
            ]
            for tool_call in tool_calls
        ]
        + [[AgentEvent(type="text_delta", content="done")]],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeWriteTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=51,
    )

    events = list(runner.run("run fifty steps"))

    assert events[-1] == AgentEvent(type="stop", stop_reason="final_answer")
    assert sum(event.type == "tool_call" for event in events) == 50
    assert sum(event.type == "tool_result" for event in events) == 50
    for conversation in llm_client.seen_conversations[1:]:
        assistant_tool_messages = [
            message
            for message in conversation
            if message["role"] == "assistant" and "tool_calls" in message
        ]
        assert assistant_tool_messages
        assert all(
            message["reasoning_content"] is None
            for message in assistant_tool_messages
        )


def test_runner_stream_honors_custom_tool_batch_stop_response() -> None:
    tool_call = AgentToolCall(
        id="call_control_stream",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    client = RecordingLLMClient(
        responses=[],
        stream_events=[[AgentEvent(type="tool_call", tool_call=tool_call)]],
    )

    def stop_after_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=(
                ToolCallExecution(
                    tool_call=tool_calls[0],
                    result=registry.run_tool(
                        tool_calls[0].name,
                        tool_calls[0].arguments,
                    ),
                ),
            ),
            stop_response=AgentModelResponse(
                content="control complete",
                stop_reason="control_tool",
            ),
        )

    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        tool_batch_handler=stop_after_batch,
    )

    events = list(runner.run("use control"))

    assert events[-3] == AgentEvent(
        type="tool_result",
        tool_result=ToolResult.success(
            content="fake result: hello",
            metadata={"text": "hello"},
        ),
    )
    assert events[-2] == AgentEvent(
        type="progress",
        progress=AgentProgressSnapshot(
            same_tool_repeat=1,
            same_result_repeat=1,
            reason="run_started",
        ),
    )
    assert events[-1] == AgentEvent(
        type="stop",
        content="control complete",
        stop_reason="control_tool",
    )


def test_runner_stream_emits_context_notice_once_per_user_turn() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=tool_call)],
            [AgentEvent(type="text_delta", content="final answer")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 100),
                Message(role="assistant", content="old reply " * 100),
            ]
        ),
        context_budget=context_budget(400),
    )

    events = list(runner.run("current"))

    assert sum(event.type == "context" for event in events) == 1
    assert len(llm_client.seen_conversations) == 2
    assert llm_client.responses == []
    assert events[-1] == AgentEvent(type="stop", stop_reason="final_answer")


def test_runner_stream_updates_context_when_later_turn_overflows() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "hello"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=tool_call)],
            [AgentEvent(type="text_delta", content="unused")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        context_budget=context_budget(150),
    )

    events = list(runner.run("use a tool"))
    context_events = [event for event in events if event.type == "context"]

    assert len(context_events) == 2
    assert "over_budget=False" not in context_events[0].content
    assert "over_budget=True" in context_events[1].content
    assert events[-1] == AgentEvent(type="stop", stop_reason="context_overflow")
    assert len(llm_client.seen_conversations) == 1


def test_runner_stream_reports_previous_actual_input_on_next_user_turn() -> None:
    usage = TokenUsage(prompt_tokens=80, completion_tokens=10, total_tokens=90)
    client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(content="first"),
            AgentModelResponse(content="second"),
        ],
        token_usages=[usage, None],
    )
    runner = AgentRunner(llm_client=client, tool_registry=ToolRegistry())

    list(runner.run("a" * 100))
    second_events = list(runner.run("next"))

    assert runner.last_model_context is not None
    assert second_events[0] == AgentEvent(
        type="turn", turn_number=1, max_turns=50
    )
    assert second_events[1].type == "context"
    assert "calibrated estimate" in second_events[1].content
    assert "messages=3/3" in second_events[1].content
    assert "previous_actual_input=80 tokens" in second_events[1].content


def test_runner_streams_model_error_and_stop_event() -> None:
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="error", error="invalid tool arguments")],
        ],
    )
    runner = AgentRunner(llm_client=llm_client, tool_registry=ToolRegistry())

    events = list(runner.run("hello"))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert events[2:] == [
        AgentEvent(type="model_start"),
        AgentEvent(type="error", error="invalid tool arguments"),
        AgentEvent(type="stop", stop_reason="model_error"),
    ]


def test_runner_streams_model_error_when_streaming_raises() -> None:
    runner = AgentRunner(
        llm_client=FailingLLMClient(error=RuntimeError("stream broke")),
        tool_registry=ToolRegistry(),
    )

    events = list(runner.run("hello"))

    assert events[0] == AgentEvent(type="turn", turn_number=1, max_turns=50)
    assert events[1].type == "context"
    assert events[2:] == [
        AgentEvent(type="model_start"),
        AgentEvent(type="error", error="模型流式请求失败：stream broke"),
        AgentEvent(type="stop", stop_reason="model_error"),
    ]


def test_runner_stream_stops_after_max_turns(monkeypatch) -> None:
    import mycode.context_builder as builder_module

    requests = []
    real_budget = builder_module.budget_model_context

    def budget(messages, *args, **kwargs):
        result = real_budget(messages, *args, **kwargs)
        requests.append((messages, kwargs["tools"], result))
        return result

    monkeypatch.setattr(builder_module, "budget_model_context", budget)
    first_tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "first"},
    )
    second_tool_call = AgentToolCall(
        id="call_456",
        name="fake_tool",
        arguments={"text": "second"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=first_tool_call)],
            [AgentEvent(type="tool_call", tool_call=second_tool_call)],
        ],
        plain_responses=["根据现有信息得到阶段性结论。"],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        max_turns=2,
        near_limit_remaining_turns=None,
        near_limit_prompt=None,
    )

    events = list(runner.run("loop"))

    assert len(requests) == 3  # Two tool requests and one independent finalization.
    assert requests[-1][0][-1].content == MAX_TURNS_FINALIZATION_PROMPT
    assert requests[-1][1] == []
    assert runner.last_model_context.messages == requests[-1][2].messages
    assert runner.last_model_context.estimate == requests[-1][2].estimate
    assert events[-3:] == [
        AgentEvent(type="model_start"),
        AgentEvent(type="text_delta", content="根据现有信息得到阶段性结论。"),
        AgentEvent(
        type="stop",
            content="本轮已达到 2 轮上限；上面是基于现有信息整理的阶段性结果。",
        stop_reason="max_turns",
        ),
    ]
    assert llm_client.seen_plain_conversations[-1][-1]["role"] == "user"
    assert "不要请求或假设任何新的工具结果" in str(
        llm_client.seen_plain_conversations[-1][-1]["content"]
    )
    assert "已读取文件及范围" in str(
        llm_client.seen_plain_conversations[-1][-1]["content"]
    )
    assert runner.last_runtime_state is not None
    assert "数量、列表和文件名是否前后一致" in str(
        llm_client.seen_plain_conversations[-1][-1]["content"]
    )
    assert runner.conversation.get_messages()[-1] == Message(
        role="assistant",
        content="## 续跑检查点\n\n根据现有信息得到阶段性结论。",
    )


def test_runner_stream_stops_on_repeated_tool_call() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="fake_tool",
        arguments={"text": "same"},
    )
    repeated_tool_call = AgentToolCall(
        id="call_456",
        name="fake_tool",
        arguments={"text": "same"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=tool_call)],
            [AgentEvent(type="tool_call", tool_call=repeated_tool_call)],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        repeated_tool_call_limit=2,
    )

    events = list(runner.run("repeat"))

    assert events[-1].type == "stop"
    assert events[-1].stop_reason == "repeated_tool_call"
    assert "repeated the same tool call" in events[-1].content


def test_format_tool_result_formats_success_and_metadata() -> None:
    result = ToolResult.success(
        content="1 | hello",
        metadata={"path": "README.md", "truncated": False},
    )

    assert format_tool_result(result) == (
        'OK\n1 | hello\n\nMETADATA\n{"path": "README.md", "truncated": false}'
    )


def test_format_tool_result_formats_failure_and_metadata() -> None:
    result = ToolResult.failure(
        error="Invalid tool arguments",
        metadata={"tool_name": "fake_tool"},
    )

    assert format_tool_result(result) == (
        'ERROR\nInvalid tool arguments\n\nMETADATA\n{"tool_name": "fake_tool"}'
    )


def test_runner_recalls_current_disk_memory_without_persisting_it(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(
        ProjectIdentity.from_workspace(workspace),
        base_directory=tmp_path / "user-state",
    )
    store.save(
        scope="project",
        kind="fact",
        key="test.command",
        content="uv run pytest",
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(content="first answer"),
            AgentModelResponse(content="second answer"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        conversation=Conversation.from_messages(
            [
                Message(role="system", content="core and current project rules"),
                Message(role="user", content="restored request"),
                Message(role="assistant", content="restored answer"),
            ]
        ),
        memory_context_selector=MemoryContextSelector(store),
    )

    _run_response(runner, "What test command should I use?")

    first_model_messages = llm_client.seen_conversations[0]
    assert [message["role"] for message in first_model_messages[:2]] == [
        "system",
        "system",
    ]
    assert "uv run pytest" in str(first_model_messages[1]["content"])
    assert first_model_messages[2:] == [
        {"role": "user", "content": "restored request"},
        {"role": "assistant", "content": "restored answer"},
        {"role": "user", "content": "What test command should I use?"},
    ]
    assert all(
        "BEGIN_MYCODE_MEMORY_JSON" not in message.content
        for message in runner.conversation.get_messages()
    )

    path = store.path_for_scope("project")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "uv run pytest",
            "uv run pytest -q",
        ),
        encoding="utf-8",
    )
    _run_response(runner, "What test command should I use now?")

    second_model_messages = llm_client.seen_conversations[1]
    memory_contents = [
        str(message["content"])
        for message in second_model_messages
        if message["role"] == "system"
        and "BEGIN_MYCODE_MEMORY_JSON" in str(message["content"])
    ]
    assert len(memory_contents) == 1
    assert "uv run pytest -q" in memory_contents[0]


def test_runner_stream_reports_memory_counts_without_memory_content(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(
        ProjectIdentity.from_workspace(workspace),
        base_directory=tmp_path / "user-state",
    )
    store.save(
        scope="project",
        kind="fact",
        key="test.command",
        content="private synthetic command",
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [
                AgentEvent(type="text_delta", content="answer"),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        memory_context_selector=MemoryContextSelector(store),
    )

    events = list(runner.run("test command"))

    context_event = next(event for event in events if event.type == "context")
    assert "memory=1 injected/1 selected/1 relevant/1 safe" in context_event.content
    assert "private synthetic command" not in context_event.content


@pytest.mark.parametrize("omit_memory", [False, True])
def test_runner_budgets_guidance_and_memory_together(omit_memory, monkeypatch) -> None:
    import mycode.context_builder as builder_module

    history = Conversation.from_messages([
        Message(role="system", content="rules"),
        Message(role="user", content="current"),
    ])
    memory = Message(role="system", content="memory " * 100)
    guidance = ("current runtime guidance " * 20,)
    registry = ToolRegistry.from_tools([FakeTool()])
    tools = registry.get_schemas()
    without_guidance = estimate_conversation(
        Conversation.from_messages([*history.get_messages(), memory]), tools=tools,
    ).estimated_input_tokens
    runner = AgentRunner(
        llm_client=RecordingLLMClient(responses=[]), tool_registry=registry,
        conversation=history,
        context_budget=context_budget(without_guidance if omit_memory else 5000),
    )
    runner.last_memory_recall = MemoryRecall(
        message=memory, entries=(), stats=MemoryContextStats(selected_entry_count=1),
    )
    candidates = []
    real_budget = builder_module.budget_model_context

    def budget(messages, *args, **kwargs):
        candidates.append(messages)
        return real_budget(messages, *args, **kwargs)

    monkeypatch.setattr(builder_module, "budget_model_context", budget)
    context = runner._model_context(tools, guidance=guidance)

    assert len(candidates) == 1
    assert candidates[0][-2:] == (memory, Message(role="system", content=guidance[0]))
    assert context.memory_stats.included_entry_count == (0 if omit_memory else 1)
    assert context.memory_stats.selected_entry_count == 1
    assert (memory in context.messages) is not omit_memory
    assert context.messages[-1] == Message(role="user", content="current")
    assert not context.estimate.over_budget
    assert len(history.get_messages()) == 2


@pytest.mark.parametrize("overflow", [False, True])
def test_finalization_budgets_complete_request_without_pretrimming(overflow, monkeypatch) -> None:
    import mycode.context_builder as builder_module

    history = Conversation.from_messages([
        Message(role="system", content="rules " * (1000 if overflow else 1)),
        Message(role="user", content="current task request"),
        Message(role="assistant", content="oversized last message " * 1000),
    ])
    client = RecordingLLMClient(responses=[], plain_responses=["checkpoint"])
    runner = AgentRunner(
        llm_client=client, tool_registry=ToolRegistry(), conversation=history,
        context_budget=context_budget(2000),
    )
    candidates = []
    real_budget = builder_module.budget_model_context

    def budget(messages, *args, **kwargs):
        candidates.append(messages)
        return real_budget(messages, *args, **kwargs)

    monkeypatch.setattr(builder_module, "budget_model_context", budget)
    events = list(runner._stream_finalization_after_max_turns())

    assert len(candidates) == 1
    assert candidates[0][-1].content == MAX_TURNS_FINALIZATION_PROMPT
    assert candidates[0][-2].content == "oversized last message " * 1000
    assert runner.last_model_context.estimate.over_budget is overflow
    assert any(m.content == "current task request" for m in runner.last_model_context.messages)
    assert len(client.seen_plain_conversations) == (0 if overflow else 1)
    assert events[-1].stop_reason == "max_turns"
    assert all(m.content != MAX_TURNS_FINALIZATION_PROMPT for m in history.get_messages())


class RecordingLLMClient:
    def __init__(
        self,
        responses: list[AgentModelResponse],
        stream_events: list[list[AgentEvent]] | None = None,
        plain_responses: list[str] | None = None,
    ) -> None:
        self.responses = responses
        self.stream_events = [] if stream_events is None else stream_events
        self.plain_responses = [] if plain_responses is None else plain_responses
        self.seen_conversations: list[list[dict[str, object]]] = []
        self.seen_tools: list[list[dict[str, object]]] = []
        self.seen_plain_conversations: list[list[dict[str, object]]] = []

    def complete(self, conversation: Conversation) -> Message:
        self.seen_plain_conversations.append(conversation.to_model_messages())
        if not self.plain_responses:
            raise RuntimeError("RecordingLLMClient has no plain responses left")
        return Message(role="assistant", content=self.plain_responses.pop(0))

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        yield self.complete(conversation).content

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        self.seen_conversations.append(conversation.to_model_messages())
        self.seen_tools.append(tools)

        if self.stream_events:
            yield from self.stream_events.pop(0)
            return
        if not self.responses:
            raise RuntimeError("RecordingLLMClient has no responses left")

        response = self.responses.pop(0)
        if response.reasoning_content is not None:
            yield AgentEvent(
                type="reasoning_delta",
                reasoning_content=response.reasoning_content,
            )
        if response.tool_calls and response.reasoning_state != "absent":
            yield AgentEvent(
                type="reasoning_state",
                reasoning_state=response.reasoning_state,
            )
        if response.stop_reason == "model_error":
            yield AgentEvent(type="error", error=response.content)
        elif response.content:
            yield AgentEvent(type="text_delta", content=response.content)
        for tool_call in response.tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tool_call)


class FailingLLMClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, conversation: Conversation) -> Message:
        raise NotImplementedError

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        raise NotImplementedError

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        raise self.error
        yield


class FakeArgs(ToolArgs):
    text: str


class FakeTool(BaseTool[FakeArgs]):
    name = "fake_tool"
    description = "Fake tool."
    args_model = FakeArgs
    capability = "read"
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(
            content=f"fake result: {args.text}",
            metadata={"text": args.text},
        )


class FakeArtifactReadTool(FakeTool):
    name = "read_artifact"
    description = "Recover an already externalized result."


class FakeWriteTool(BaseTool[FakeArgs]):
    name = "write_file"
    description = "Fake write tool."
    args_model = FakeArgs
    capability = "write"
    risk = "medium"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(content=f"wrote: {args.text}")


class FakeValidationArgs(ToolArgs):
    command: list[str]


class FakeValidationTool(BaseTool[FakeValidationArgs]):
    name = "run_validation"
    description = "Fake validation tool."
    args_model = FakeValidationArgs
    capability = "command"
    risk = "medium"

    def _run(self, args: FakeValidationArgs) -> ToolResult:
        return ToolResult.success(content=f"validated: {' '.join(args.command)}")


class ActivityProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.serial_overlap = False
        self.safe_started_during_serial = False
        self.serial_active = False
        self.calls: list[str] = []

    def run(self, name: str) -> None:
        with self._lock:
            self.calls.append(name)
            if name == "serial":
                self.serial_active = True
            elif self.serial_active:
                self.safe_started_during_serial = True
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if name == "serial" and self.active > 1:
                self.serial_overlap = True
        time.sleep(0.03)
        with self._lock:
            if name == "serial":
                self.serial_active = False
            self.active -= 1


class InstrumentedTool(BaseTool[FakeArgs]):
    name = "instrumented_tool"
    description = "Instrumented tool for concurrency tests."
    args_model = FakeArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(
        self,
        name: str,
        probe: ActivityProbe,
        *,
        concurrency_safe: bool = True,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.probe = probe
        self.concurrency_safe = concurrency_safe
        self.fail = fail

    def _run(self, args: FakeArgs) -> ToolResult:
        self.probe.run(self.name)
        if self.fail:
            return ToolResult.failure("synthetic failure")
        return ToolResult.success(f"{self.name}:{args.text}")


class RaisingAsyncRegistry(ToolRegistry):
    async def run_tool_async(
        self,
        name,
        arguments,
        *,
        permission_lock,
    ) -> ToolResult:
        if name == "raising":
            raise RuntimeError("synthetic async crash")
        return await super().run_tool_async(
            name,
            arguments,
            permission_lock=permission_lock,
        )


class AskingPermissionChecker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def check(self, request, profile) -> PermissionDecision:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return PermissionDecision.ask(message=f"Confirm {request.tool_name}")


class SerialConfirmer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def confirm(self, request) -> ConfirmationResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return ConfirmationResult.approved("synthetic approval")
