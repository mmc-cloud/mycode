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
from mycode.context_budget import ContextBudget, TokenUsage
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.memory import MemoryStore
from mycode.memory_context import MemoryContextSelector
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.runner import (
    AgentRunner,
    DEFAULT_MAX_CONCURRENT_SAFE_TOOLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
    ToolBatchExecution,
    ToolBatchContractError,
    ToolCallExecution,
    _execute_tool_batch_with_policy,
    _observe_tool_turn_progress,
    execute_tool_batch,
    format_tool_result,
)
from mycode.run_progress import (
    DEFAULT_READONLY_TURN_LIMIT,
    DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT,
    DEFAULT_READY_ACTION_TURN_LIMIT,
    RunProgress,
)
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
    llm_client = RecordingLLMClient(
        responses=[AgentModelResponse(content="final answer")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    response = _run_response(runner, "hello")

    assert response == AgentModelResponse(content="final answer")
    assert runner.conversation.get_messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="final answer"),
    ]


def test_runner_intercepts_unverified_completion_once_for_current_revision() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(content="premature"),
            AgentModelResponse(content="cannot validate; incomplete"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    response = _run_response(runner, "make a change")

    assert response.content == "cannot validate; incomplete"
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 1
    assert len(llm_client.seen_conversations) == 3
    correction_messages = [
        str(message["content"])
        for message in llm_client.seen_conversations[2]
        if message["role"] == "system"
    ]
    assert any(
        "还没有通过 Runtime 认可的最终验证" in message
        and "`run_validation`" in message
        for message in correction_messages
    )
    assert all(
        message.content != "premature"
        for message in runner.conversation.get_messages()
    )


def test_runner_allows_completion_after_current_revision_validation() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    validation_call = AgentToolCall(
        id="call_validation",
        name="run_validation",
        arguments={"text": "tests"},
    )

    def validated_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success(
                        f"completed {tool_call.name}",
                        metadata=(
                            {"exit_code": 0, "timed_out": False}
                            if tool_call.name == "run_validation"
                            else {}
                        ),
                    ),
                )
                for tool_call in tool_calls
            )
        )

    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(
                tool_calls=[validation_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="verified"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=validated_batch,
    )

    response = _run_response(runner, "make and validate a change")

    assert response.content == "verified"
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision is None
    assert runner.last_run_progress.validated_revision == 1


def test_runner_reopens_completion_after_later_validation_failure() -> None:
    calls = [
        AgentToolCall(
            id="write",
            name="write_file",
            arguments={"text": "change"},
        ),
        AgentToolCall(
            id="pass",
            name="run_validation",
            arguments={"text": "targeted pass"},
        ),
        AgentToolCall(
            id="fail",
            name="run_validation",
            arguments={"text": "broader fail"},
        ),
    ]

    def validation_sequence(registry, tool_calls):
        executions = []
        for call in tool_calls:
            metadata = {}
            ok = True
            if call.id == "pass":
                metadata = {"exit_code": 0, "timed_out": False}
            elif call.id == "fail":
                metadata = {"exit_code": 1, "timed_out": False}
                ok = False
            result = (
                ToolResult.success(f"completed {call.id}", metadata=metadata)
                if ok
                else ToolResult.failure(f"failed {call.id}", metadata=metadata)
            )
            executions.append(ToolCallExecution(tool_call=call, result=result))
        return ToolBatchExecution(executions=tuple(executions))

    llm_client = RecordingLLMClient(
        responses=[
            *(AgentModelResponse(tool_calls=[call], stop_reason="tool_calls") for call in calls),
            AgentModelResponse(content="premature after stale pass"),
            AgentModelResponse(content="reported failed validation"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=validation_sequence,
    )

    response = _run_response(runner, "change and validate")

    assert response.content == "reported failed validation"
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.task_phase == "ACT"
    assert runner.last_run_progress.mutation_revision == 1
    assert runner.last_run_progress.validated_revision == 0
    assert runner.last_run_progress.completion_correction_revision == 1
    assert len(llm_client.seen_conversations) == 5


def test_runner_allows_a_new_completion_correction_after_new_revision() -> None:
    first_write = AgentToolCall(
        id="call_write_1",
        name="write_file",
        arguments={"text": "first change"},
    )
    second_write = AgentToolCall(
        id="call_write_2",
        name="write_file",
        arguments={"text": "second change"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[first_write], stop_reason="tool_calls"),
            AgentModelResponse(content="first premature answer"),
            AgentModelResponse(tool_calls=[second_write], stop_reason="tool_calls"),
            AgentModelResponse(content="second premature answer"),
            AgentModelResponse(content="accepted without validation"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeWriteTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=4,
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "make two changes")

    assert response == AgentModelResponse(content="accepted without validation")
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 2
    assert runner.last_run_progress.mutation_revision == 2
    assert runner.last_run_progress.validated_revision == 0
    assert runner.last_run_progress.task_phase == "VERIFY"
    assert len(llm_client.seen_conversations) == 5
    assert all(
        message.content
        not in {"first premature answer", "second premature answer"}
        for message in runner.conversation.get_messages()
    )


def test_runner_does_not_hard_terminate_after_validation() -> None:
    calls = [
        AgentToolCall(
            id="call_write",
            name="write_file",
            arguments={"text": "change"},
        ),
        AgentToolCall(
            id="call_validation",
            name="run_validation",
            arguments={"text": "tests"},
        ),
        AgentToolCall(
            id="call_read",
            name="fake_tool",
            arguments={"text": "follow-up evidence"},
        ),
    ]

    def validation_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success(
                        f"completed {tool_call.name}",
                        metadata=(
                            {"exit_code": 0, "timed_out": False}
                            if tool_call.name == "run_validation"
                            else {}
                        ),
                    ),
                )
                for tool_call in tool_calls
            )
        )

    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[call], stop_reason="tool_calls")
            for call in calls
        ]
        + [AgentModelResponse(content="done after follow-up")]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=validation_batch,
    )

    response = _run_response(runner, "change, validate, and inspect")

    assert response == AgentModelResponse(content="done after follow-up")
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.task_phase == "VALIDATED"
    assert runner.last_run_progress.post_validation_tool_turn_count == 1
    assert len(llm_client.seen_conversations) == 4


def test_runner_reopens_correction_after_failed_correction_validation() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    validation_call = AgentToolCall(
        id="call_validation",
        name="run_validation",
        arguments={"text": "tests"},
    )

    def failing_validation_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=(
                        ToolResult.failure(
                            "tests failed",
                            metadata={"exit_code": 1, "timed_out": False},
                        )
                        if tool_call.name == "run_validation"
                        else ToolResult.success(f"completed {tool_call.name}")
                    ),
                )
                for tool_call in tool_calls
            )
        )

    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(content="premature"),
            AgentModelResponse(
                tool_calls=[validation_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="tests still fail; incomplete"),
            AgentModelResponse(content="accepted after renewed correction"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=failing_validation_batch,
    )

    response = _run_response(runner, "make and validate a change")

    assert response.content == "accepted after renewed correction"
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 1
    assert runner.last_run_progress.last_verification_succeeded is False
    assert len(llm_client.seen_conversations) == 5


def test_completion_correction_can_validate_and_finish_at_turn_limit() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(content="premature"),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation",
                        name="run_validation",
                        arguments={"text": "tests"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="incomplete after correction"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=lambda registry, tool_calls: ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success(
                        f"completed {tool_call.name}",
                        metadata=(
                            {"exit_code": 0, "timed_out": False}
                            if tool_call.name == "run_validation"
                            else {}
                        ),
                    ),
                )
                for tool_call in tool_calls
            )
        ),
        max_turns=2,
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "make a change")

    assert response.stop_reason == "final_answer"
    assert response.content == "incomplete after correction"
    assert len(llm_client.seen_conversations) == 4


def test_completion_correction_can_accept_second_final_at_turn_limit() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[write_call], stop_reason="tool_calls"),
            AgentModelResponse(content="premature"),
            AgentModelResponse(content="cannot validate; incomplete"),
        ]
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeWriteTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=2,
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "make a change")

    assert response == AgentModelResponse(content="cannot validate; incomplete")
    assert len(llm_client.seen_conversations) == 3
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 1


def test_historical_completion_correction_does_not_unlock_extra_turns() -> None:
    calls = [
        AgentToolCall(
            id="call_write",
            name="write_file",
            arguments={"text": "change"},
        ),
        AgentToolCall(
            id="call_validation",
            name="run_validation",
            arguments={"text": "tests"},
        ),
        AgentToolCall(
            id="call_read_1",
            name="fake_tool",
            arguments={"text": "follow-up-1"},
        ),
        AgentToolCall(
            id="call_read_2",
            name="fake_tool",
            arguments={"text": "follow-up-2"},
        ),
    ]

    def validation_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success(
                        f"completed {tool_call.name}",
                        metadata=(
                            {"exit_code": 0, "timed_out": False}
                            if tool_call.name == "run_validation"
                            else {}
                        ),
                    ),
                )
                for tool_call in tool_calls
            )
        )

    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[calls[0]], stop_reason="tool_calls"),
            AgentModelResponse(content="premature"),
            AgentModelResponse(tool_calls=[calls[1]], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[calls[2]], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[calls[3]], stop_reason="tool_calls"),
            AgentModelResponse(content="unexpected extra turn"),
        ],
        plain_responses=["checkpoint"],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=validation_batch,
        max_turns=5,
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "make, validate, and keep inspecting")

    assert response.stop_reason == "max_turns"
    assert len(llm_client.seen_conversations) == 5
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 1
    assert runner.last_run_progress.validated_revision == 1
    assert runner.last_run_progress.task_phase == "VALIDATED"


def test_multiple_revision_corrections_keep_fixed_runner_upper_bound() -> None:
    first_write = AgentToolCall(
        id="call_write_1",
        name="write_file",
        arguments={"text": "first change"},
    )
    second_write = AgentToolCall(
        id="call_write_2",
        name="write_file",
        arguments={"text": "second change"},
    )
    follow_up_calls = [
        AgentToolCall(
            id=f"call_read_{index}",
            name="fake_tool",
            arguments={"text": f"follow-up-{index}"},
        )
        for index in range(1, 3)
    ]
    llm_client = RecordingLLMClient(
        responses=[
            AgentModelResponse(tool_calls=[first_write], stop_reason="tool_calls"),
            AgentModelResponse(content="first premature answer"),
            AgentModelResponse(tool_calls=[second_write], stop_reason="tool_calls"),
            AgentModelResponse(content="second premature answer"),
            AgentModelResponse(
                tool_calls=[follow_up_calls[0]],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[follow_up_calls[1]],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="unexpected unbounded turn"),
        ],
        plain_responses=["checkpoint"],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool(), FakeWriteTool()]),
        tool_batch_handler=_successful_batch,
        max_turns=4,
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "make two revisions")

    assert response.stop_reason == "max_turns"
    assert len(llm_client.seen_conversations) == 6
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 2
    assert runner.last_run_progress.mutation_revision == 2
    assert runner.last_run_progress.validated_revision == 0


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


def test_convergence_policy_filters_each_call_and_preserves_batch_order() -> None:
    calls = [
        AgentToolCall(
            id="read",
            name="fake_tool",
            arguments={"text": "more evidence"},
        ),
        AgentToolCall(
            id="write",
            name="write_file",
            arguments={"text": "minimal edit"},
        ),
        AgentToolCall(
            id="validate",
            name="run_validation",
            arguments={"text": "targeted test"},
        ),
    ]
    registry = ToolRegistry.from_tools(
        [FakeTool(), FakeWriteTool(), FakeValidationTool()]
    )
    executed: list[str] = []

    def recording_batch(registry, tool_calls):
        executed.extend(call.id for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    batch = _execute_tool_batch_with_policy(
        registry,
        calls,
        tool_policy="block_investigate",
        handler=recording_batch,
    )

    assert executed == ["write", "validate"]
    assert [execution.tool_call.id for execution in batch.executions] == [
        "read",
        "write",
        "validate",
    ]
    assert [execution.result.ok for execution in batch.executions] == [
        False,
        True,
        True,
    ]
    assert batch.executions[0].result.metadata["reason"] == (
        "runtime_policy_blocked_investigation"
    )


def test_convergence_policy_classifies_run_command_arguments_per_call() -> None:
    commands = [
        ("rg", ["rg", "needle", "."]),
        ("pytest", ["pytest", "-q"]),
        ("cp", ["cp", "source.py", "target.py"]),
        ("neutral", ["python", "script.py"]),
    ]
    calls = [
        AgentToolCall(
            id=call_id,
            name="run_command",
            arguments={"command": command},
        )
        for call_id, command in commands
    ]
    registry = ToolRegistry.from_tools([FakeRunCommandTool()])
    executed: list[str] = []

    def recording_batch(registry, tool_calls):
        executed.extend(call.id for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    batch = _execute_tool_batch_with_policy(
        registry,
        calls,
        tool_policy="block_investigate",
        handler=recording_batch,
    )

    assert executed == ["pytest", "cp", "neutral"]
    assert [execution.tool_call.id for execution in batch.executions] == [
        "rg",
        "pytest",
        "cp",
        "neutral",
    ]
    assert batch.executions[0].result.ok is False
    assert all(execution.result.ok for execution in batch.executions[1:])


def test_convergence_policy_blocks_investigative_delegates_per_call() -> None:
    calls = [
        AgentToolCall(
            id=role,
            name="delegate_task",
            arguments={"role": role, "objective": f"Run {role}"},
        )
        for role in ("explorer", "reviewer", "tester")
    ]
    registry = ToolRegistry.from_tools([FakeControlTool()])
    executed: list[str] = []

    def recording_batch(registry, tool_calls):
        executed.extend(call.id for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    batch = _execute_tool_batch_with_policy(
        registry,
        calls,
        tool_policy="block_investigate",
        handler=recording_batch,
    )
    progress = RunProgress(readiness_mode="ready_action")
    _observe_tool_turn_progress(
        progress,
        turn_number=1,
        registry=registry,
        batch=batch,
    )

    assert executed == ["tester"]
    assert [execution.tool_call.id for execution in batch.executions] == [
        "explorer",
        "reviewer",
        "tester",
    ]
    assert [execution.result.ok for execution in batch.executions] == [
        False,
        False,
        True,
    ]
    assert all(
        execution.result.metadata.get("reason")
        == "runtime_policy_blocked_investigation"
        for execution in batch.executions[:2]
    )
    assert progress.ready_action_turn_count == 1
    assert progress.validated_revision == 0


def test_open_policy_keeps_all_delegate_roles_unchanged() -> None:
    calls = [
        AgentToolCall(
            id=role,
            name="delegate_task",
            arguments={"role": role, "objective": f"Run {role}"},
        )
        for role in ("explorer", "reviewer", "tester")
    ]
    registry = ToolRegistry.from_tools([FakeControlTool()])
    executed: list[str] = []

    def recording_batch(registry, tool_calls):
        executed.extend(call.id for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    batch = _execute_tool_batch_with_policy(
        registry,
        calls,
        tool_policy="open",
        handler=recording_batch,
    )

    assert executed == ["explorer", "reviewer", "tester"]
    assert all(execution.result.ok for execution in batch.executions)


def test_convergence_policy_blocks_explorer_but_executes_same_batch_write() -> None:
    calls = [
        AgentToolCall(
            id="explorer",
            name="delegate_task",
            arguments={"role": "explorer", "objective": "Inspect"},
        ),
        AgentToolCall(
            id="write",
            name="write_file",
            arguments={"text": "minimal edit"},
        ),
    ]
    registry = ToolRegistry.from_tools([FakeControlTool(), FakeWriteTool()])
    executed: list[str] = []

    def recording_batch(registry, tool_calls):
        executed.extend(call.id for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    batch = _execute_tool_batch_with_policy(
        registry,
        calls,
        tool_policy="block_investigate",
        handler=recording_batch,
    )

    assert executed == ["write"]
    assert [execution.result.ok for execution in batch.executions] == [False, True]


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


def test_runner_stops_after_max_turns() -> None:
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
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    response = _run_response(runner, "loop")

    assert response == AgentModelResponse(
        content="同步阶段性结论。",
        stop_reason="max_turns",
    )
    assert len(llm_client.seen_conversations) == 2
    assert len(llm_client.seen_plain_conversations) == 1
    assert runner.conversation.get_messages()[-1].content.endswith("同步阶段性结论。")
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.task_phase == "INVESTIGATE"


def test_runner_default_max_turns_allows_bounded_project_exploration() -> None:
    assert DEFAULT_MAX_TURNS == 50
    assert AgentRunner(
        llm_client=FakeLLMClient(responses=[]),
        tool_registry=ToolRegistry(),
    ).max_turns == 50


def test_runner_rejects_invalid_readonly_turn_limit() -> None:
    with pytest.raises(ValueError, match="readonly_turn_limit"):
        AgentRunner(
            llm_client=FakeLLMClient(responses=[]),
            tool_registry=ToolRegistry(),
            readonly_turn_limit=0,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"convergence_remaining_turns": None},
        {"convergence_prompt": None},
        {"convergence_remaining_turns": 0},
        {"convergence_remaining_turns": 50},
        {"convergence_prompt": "   "},
    ],
)
def test_runner_rejects_invalid_convergence_configuration(options) -> None:
    with pytest.raises(ValueError, match="convergence_"):
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


def test_runner_stream_reports_convergence_when_five_turns_remain() -> None:
    tool_call = AgentToolCall(
        id="call_1",
        name="fake_tool",
        arguments={"text": "inspect"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=tool_call)],
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
    assert turn_events == [
        AgentEvent(type="turn", turn_number=1, max_turns=6),
        AgentEvent(
            type="turn",
            content="剩余约 5 轮，Agent 将优先收敛",
            turn_number=2,
            max_turns=6,
        ),
    ]
    second_context = llm_client.seen_conversations[1]
    assert any(
        message["role"] == "system" and "实施最小修改和关键验证" in str(message["content"])
        for message in second_context
    )


def _readonly_responses() -> list[AgentModelResponse]:
    return [
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id=f"call_read_{turn}",
                    name="fake_tool",
                    arguments={"text": f"evidence-{turn}"},
                )
            ],
            stop_reason="tool_calls",
        )
        for turn in range(1, DEFAULT_READONLY_TURN_LIMIT + 1)
    ]


def _readonly_stream_events() -> list[list[AgentEvent]]:
    return [
        [
            AgentEvent(
                type="tool_call",
                tool_call=AgentToolCall(
                    id=f"call_read_{turn}",
                    name="fake_tool",
                    arguments={"text": f"evidence-{turn}"},
                ),
            )
        ]
        for turn in range(1, DEFAULT_READONLY_TURN_LIMIT + 1)
    ]


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


def test_runner_keeps_normal_tools_after_eight_readonly_turns() -> None:
    responses = _readonly_responses()
    responses.append(AgentModelResponse(content="enough evidence"))
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool(), FakeWriteTool()]),
    )

    response = _run_response(runner, "inspect before editing")

    assert response == AgentModelResponse(content="enough evidence")
    assert [schema["name"] for schema in llm_client.seen_tools[8]] == [
        "fake_tool",
        "write_file",
    ]
    assert any(
        message["role"] == "system"
        and "已经进行了较长调查" in str(message["content"])
        for message in llm_client.seen_conversations[8]
    )
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_turn == 9
    assert runner.last_run_progress.first_edit_turn is None


def test_runner_keeps_normal_tools_during_soft_convergence() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_read",
                        name="fake_tool",
                        arguments={"text": "targeted evidence"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                content="继续总结",
            ),
        ]
    )
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool(), FakeWriteTool()]),
    )

    response = _run_response(runner, "inspect before editing")

    assert response == AgentModelResponse(content="继续总结")
    assert [schema["name"] for schema in llm_client.seen_tools[8]] == [
        "fake_tool",
        "write_file",
    ]
    assert [schema["name"] for schema in llm_client.seen_tools[9]] == [
        "fake_tool",
        "write_file",
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_mode == "open"


def test_runner_soft_convergence_keeps_tools_and_successful_edit_resets_cycle() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"text": "minimal edit"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ]
    )
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeControlTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    _run_response(runner, "inspect and edit")

    assert [schema["name"] for schema in llm_client.seen_tools[8]] == [
        "fake_tool",
        "write_file",
        "delegate_task",
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_turn == 9
    assert runner.last_run_progress.first_edit_turn == 9
    assert runner.last_run_progress.readiness_to_edit_gap == 0
    assert runner.last_run_progress.readiness_mode == "open"


def test_runner_soft_convergence_enters_bounded_action_window() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_0",
                        name="fake_tool",
                        arguments={"text": "observe"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_1",
                        name="fake_tool",
                        arguments={"text": "observe again"},
                    )
                ],
                stop_reason="tool_calls",
            ),
        ]
    )
    for turn in range(DEFAULT_READY_ACTION_TURN_LIMIT - 1):
        responses.append(
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id=f"call_restricted_{turn}",
                        name="delegate_task",
                        arguments={"text": f"investigate {turn}"},
                    )
                ],
                stop_reason="tool_calls",
            )
        )
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_write_after_prepare",
                        name="write_file",
                        arguments={"text": "minimal edit"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation_after_prepare",
                        name="run_validation",
                        arguments={"text": "targeted validation"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ]
    )
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool(), FakeControlTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    _run_response(runner, "inspect, prepare, and edit")

    action_tool_names = ["write_file", "run_validation", "delegate_task"]
    for seen_tools in llm_client.seen_tools[
        DEFAULT_READONLY_TURN_LIMIT + DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT :
        DEFAULT_READONLY_TURN_LIMIT
        + DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
        + DEFAULT_READY_ACTION_TURN_LIMIT
    ]:
        assert [schema["name"] for schema in seen_tools] == action_tool_names
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.first_edit_turn == (
        DEFAULT_READONLY_TURN_LIMIT
        + DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
        + DEFAULT_READY_ACTION_TURN_LIMIT
    )
    assert runner.last_run_progress.readiness_mode == "open"


def test_runner_action_window_rejects_read_before_edit() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_0",
                        name="fake_tool",
                        arguments={"text": "observe"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_1",
                        name="fake_tool",
                        arguments={"text": "observe again"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_forbidden_read",
                        name="fake_tool",
                        arguments={"text": "more investigation"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_write_after_rejection",
                        name="write_file",
                        arguments={"text": "minimal edit"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation_after_rejection",
                        name="run_validation",
                        arguments={"text": "targeted validation"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ]
    )
    executed_names: list[str] = []

    def recording_batch(registry, tool_calls):
        executed_names.extend(call.name for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool(), FakeControlTool()]
        ),
        tool_batch_handler=recording_batch,
    )

    _run_response(runner, "inspect and edit")

    assert executed_names == (
        ["fake_tool"] * DEFAULT_READONLY_TURN_LIMIT
        + ["fake_tool"] * DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
        + ["write_file", "run_validation"]
    )
    assert [schema["name"] for schema in llm_client.seen_tools[10]] == [
        "write_file",
        "run_validation",
        "delegate_task",
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_mode == "open"
    assert runner.last_run_progress.ready_investigation_turn_count == 1


def test_runner_action_window_allows_artifact_rehydration_before_edit() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_0",
                        name="fake_tool",
                        arguments={"text": "observe"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_1",
                        name="fake_tool",
                        arguments={"text": "observe again"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_artifact",
                        name="read_artifact",
                        arguments={"text": "recover existing evidence"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"text": "minimal edit"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation",
                        name="run_validation",
                        arguments={"text": "targeted validation"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ]
    )
    executed_names: list[str] = []

    def recording_batch(registry, tool_calls):
        executed_names.extend(call.name for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [
                FakeTool(),
                FakeArtifactReadTool(),
                FakeWriteTool(),
                FakeValidationTool(),
                FakeControlTool(),
            ]
        ),
        tool_batch_handler=recording_batch,
    )

    _run_response(runner, "inspect, recover saved evidence, and edit")

    assert [schema["name"] for schema in llm_client.seen_tools[10]] == [
        "read_artifact",
        "write_file",
        "run_validation",
        "delegate_task",
    ]
    assert executed_names == (
        ["fake_tool"] * DEFAULT_READONLY_TURN_LIMIT
        + ["fake_tool"] * DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
        + ["read_artifact", "write_file", "run_validation"]
    )
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.rehydration_turn_count == 1
    assert runner.last_run_progress.ready_investigation_turn_count == 0


def test_runner_soft_convergence_with_read_only_registry_can_answer() -> None:
    responses = [
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_read",
                    name="fake_tool",
                    arguments={"text": "evidence"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(content="read-only conclusion"),
    ]
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        readonly_turn_limit=1,
    )

    response = _run_response(runner, "inspect only")

    assert response == AgentModelResponse(content="read-only conclusion")
    assert llm_client.seen_tools[1] == [
        {
            "name": "fake_tool",
            "description": "Fake tool.",
            "parameters": {
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "type": "object",
            },
        }
    ]


def test_runner_action_window_rejects_read_and_replans() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_0",
                        name="fake_tool",
                        arguments={"text": "observe"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_soft_1",
                        name="fake_tool",
                        arguments={"text": "observe again"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_forbidden_read",
                        name="fake_tool",
                        arguments={"text": "must not run"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="stopped after rejection"),
        ]
    )
    executed_names: list[str] = []

    def recording_batch(registry, tool_calls):
        executed_names.extend(call.name for call in tool_calls)
        return _successful_batch(registry, tool_calls)

    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool(), FakeControlTool()]
        ),
        tool_batch_handler=recording_batch,
    )

    _run_response(runner, "inspect and edit")

    assert executed_names == (
        ["fake_tool"] * DEFAULT_READONLY_TURN_LIMIT
        + ["fake_tool"] * DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
    )
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_mode == "open"
    tool_results = [
        message
        for message in runner.conversation.get_messages()
        if message.role == "tool"
        and "Runtime convergence policy" in message.content
    ]
    assert len(tool_results) == 1


def test_runner_stream_uses_same_hybrid_readiness_path() -> None:
    stream_events = _readonly_stream_events()
    stream_events.extend(
        [
            [
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="call_soft_0",
                        name="fake_tool",
                        arguments={"text": "observe"},
                    ),
                )
            ],
            [
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="call_soft_1",
                        name="fake_tool",
                        arguments={"text": "observe again"},
                    ),
                )
            ],
            [
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"text": "minimal edit"},
                    ),
                )
            ],
            [AgentEvent(type="text_delta", content="done")],
        ]
    )
    llm_client = RecordingLLMClient(responses=[], stream_events=stream_events)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeControlTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    events = list(runner.run("inspect and edit"))

    emitted_tool_names = [
        event.tool_call.name
        for event in events
        if event.type == "tool_call" and event.tool_call is not None
    ]
    assert "write_file" in emitted_tool_names
    assert [schema["name"] for schema in llm_client.seen_tools[8]] == [
        "fake_tool",
        "write_file",
        "delegate_task",
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.first_edit_turn == 11
    assert runner.last_run_progress.readiness_to_edit_gap == 2
    assert runner.last_run_progress.task_phase_history == [
        "INVESTIGATE",
        "VERIFY",
    ]
    assert any(
        message["role"] == "system"
        and "target_phase: VERIFY" in str(message["content"])
        and "reason: task_phase_verify" in str(message["content"])
        for message in llm_client.seen_conversations[11]
    )


def test_runner_stream_can_answer_after_readonly_threshold() -> None:
    stream_events = _readonly_stream_events()
    stream_events.append(
        [
            AgentEvent(type="text_delta", content="enough evidence"),
        ]
    )
    llm_client = RecordingLLMClient(responses=[], stream_events=stream_events)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    events = list(runner.run("inspect"))

    stop_events = [event for event in events if event.type == "stop"]
    assert stop_events == [AgentEvent(type="stop", stop_reason="final_answer")]
    assert [schema["name"] for schema in llm_client.seen_tools[8]] == [
        "fake_tool"
    ]


def test_runner_treats_an_executed_validation_as_key_test_even_when_it_fails() -> None:
    responses = _readonly_responses()
    responses.extend(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation",
                        name="run_validation",
                        arguments={"text": "targeted tests"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="validation failed with evidence"),
        ]
    )
    llm_client = RecordingLLMClient(responses=responses)

    def validation_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=(
                        ToolResult.failure(
                            "tests failed",
                            metadata={"exit_code": 1, "timed_out": False},
                        )
                        if tool_call.name == "run_validation"
                        else ToolResult.success(f"completed {tool_call.name}")
                    ),
                )
                for tool_call in tool_calls
            )
        )

    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool(), FakeValidationTool()]),
        tool_batch_handler=validation_batch,
    )

    _run_response(runner, "inspect and validate")

    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_turn == 9
    assert runner.last_run_progress.first_key_test_turn == 9
    assert runner.last_run_progress.last_verification_succeeded is False
    assert [schema["name"] for schema in llm_client.seen_tools[9]] == [
        "fake_tool",
        "run_validation",
    ]


def test_runner_counts_inspect_run_command_as_investigation() -> None:
    responses = [
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_read",
                    name="fake_tool",
                    arguments={"text": "source"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_git_diff",
                    name="run_command",
                    arguments={"text": "git diff", "command": ["git", "diff"]},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(content="inspection complete"),
    ]

    def inspect_command_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=(
                        ToolResult.success(
                            "diff inspected",
                            metadata={
                                "command_risk_category": "inspect",
                                "exit_code": 0,
                                "timed_out": False,
                            },
                        )
                        if tool_call.name == "run_command"
                        else ToolResult.success("source inspected")
                    ),
                )
                for tool_call in tool_calls
            )
        )

    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool(), FakeRunCommandTool()]),
        readonly_turn_limit=2,
        tool_batch_handler=inspect_command_batch,
    )

    response = _run_response(runner, "inspect with commands")

    assert response == AgentModelResponse(content="inspection complete")
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.readiness_turn == 3
    assert runner.last_run_progress.investigation_turn_count == 2
    assert [schema["name"] for schema in llm_client.seen_tools[2]] == [
        "fake_tool",
        "run_command",
    ]
    assert runner.last_run_progress.readiness_mode == "open"


def test_runner_records_act_verify_failure_repair_and_validated_loop() -> None:
    responses = [
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_read",
                    name="fake_tool",
                    arguments={"text": "source"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_write_1",
                    name="write_file",
                    arguments={"text": "first fix"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_test_1",
                    name="run_validation",
                    arguments={"text": "targeted tests"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_write_2",
                    name="write_file",
                    arguments={"text": "repair"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id="call_test_2",
                    name="run_validation",
                    arguments={"text": "targeted tests"},
                )
            ],
            stop_reason="tool_calls",
        ),
        AgentModelResponse(content="done"),
    ]
    validation_count = 0

    def phased_batch(registry, tool_calls):
        nonlocal validation_count
        executions = []
        for tool_call in tool_calls:
            if tool_call.name == "run_validation":
                validation_count += 1
                passed = validation_count == 2
                result = (
                    ToolResult.success(
                        "tests passed",
                        metadata={"exit_code": 0, "timed_out": False},
                    )
                    if passed
                    else ToolResult.failure(
                        "tests failed",
                        metadata={"exit_code": 1, "timed_out": False},
                    )
                )
            else:
                result = ToolResult.success(f"completed {tool_call.name}")
            executions.append(ToolCallExecution(tool_call=tool_call, result=result))
        return ToolBatchExecution(executions=tuple(executions))

    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool()]
        ),
        readonly_turn_limit=1,
        tool_batch_handler=phased_batch,
    )

    response = _run_response(runner, "fix and verify")

    assert response == AgentModelResponse(content="done")
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.task_phase == "VALIDATED"
    assert runner.last_run_progress.task_phase_history == [
        "INVESTIGATE",
        "VERIFY",
        "ACT",
        "VERIFY",
        "VALIDATED",
    ]
    assert runner.last_run_progress.last_verification_succeeded is True
    phase_prompts = [
        next(
            (
                str(message["content"])
                for message in conversation
                if message["role"] == "system"
                and "reason: task_phase_" in str(message["content"])
            ),
            "",
        )
        for conversation in llm_client.seen_conversations
    ]
    assert phase_prompts[0] == ""
    assert phase_prompts[1] == ""
    assert "target_phase: VERIFY" in phase_prompts[2]
    assert "target_phase: ACT" in phase_prompts[3]
    assert "target_phase: VERIFY" in phase_prompts[4]
    assert "target_phase: VALIDATED" in phase_prompts[5]
    assert not any(
        message.role == "system" and "Current state:" in message.content
        for message in runner.conversation.get_messages()
    )


def test_runner_stream_requests_replan_before_repeated_call_hard_stop() -> None:
    calls = [
        AgentToolCall(
            id=f"call_{index}",
            name="fake_tool",
            arguments={"text": "same"},
        )
        for index in range(2)
    ]
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=calls[0])],
            [AgentEvent(type="tool_call", tool_call=calls[1])],
            [AgentEvent(type="text_delta", content="replanned")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
    )

    events = list(runner.run("repeat"))

    third_turn = [event for event in events if event.type == "turn"][2]
    assert "重复了相同的工具调用" in third_turn.content
    assert any(
        message["role"] == "system"
        and "不要重复相同的失败动作" in str(message["content"])
        for message in llm_client.seen_conversations[2]
    )


def test_runner_incorporates_replan_into_active_act_directive() -> None:
    calls = [
        AgentToolCall(id="call-write", name="write_file", arguments={"text": "edit"}),
        AgentToolCall(
            id="call-verify",
            name="run_validation",
            arguments={"text": "test"},
        ),
        AgentToolCall(id="call-read-1", name="fake_tool", arguments={"text": "same"}),
        AgentToolCall(id="call-read-2", name="fake_tool", arguments={"text": "same"}),
    ]
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            *([AgentEvent(type="tool_call", tool_call=call)] for call in calls),
            [AgentEvent(type="text_delta", content="candidate")],
            [AgentEvent(type="text_delta", content="done")],
        ],
    )

    def failed_validation_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=(
                        ToolResult.failure(
                            "validation failed",
                            metadata={"exit_code": 1, "timed_out": False},
                        )
                        if tool_call.name == "run_validation"
                        else ToolResult.success(f"completed {tool_call.name}")
                    ),
                )
                for tool_call in tool_calls
            )
        )

    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeTool(), FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=failed_validation_batch,
    )

    list(runner.run("edit and verify"))

    act_directives = [
        str(message["content"])
        for message in llm_client.seen_conversations[4]
        if message["role"] == "system" and "Current state:" in str(message["content"])
    ]
    assert len(act_directives) == 1
    assert "target_phase: ACT" in act_directives[0]
    assert "reason: task_phase_act_with_replan" in act_directives[0]
    assert "模型重复了相同的工具调用" in act_directives[0]
    assert "不要重复相同的失败动作" in act_directives[0]


def test_runner_stream_replans_after_three_turns_without_new_tool_evidence() -> None:
    calls = [
        AgentToolCall(
            id=f"call_{index}",
            name="probe",
            arguments={"text": str(index)},
        )
        for index in range(4)
    ]
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            *([AgentEvent(type="tool_call", tool_call=call)] for call in calls),
            [AgentEvent(type="text_delta", content="replanned")],
        ],
    )

    def same_evidence_batch(registry, tool_calls):
        return ToolBatchExecution(
            executions=tuple(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=ToolResult.success("same evidence"),
                )
                for tool_call in tool_calls
            )
        )

    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        tool_batch_handler=same_evidence_batch,
    )

    events = list(runner.run("probe"))

    fifth_turn = [event for event in events if event.type == "turn"][4]
    assert "连续多轮没有新增工具证据" in fifth_turn.content
    assert any(
        message["role"] == "system"
        and "一个能够产生新信息的直接动作" in str(message["content"])
        and "形成结论并结束" in str(message["content"])
        for message in llm_client.seen_conversations[4]
    )


def test_runner_routes_resource_stagnation_through_single_replan_guidance(
    tmp_path,
) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / "query.py").write_text(
        "\n".join(f"line {index}" for index in range(1, 20)),
        encoding="utf-8",
    )
    (tmp_path / "other.py").write_text("other\n", encoding="utf-8")
    (tmp_path / "third.py").write_text("third\n", encoding="utf-8")
    paths = [
        "query.py",
        "other.py",
        "query.py",
        "query.py",
        "third.py",
        "query.py",
        "query.py",
    ]
    responses = [
        AgentModelResponse(
            tool_calls=[
                AgentToolCall(
                    id=f"call_{index}",
                    name="read_file",
                    arguments={
                        "path": path,
                        "start_line": index + 1 if path == "query.py" else 1,
                        "max_lines": 1,
                    },
                )
            ],
            stop_reason="tool_calls",
        )
        for index, path in enumerate(paths)
    ]
    responses.append(AgentModelResponse(content="replanned conclusion"))
    llm_client = RecordingLLMClient(responses=responses)
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools([ReadFileTool(workspace)]),
    )

    response = _run_response(runner, "inspect the implementation")

    assert response.content == "replanned conclusion"
    runtime_messages = [
        message
        for message in llm_client.seen_conversations[7]
        if message["role"] == "system"
        and "Current state:" in str(message["content"])
    ]
    assert len(runtime_messages) == 1
    assert "query.py" in str(runtime_messages[0]["content"])
    assert [schema["name"] for schema in llm_client.seen_tools[7]] == [
        "read_file"
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.last_stagnant_resource == "file:query.py"
    assert list(runner.last_run_progress.recent_investigation_resources) == []


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
                task_phase="INVESTIGATE",
                transition_reason="final_answer",
            ),
        ),
        AgentEvent(type="stop", stop_reason="final_answer"),
    ]
    assert runner.conversation.get_messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hello world"),
    ]


def test_runner_stream_intercepts_unverified_completion_once_for_revision() -> None:
    write_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"text": "change"},
    )
    llm_client = RecordingLLMClient(
        responses=[],
        stream_events=[
            [AgentEvent(type="tool_call", tool_call=write_call)],
            [AgentEvent(type="text_delta", content="premature")],
            [AgentEvent(type="text_delta", content="incomplete")],
        ],
    )
    runner = AgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry.from_tools(
            [FakeWriteTool(), FakeValidationTool()]
        ),
        tool_batch_handler=_successful_batch,
    )

    events = list(runner.run("make a change"))

    assert [event.stop_reason for event in events if event.type == "stop"] == [
        "final_answer"
    ]
    assert any(
        event.type == "progress"
        and event.progress is not None
        and event.progress.transition_reason == "completion_correction_required"
        for event in events
    )
    assert any(
        event.type == "turn" and "纠偏机会" in event.content
        for event in events
    )
    assert not any(
        event.type == "text_delta" and event.content == "premature"
        for event in events
    )
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.completion_correction_revision == 1
    assert runner.conversation.get_messages()[-1] == Message(
        role="assistant",
        content="incomplete",
    )


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
                task_phase="INVESTIGATE",
                effects=("investigate",),
                transition_reason="tool_turn_observed",
            ),
        ),
        AgentEvent(type="turn", turn_number=2, max_turns=50),
        AgentEvent(type="model_start"),
        AgentEvent(type="text_delta", content="final answer"),
        AgentEvent(
            type="progress",
            progress=AgentProgressSnapshot(
                task_phase="INVESTIGATE",
                transition_reason="final_answer",
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
    assert result.metadata["max_lines"] == MAX_LINES_LIMIT
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
        tool_registry=ToolRegistry.from_tools([FakeTool()]),
        max_turns=51,
        readonly_turn_limit=None,
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
            task_phase="INVESTIGATE",
            effects=("investigate",),
            transition_reason="tool_turn_observed",
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


def test_runner_stream_stops_after_max_turns() -> None:
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
        convergence_remaining_turns=None,
        convergence_prompt=None,
    )

    events = list(runner.run("loop"))

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
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.task_phase == "INVESTIGATE"
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


class FakeValidationTool(BaseTool[FakeArgs]):
    name = "run_validation"
    description = "Fake validation tool."
    args_model = FakeArgs
    capability = "command"
    risk = "medium"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(content=f"validated: {args.text}")


class FakeRunCommandTool(BaseTool[FakeArgs]):
    name = "run_command"
    description = "Fake command tool."
    args_model = FakeArgs
    capability = "command"
    risk = "medium"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(content=f"command: {args.text}")


class FakeControlTool(BaseTool[FakeArgs]):
    name = "delegate_task"
    description = "Fake control tool."
    args_model = FakeArgs
    capability = "control"
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(content=f"delegated: {args.text}")


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
