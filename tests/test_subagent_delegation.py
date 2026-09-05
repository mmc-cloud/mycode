from collections.abc import Iterator
import json
from pathlib import Path
import threading
import time

import pytest

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.runner import (
    AgentRunner,
    DEFAULT_MAX_CONCURRENT_SAFE_TOOLS,
    DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
)
from mycode.subagents.contracts import (
    ExplorerResult,
    SubAgentResult,
    SubAgentTask,
)
from mycode.subagents.delegate import DelegateTaskTool
from mycode.subagents.delegation import DelegationToolBatchHandler
from mycode.subagents.limits import (
    DEFAULT_MAX_CONCURRENT_DELEGATIONS,
    DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN,
)
from mycode.subagents.tool_batch import SubAgentToolBatchHandler
from mycode.subagents.runtime import SubAgentExecution
from mycode.tools import PydanticTool, ToolArgs, ToolRegistry, ToolResult
from mycode.tools.defaults import create_default_tool_registry
from mycode.tools.workspace import Workspace


def test_delegate_task_schema_exposes_closed_roles_and_control_permission() -> None:
    tool = DelegateTaskTool(RecordingRuntime())

    schema = tool.get_schema()

    assert schema["name"] == "delegate_task"
    assert schema["parameters"]["properties"]["role"]["enum"] == [
        "explorer",
        "tester",
        "reviewer",
    ]
    assert tool.get_permission_profile().capability == "control"
    assert tool.get_permission_profile().risk == "low"


def test_delegate_task_returns_only_bounded_child_result_and_safe_metadata() -> None:
    runtime = RecordingRuntime(child_private_transcript="PRIVATE CHILD TOOL LOG")
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])

    result = registry.run_tool(
        "delegate_task",
        {
            "role": "explorer",
            "objective": "Locate the runner.",
            "scope_paths": ["mycode"],
        },
    )

    assert result.ok is True
    payload = json.loads(result.content)
    assert payload["role"] == "explorer"
    assert payload["status"] == "completed"
    assert payload["payload"]["summary"] == "bounded child result"
    assert "PRIVATE CHILD TOOL LOG" not in result.content
    assert "PRIVATE CHILD TOOL LOG" not in json.dumps(result.metadata)
    assert result.metadata == {
        "tool_name": "delegate_task",
        "run_id": "child-run-1",
        "role": "explorer",
        "child_status": "completed",
        "child_stop_reason": "submitted",
        "conversation_message_count": 4,
        "tool_call_count": 2,
        "validation_execution_count": 0,
        "result_chars": len(result.content),
    }
    assert runtime.tasks == [
        SubAgentTask(
            role="explorer",
            objective="Locate the runner.",
            scope_paths=["mycode"],
        )
    ]


def test_delegate_task_preserves_structured_child_failure_as_tool_success() -> None:
    runtime = RecordingRuntime(result=_failed_child_result())
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])

    result = registry.run_tool(
        "delegate_task",
        {"role": "reviewer", "objective": "Review the change."},
    )

    assert result.ok is True
    assert json.loads(result.content) == {
        "run_id": "child-run-failed",
        "role": "reviewer",
        "status": "failed",
        "stop_reason": "model_error",
        "summary": "Child could not finish.",
        "error": "Provider unavailable.",
    }
    assert result.metadata["child_status"] == "failed"


def test_delegate_task_enforces_single_level_even_when_called_directly() -> None:
    runtime = RecordingRuntime()
    registry = ToolRegistry.from_tools(
        [DelegateTaskTool(runtime, current_depth=1)]
    )

    result = registry.run_tool(
        "delegate_task",
        {"role": "explorer", "objective": "Try nested delegation."},
    )

    assert result.ok is False
    assert result.metadata["reason"] == "delegation_depth_limit"
    assert runtime.tasks == []


def test_delegate_task_does_not_swallow_child_keyboard_interrupt() -> None:
    registry = ToolRegistry.from_tools([DelegateTaskTool(InterruptingRuntime())])

    with pytest.raises(KeyboardInterrupt):
        registry.run_tool(
            "delegate_task",
            {"role": "tester", "objective": "Run validation."},
        )


def test_default_registry_can_append_delegate_without_coupling_to_runtime(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()

    registry = create_default_tool_registry(
        Workspace(tmp_path),
        extra_tools=(DelegateTaskTool(runtime),),
    )

    assert [tool.name for tool in registry.list_tools()][-1] == "delegate_task"


def test_delegation_barrier_runs_all_children_then_parent_replans_once() -> None:
    runtime = RecordingRuntime()
    stale_tool = RecordingSideEffectTool()
    explorer_call = _delegate_call(
        "call_explorer",
        objective="Inspect runner.",
        role="explorer",
    )
    reviewer_call = _delegate_call(
        "call_reviewer",
        objective="Review runner.",
        role="reviewer",
    )
    stale_call = AgentToolCall(
        id="call_stale",
        name="record_side_effect",
        arguments={"text": "stale plan"},
    )
    client = RecordingParentLLM(
        responses=[
            AgentModelResponse(
                tool_calls=[stale_call, explorer_call, reviewer_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="Replanned final answer."),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools(
            [stale_tool, DelegateTaskTool(runtime)]
        ),
        tool_batch_handler=DelegationToolBatchHandler(),
    )

    events = list(runner.run("Investigate before continuing."))

    assert [event.content for event in events if event.type == "text_delta"] == [
        "Replanned final answer."
    ]
    assert stale_tool.calls == []
    assert {task.objective for task in runtime.tasks} == {
        "Inspect runner.",
        "Review runner.",
    }
    assert len(client.seen_conversations) == 2
    second_context = json.dumps(client.seen_conversations[1], ensure_ascii=False)
    assert "skipped_due_to_delegation_barrier" in second_context
    assert "bounded child result" in second_context
    tool_messages = [
        message for message in runner.conversation.get_messages()
        if message.role == "tool"
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "call_stale",
        "call_explorer",
        "call_reviewer",
    ]


def test_multiple_independent_delegations_run_in_parallel_and_keep_order() -> None:
    runtime = ConcurrentRecordingRuntime(delay_seconds=0.04)
    handler = DelegationToolBatchHandler(max_concurrent_delegations=3)
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        _delegate_call("call_one", objective="First.", role="explorer"),
        _delegate_call("call_two", objective="Second.", role="tester"),
        _delegate_call("call_three", objective="Third.", role="reviewer"),
    ]

    batch = handler(registry, calls)

    assert runtime.max_active == 3
    assert {task.role for task in runtime.tasks} == {
        "explorer",
        "tester",
        "reviewer",
    }
    assert [item.tool_call.id for item in batch.executions] == [
        call.id for call in calls
    ]
    assert [
        json.loads(item.result.content)["summary"]
        for item in batch.executions
    ] == [
        "finished:First.",
        "finished:Second.",
        "finished:Third.",
    ]


def test_delegation_parallelism_is_bounded_across_multiple_chunks() -> None:
    runtime = ConcurrentRecordingRuntime(delay_seconds=0.03)
    handler = DelegationToolBatchHandler(
        max_delegations_per_run=10,
        max_concurrent_delegations=3,
    )
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        _delegate_call(f"call-{index}", objective=f"Task {index}.")
        for index in range(7)
    ]

    batch = handler(registry, calls)

    assert len(runtime.tasks) == 7
    assert runtime.max_active == 3
    assert all(execution.result.ok for execution in batch.executions)


def test_one_delegation_failure_does_not_cancel_siblings() -> None:
    runtime = ConcurrentRecordingRuntime(
        delay_seconds=0.03,
        failing_objectives={"Fail."},
    )
    handler = DelegationToolBatchHandler(max_concurrent_delegations=3)
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        _delegate_call("call-first", objective="First."),
        _delegate_call("call-fail", objective="Fail."),
        _delegate_call("call-last", objective="Last."),
    ]

    batch = handler(registry, calls)

    assert [execution.result.ok for execution in batch.executions] == [
        True,
        False,
        True,
    ]
    assert {
        task.objective for task in runtime.completed_tasks
    } == {"First.", "Last."}
    assert batch.executions[1].result.metadata["exception_type"] == "RuntimeError"


def test_unexpected_worker_exception_isolated_without_raw_error_text() -> None:
    runtime = ConcurrentRecordingRuntime(delay_seconds=0.01)
    handler = DelegationToolBatchHandler(max_concurrent_delegations=2)
    registry = RaisingDelegationRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        _delegate_call("call-worker-fail", objective="Worker crash."),
        _delegate_call("call-worker-ok", objective="Worker succeeds."),
    ]

    batch = handler(registry, calls)

    assert batch.executions[0].result.ok is False
    assert batch.executions[0].result.error == (
        "Delegation execution failed unexpectedly."
    )
    assert "PRIVATE WORKER ERROR" not in batch.executions[0].result.error
    assert batch.executions[0].result.metadata == {
        "reason": "delegation_worker_error",
        "exception_type": "RuntimeError",
    }
    assert batch.executions[1].result.ok is True
    assert [task.objective for task in runtime.tasks] == ["Worker succeeds."]


def test_batch_delegation_limit_is_attempt_based_and_ordered() -> None:
    runtime = ConcurrentRecordingRuntime(delay_seconds=0.01)
    handler = DelegationToolBatchHandler(
        max_delegations_per_run=2,
        max_concurrent_delegations=2,
    )
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        AgentToolCall(
            id="call-invalid",
            name="delegate_task",
            arguments={"role": "explorer"},
        ),
        _delegate_call("call-valid", objective="Valid."),
        _delegate_call("call-over-limit", objective="Too many."),
    ]

    batch = handler(registry, calls)

    assert handler.delegation_count == 3
    assert [execution.result.ok for execution in batch.executions] == [
        False,
        True,
        False,
    ]
    assert batch.executions[0].result.error == "Invalid tool arguments"
    assert batch.executions[2].result.metadata["reason"] == "delegation_limit"
    assert [task.objective for task in runtime.tasks] == ["Valid."]


def test_default_delegation_limits_are_separate_from_tool_limits() -> None:
    handler = DelegationToolBatchHandler()

    assert handler.max_concurrent_delegations == (
        DEFAULT_MAX_CONCURRENT_DELEGATIONS
    )
    assert handler.max_delegations_per_run == (
        DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN
    )
    assert DEFAULT_MAX_CONCURRENT_DELEGATIONS < (
        DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN
    )
    assert DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN < (
        DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE
    )


def test_process_interrupt_waits_for_started_sibling_and_stops_later_chunk() -> None:
    runtime = CoordinatedInterruptingRuntime()
    handler = DelegationToolBatchHandler(
        max_delegations_per_run=3,
        max_concurrent_delegations=2,
    )
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])

    with pytest.raises(KeyboardInterrupt):
        handler(
            registry,
            [
                _delegate_call("call-interrupt", objective="Interrupt."),
                _delegate_call("call-sibling", objective="Sibling."),
                _delegate_call("call-later", objective="Later."),
            ],
        )

    assert runtime.sibling_completed.is_set()
    assert {task.objective for task in runtime.tasks} == {
        "Interrupt.",
        "Sibling.",
    }


def test_subagent_regular_batch_uses_shared_concurrency_limit_and_order() -> None:
    probe = SubAgentActivityProbe()
    tools = [
        ConcurrentSubAgentTool(f"subagent_safe_{index}", probe)
        for index in range(DEFAULT_MAX_CONCURRENT_SAFE_TOOLS * 2 + 1)
    ]
    handler = SubAgentToolBatchHandler(
        role="explorer",
        max_final_payload_chars=2000,
        max_validation_calls=2,
    )
    registry = ToolRegistry.from_tools(tools)
    calls = [
        AgentToolCall(
            id=f"call-{index}",
            name=f"subagent_safe_{index}",
            arguments={"text": str(index)},
        )
        for index in range(len(tools))
    ]

    batch = handler(registry, calls)

    assert [execution.tool_call.id for execution in batch.executions] == [
        call.id for call in calls
    ]
    assert probe.max_active == DEFAULT_MAX_CONCURRENT_SAFE_TOOLS
    assert [execution.result.content for execution in batch.executions] == [
        f"subagent_safe_{index}:{index}" for index in range(len(tools))
    ]


def test_subagent_regular_batch_applies_total_tool_call_limit() -> None:
    probe = SubAgentActivityProbe()
    tools = [
        ConcurrentSubAgentTool(f"subagent_limited_{index}", probe)
        for index in range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE + 1)
    ]
    handler = SubAgentToolBatchHandler(
        role="explorer",
        max_final_payload_chars=2000,
        max_validation_calls=2,
    )
    registry = ToolRegistry.from_tools(tools)
    calls = [
        AgentToolCall(
            id=f"call-{index}",
            name=f"subagent_limited_{index}",
            arguments={"text": str(index)},
        )
        for index in range(len(tools))
    ]

    batch = handler(registry, calls)

    assert len(batch.executions) == len(calls)
    assert all(
        execution.result.ok
        for execution in batch.executions[:DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE]
    )
    assert batch.executions[-1].result.ok is False
    assert batch.executions[-1].result.metadata["reason"] == (
        "tool_call_response_limit"
    )
    assert len(probe.calls) == DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE


def test_subagent_regular_batch_isolates_async_tool_exception() -> None:
    probe = SubAgentActivityProbe()
    raising = ConcurrentSubAgentTool("subagent_raising", probe)
    succeeding = ConcurrentSubAgentTool("subagent_succeeding", probe)
    handler = SubAgentToolBatchHandler(
        role="explorer",
        max_final_payload_chars=2000,
        max_validation_calls=2,
    )
    registry = RaisingSubAgentRegistry.from_tools([raising, succeeding])

    batch = handler(
        registry,
        [
            AgentToolCall(
                id="call-raising",
                name="subagent_raising",
                arguments={"text": "x"},
            ),
            AgentToolCall(
                id="call-succeeding",
                name="subagent_succeeding",
                arguments={"text": "y"},
            ),
        ],
    )

    assert batch.executions[0].result.error == (
        "Tool execution failed: synthetic SubAgent async crash"
    )
    assert batch.executions[1].result == ToolResult.success(
        "subagent_succeeding:y"
    )
    assert probe.calls == ["subagent_succeeding"]


def test_delegate_after_total_limit_is_returned_as_failure_without_execution() -> None:
    runtime = RecordingRuntime()
    handler = DelegationToolBatchHandler()
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        AgentToolCall(
            id=f"call-missing-{index}",
            name="missing_tool",
            arguments={},
        )
        for index in range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE)
    ]
    calls.append(_delegate_call("call-over-limit-delegate", objective="Too late."))

    batch = handler(registry, calls)

    assert runtime.tasks == []
    assert len(batch.executions) == len(calls)
    assert batch.executions[-1].result.ok is False
    assert batch.executions[-1].result.metadata["reason"] == (
        "tool_call_response_limit"
    )


def test_over_limit_delegate_keeps_barrier_from_running_side_effects() -> None:
    runtime = RecordingRuntime()
    side_effect_tool = RecordingSideEffectTool()
    handler = DelegationToolBatchHandler()
    registry = ToolRegistry.from_tools(
        [side_effect_tool, DelegateTaskTool(runtime)]
    )
    calls = [
        AgentToolCall(
            id=f"call-side-effect-{index}",
            name="record_side_effect",
            arguments={"text": str(index)},
        )
        for index in range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE)
    ]
    calls.append(_delegate_call("call-over-limit-delegate", objective="Too late."))

    batch = handler(registry, calls)

    assert side_effect_tool.calls == []
    assert runtime.tasks == []
    assert all(
        execution.result.metadata["reason"] == "skipped_due_to_delegation_barrier"
        for execution in batch.executions[:DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE]
    )
    assert batch.executions[-1].result.metadata["reason"] == (
        "tool_call_response_limit"
    )


def test_over_limit_submit_keeps_barrier_from_running_side_effects() -> None:
    side_effect_tool = RecordingSideEffectTool()
    handler = SubAgentToolBatchHandler(
        role="explorer",
        max_final_payload_chars=2000,
        max_validation_calls=2,
    )
    registry = ToolRegistry.from_tools([side_effect_tool])
    calls = [
        AgentToolCall(
            id=f"call-side-effect-{index}",
            name="record_side_effect",
            arguments={"text": str(index)},
        )
        for index in range(DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE)
    ]
    calls.append(
        AgentToolCall(
            id="call-over-limit-submit",
            name="submit_result",
            arguments={},
        )
    )

    batch = handler(registry, calls)

    assert side_effect_tool.calls == []
    assert handler.submission_attempted is True
    assert all(
        execution.result.metadata["reason"] == "skipped_due_to_submission_barrier"
        for execution in batch.executions[:DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE]
    )
    assert batch.executions[-1].result.metadata["reason"] == (
        "tool_call_response_limit"
    )


def test_delegation_limit_blocks_extra_child_call_within_parent_run() -> None:
    runtime = RecordingRuntime()
    client = RecordingParentLLM(
        responses=[
            AgentModelResponse(
                tool_calls=[_delegate_call("call_one", objective="First task.")],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[_delegate_call("call_two", objective="Second task.")],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="Done."),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([DelegateTaskTool(runtime)]),
        tool_batch_handler=DelegationToolBatchHandler(max_delegations_per_run=1),
    )

    list(runner.run("Use at most one child."))

    assert len(runtime.tasks) == 1
    assert "delegation_limit" in json.dumps(
        client.seen_conversations[-1], ensure_ascii=False
    )


def test_parent_resets_delegation_limit_for_each_user_run() -> None:
    runtime = RecordingRuntime()
    client = RecordingParentLLM(
        responses=[
            AgentModelResponse(
                tool_calls=[_delegate_call("call_first", objective="First run.")],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="First done."),
            AgentModelResponse(
                tool_calls=[_delegate_call("call_second", objective="Second run.")],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="Second done."),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([DelegateTaskTool(runtime)]),
        tool_batch_handler=DelegationToolBatchHandler(max_delegations_per_run=1),
    )

    list(runner.run("First parent request."))
    list(runner.run("Second parent request."))

    assert [task.objective for task in runtime.tasks] == [
        "First run.",
        "Second run.",
    ]


def test_parent_event_loop_uses_same_barrier_and_resets_per_user_run() -> None:
    runtime = RecordingRuntime()
    first_calls = [
        _delegate_call("call_first_stream_a", objective="First stream A."),
        _delegate_call("call_first_stream_b", objective="First stream B."),
    ]
    second_calls = [
        _delegate_call("call_second_stream_a", objective="Second stream A."),
        _delegate_call("call_second_stream_b", objective="Second stream B."),
    ]
    client = RecordingParentLLM(
        stream_events=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call)
                for tool_call in first_calls
            ],
            [AgentEvent(type="text_delta", content="First done.")],
            [
                AgentEvent(type="tool_call", tool_call=tool_call)
                for tool_call in second_calls
            ],
            [AgentEvent(type="text_delta", content="Second done.")],
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([DelegateTaskTool(runtime)]),
        tool_batch_handler=DelegationToolBatchHandler(
            max_delegations_per_run=2,
            max_concurrent_delegations=2,
        ),
    )

    first_events = list(runner.run("First parent request."))
    second_events = list(runner.run("Second parent request."))

    assert sorted(task.objective for task in runtime.tasks) == [
        "First stream A.",
        "First stream B.",
        "Second stream A.",
        "Second stream B.",
    ]
    assert sum(event.type == "tool_result" for event in first_events) == 2
    assert sum(event.type == "tool_result" for event in second_events) == 2
    assert first_events[-1].stop_reason == "final_answer"
    assert second_events[-1].stop_reason == "final_answer"


def _delegate_call(
    call_id: str,
    *,
    objective: str,
    role: str = "explorer",
) -> AgentToolCall:
    return AgentToolCall(
        id=call_id,
        name="delegate_task",
        arguments={"role": role, "objective": objective},
    )


def _completed_child_result() -> SubAgentResult:
    return SubAgentResult(
        run_id="child-run-1",
        role="explorer",
        status="completed",
        stop_reason="submitted",
        summary="bounded child result",
        payload=ExplorerResult(
            status="no_match",
            summary="bounded child result",
            searched_scope=["."],
            findings=[],
        ),
    )


def _failed_child_result() -> SubAgentResult:
    return SubAgentResult(
        run_id="child-run-failed",
        role="reviewer",
        status="failed",
        stop_reason="model_error",
        summary="Child could not finish.",
        error="Provider unavailable.",
    )


class RecordingRuntime:
    def __init__(
        self,
        *,
        result: SubAgentResult | None = None,
        child_private_transcript: str = "",
    ) -> None:
        self.result = _completed_child_result() if result is None else result
        self.child_private_transcript = child_private_transcript
        self.tasks: list[SubAgentTask] = []

    def execute(self, task: SubAgentTask, *, observer=None) -> SubAgentExecution:
        self.tasks.append(task)
        return SubAgentExecution(
            result=self.result,
            transitions=(),
            snapshot=None,
            context=None,
            token_usage=None,
            conversation_message_count=4,
            tool_call_count=2,
            validation_execution_count=0,
        )


class ConcurrentRecordingRuntime:
    def __init__(
        self,
        *,
        delay_seconds: float,
        failing_objectives: set[str] | None = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.failing_objectives = (
            set() if failing_objectives is None else set(failing_objectives)
        )
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.tasks: list[SubAgentTask] = []
        self.completed_tasks: list[SubAgentTask] = []

    def execute(self, task: SubAgentTask, *, observer=None) -> SubAgentExecution:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.tasks.append(task)
        try:
            time.sleep(self.delay_seconds)
            if task.objective in self.failing_objectives:
                raise RuntimeError("synthetic delegation failure")
            with self._lock:
                self.completed_tasks.append(task)
            result = SubAgentResult(
                run_id=f"run-{len(task.objective)}-{task.role}",
                role=task.role,
                status="failed",
                stop_reason="model_error",
                summary=f"finished:{task.objective}",
                error="synthetic bounded result",
            )
            return SubAgentExecution(
                result=result,
                transitions=(),
                snapshot=None,
                context=None,
                token_usage=None,
                conversation_message_count=2,
                tool_call_count=0,
                validation_execution_count=0,
            )
        finally:
            with self._lock:
                self.active -= 1


class CoordinatedInterruptingRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tasks: list[SubAgentTask] = []
        self.sibling_completed = threading.Event()

    def execute(self, task: SubAgentTask, *, observer=None) -> SubAgentExecution:
        with self._lock:
            self.tasks.append(task)
        if task.objective == "Interrupt.":
            time.sleep(0.01)
            raise KeyboardInterrupt
        if task.objective == "Sibling.":
            time.sleep(0.04)
            self.sibling_completed.set()
        return SubAgentExecution(
            result=SubAgentResult(
                run_id=f"run-{task.objective.lower().strip('.')}",
                role=task.role,
                status="failed",
                stop_reason="model_error",
                summary=f"finished:{task.objective}",
                error="synthetic bounded result",
            ),
            transitions=(),
            snapshot=None,
            context=None,
            token_usage=None,
            conversation_message_count=2,
            tool_call_count=0,
            validation_execution_count=0,
        )


class SideEffectArgs(ToolArgs):
    text: str


class RecordingSideEffectTool(PydanticTool[SideEffectArgs]):
    name = "record_side_effect"
    description = "Record a call for barrier tests."
    args_model = SideEffectArgs
    capability = "read"
    risk = "low"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _run(self, args: SideEffectArgs) -> ToolResult:
        self.calls.append(args.text)
        return ToolResult.success("recorded")


class SubAgentActivityProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def run(self, name: str) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(name)
        time.sleep(0.03)
        with self._lock:
            self.active -= 1


class ConcurrentSubAgentTool(PydanticTool[SideEffectArgs]):
    name = "concurrent_subagent_tool"
    description = "Concurrent SubAgent test tool."
    args_model = SideEffectArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, name: str, probe: SubAgentActivityProbe) -> None:
        self.name = name
        self.probe = probe

    def _run(self, args: SideEffectArgs) -> ToolResult:
        self.probe.run(self.name)
        return ToolResult.success(f"{self.name}:{args.text}")


class RaisingSubAgentRegistry(ToolRegistry):
    async def run_tool_async(
        self,
        name,
        arguments,
        *,
        permission_lock,
    ) -> ToolResult:
        if name == "subagent_raising":
            raise RuntimeError("synthetic SubAgent async crash")
        return await super().run_tool_async(
            name,
            arguments,
            permission_lock=permission_lock,
        )


class RaisingDelegationRegistry(ToolRegistry):
    def run_tool(self, name, arguments) -> ToolResult:
        if arguments.get("objective") == "Worker crash.":
            raise RuntimeError("PRIVATE WORKER ERROR")
        return super().run_tool(name, arguments)


class RecordingParentLLM:
    last_token_usage = None

    def __init__(
        self,
        responses: list[AgentModelResponse] | None = None,
        stream_events: list[list[AgentEvent]] | None = None,
    ) -> None:
        self.responses = [] if responses is None else list(responses)
        self.stream_events = [] if stream_events is None else list(stream_events)
        self.seen_conversations: list[list[dict[str, object]]] = []

    def complete(self, conversation: Conversation) -> Message:
        raise NotImplementedError

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        raise NotImplementedError
        yield

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        self.seen_conversations.append(conversation.to_model_messages())
        if self.stream_events:
            yield from self.stream_events.pop(0)
            return
        response = self.responses.pop(0)
        if response.content:
            yield AgentEvent(type="text_delta", content=response.content)
        for tool_call in response.tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tool_call)


class InterruptingRuntime:
    def execute(self, task: SubAgentTask, *, observer=None) -> SubAgentExecution:
        raise KeyboardInterrupt
