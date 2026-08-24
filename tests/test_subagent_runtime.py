from collections.abc import Iterator
from pathlib import Path
import sys
import threading
import time

import pytest

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.context_budget import ContextBudget
from mycode.conversation import Conversation
from mycode.instructions import load_instruction_bundle
from mycode.memory import MemoryStore
from mycode.messages import Message
from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
)
from mycode.project import ProjectIdentity
from mycode.subagents.contracts import (
    ExplorerResult,
    SubAgentTask,
    TesterResult,
)
from mycode.subagents.delegate import DelegateTaskTool
from mycode.subagents.delegation import DelegationToolBatchHandler
from mycode.subagents.runtime import (
    DEFAULT_SUBAGENT_CONVERGENCE_REMAINING_TURNS,
    DEFAULT_SUBAGENT_MAX_TURNS,
    SubAgentRuntime,
    _collect_agent_events,
)
from mycode.tools import ToolRegistry
from mycode.tools.workspace import Workspace


def _stream_response(response: AgentModelResponse) -> Iterator[AgentEvent]:
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


def test_runtime_completes_explorer_with_independent_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project rule version one", encoding="utf-8")
    client = RecordingToolLLM([_explorer_submission("first result")])
    runtime = _runtime(workspace, lambda: client, tmp_path=tmp_path)

    execution = runtime.execute(
        SubAgentTask(role="explorer", objective="Find the runner implementation.")
    )

    assert execution.result.status == "completed"
    assert execution.result.stop_reason == "submitted"
    assert isinstance(execution.result.payload, ExplorerResult)
    assert execution.result.payload.summary == "first result"
    assert [transition.state for transition in execution.transitions] == [
        "running",
        "completed",
    ]
    assert execution.snapshot is not None
    assert execution.snapshot.instructions.sources[0].path == (
        workspace / "AGENTS.md"
    ).resolve()
    assert [schema["name"] for schema in client.seen_tools[0]] == [
        "read_file",
        "glob",
        "grep",
        "submit_result",
    ]
    assert [message["role"] for message in client.seen_conversations[0]] == [
        "system",
        "user",
    ]
    assert "project rule version one" in str(
        client.seen_conversations[0][0]["content"]
    )


def test_runtime_collector_keeps_only_the_terminal_turn_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project instructions", encoding="utf-8")
    client = RecordingToolLLM(
        [
            AgentModelResponse(
                content="early investigation notes",
                tool_calls=[
                    AgentToolCall(
                        id="call_read",
                        name="read_file",
                        arguments={"path": "AGENTS.md"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="terminal answer only"),
        ]
    )
    runtime = _runtime(workspace, lambda: client, tmp_path=tmp_path)

    execution = runtime.execute(
        SubAgentTask(role="explorer", objective="Inspect the project instructions.")
    )

    assert execution.result.status == "failed"
    assert execution.result.stop_reason == "invalid_result"
    assert execution.result.error is not None
    assert "20 characters" in execution.result.error
    assert "early investigation notes" not in execution.result.error


def test_collector_preserves_model_error_diagnostic() -> None:
    response = _collect_agent_events(
        iter(
            [
                AgentEvent(type="turn"),
                AgentEvent(type="model_start"),
                AgentEvent(type="error", error="provider unavailable"),
                AgentEvent(type="stop", stop_reason="model_error"),
            ]
        )
    )

    assert response.stop_reason == "model_error"
    assert response.content == "provider unavailable"


def test_collector_preserves_context_overflow_diagnostic() -> None:
    response = _collect_agent_events(
        iter(
            [
                AgentEvent(type="turn"),
                AgentEvent(
                    type="error",
                    error="context window exceeded configured budget",
                ),
                AgentEvent(type="stop", stop_reason="context_overflow"),
            ]
        )
    )

    assert response.stop_reason == "context_overflow"
    assert response.content == "context window exceeded configured budget"


def test_runtime_defaults_to_twenty_turns_and_three_turn_convergence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = _runtime(
        workspace,
        lambda: RecordingToolLLM([]),
        tmp_path=tmp_path,
    )

    assert DEFAULT_SUBAGENT_MAX_TURNS == 20
    assert DEFAULT_SUBAGENT_CONVERGENCE_REMAINING_TURNS == 3
    assert runtime.max_turns == 20
    assert runtime.convergence_remaining_turns == 3


@pytest.mark.parametrize(
    ("role", "role_guidance"),
    [
        ("explorer", "停止扩大搜索范围"),
        ("tester", "只完成最关键且尚未执行的验证"),
        ("reviewer", "只保留有证据的高优先级 findings"),
    ],
)
def test_runtime_uses_role_specific_convergence_prompt(
    tmp_path: Path,
    role: str,
    role_guidance: str,
) -> None:
    workspace = tmp_path / role
    workspace.mkdir()
    (workspace / "README.md").write_text("hello", encoding="utf-8")
    client = RecordingToolLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id=f"call_read_{role}",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="ordinary answer"),
        ]
    )
    runtime = _runtime(
        workspace,
        lambda: client,
        tmp_path=tmp_path,
        max_turns=4,
        convergence_remaining_turns=3,
    )

    runtime.execute(SubAgentTask(role=role, objective="Inspect the file."))

    second_context = client.seen_conversations[1]
    assert any(
        message["role"] == "system"
        and role_guidance in str(message["content"])
        and "submit_result" in str(message["content"])
        for message in second_context
    )
    assert "实施最小修改和关键验证" not in str(second_context)


def test_real_runtime_runs_all_builtin_roles_in_parallel_with_isolated_context(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    barrier = threading.Barrier(3)
    seen_by_role: dict[str, list[dict[str, object]]] = {}
    seen_lock = threading.Lock()

    runtime = _runtime(
        workspace,
        lambda: RoleAwareBarrierLLM(
            barrier=barrier,
            seen_by_role=seen_by_role,
            seen_lock=seen_lock,
        ),
        tmp_path=tmp_path,
    )
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        AgentToolCall(
            id="explorer-call",
            name="delegate_task",
            arguments={
                "role": "explorer",
                "objective": "Explore only this task.",
            },
        ),
        AgentToolCall(
            id="tester-call",
            name="delegate_task",
            arguments={
                "role": "tester",
                "objective": "Test only this task.",
            },
        ),
        AgentToolCall(
            id="reviewer-call",
            name="delegate_task",
            arguments={
                "role": "reviewer",
                "objective": "Review only this task.",
            },
        ),
    ]

    batch = DelegationToolBatchHandler(
        max_delegations_per_run=3,
        max_concurrent_delegations=3,
    )(registry, calls)

    assert all(execution.result.ok for execution in batch.executions)
    assert {
        execution.result.metadata["child_status"]
        for execution in batch.executions
    } == {"completed"}
    assert [
        execution.tool_call.id for execution in batch.executions
    ] == [call.id for call in calls]
    assert [
        execution.result.metadata["role"] for execution in batch.executions
    ] == ["explorer", "tester", "reviewer"]
    assert set(seen_by_role) == {"explorer", "tester", "reviewer"}
    assert "Explore only this task." in str(seen_by_role["explorer"])
    assert "Test only this task." not in str(seen_by_role["explorer"])
    assert "Test only this task." in str(seen_by_role["tester"])
    assert "Review only this task." not in str(seen_by_role["tester"])
    assert "Review only this task." in str(seen_by_role["reviewer"])
    assert "Explore only this task." not in str(seen_by_role["reviewer"])


def test_runtime_creates_new_runner_and_refreshes_snapshot_for_each_call(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction_path = workspace / "AGENTS.md"
    instruction_path.write_text("instruction version one", encoding="utf-8")
    first_client = RecordingToolLLM([_explorer_submission("first")])
    second_client = RecordingToolLLM([_explorer_submission("second")])
    clients = [first_client, second_client]
    runtime = _runtime(workspace, lambda: clients.pop(0), tmp_path=tmp_path)

    first = runtime.execute(SubAgentTask(role="explorer", objective="first task"))
    instruction_path.write_text("instruction version two", encoding="utf-8")
    second = runtime.execute(SubAgentTask(role="explorer", objective="second task"))

    assert first.result.run_id != second.result.run_id
    assert first.snapshot is not None
    assert second.snapshot is not None
    assert first.snapshot.combined_sha256 != second.snapshot.combined_sha256
    assert "first task" in str(first_client.seen_conversations[0][-1]["content"])
    assert "second task" not in str(first_client.seen_conversations[0])
    assert "second task" in str(second_client.seen_conversations[0][-1]["content"])
    assert "first task" not in str(second_client.seen_conversations[0])
    assert "instruction version one" in str(first_client.seen_conversations[0][0])
    assert "instruction version two" in str(second_client.seen_conversations[0][0])


def test_runtime_keeps_instruction_and_memory_snapshot_fixed_during_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction_path = workspace / "AGENTS.md"
    instruction_path.write_text("frozen instruction version one", encoding="utf-8")
    store = MemoryStore(
        ProjectIdentity.from_workspace(workspace),
        base_directory=tmp_path / "memory",
    )
    store.save(
        scope="project",
        kind="fact",
        key="test.command",
        content="frozen test command version one",
    )

    def mutate_sources() -> None:
        instruction_path.write_text("changed instruction version two", encoding="utf-8")
        store.save(
            scope="project",
            kind="fact",
            key="test.command",
            content="changed test command version two",
        )

    client = RecordingToolLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_read",
                        name="read_file",
                        arguments={"path": "AGENTS.md"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            _explorer_submission("snapshot stayed fixed"),
        ],
        before_first_response=mutate_sources,
    )
    runtime = _runtime(
        workspace,
        lambda: client,
        tmp_path=tmp_path,
        memory_store=store,
    )

    execution = runtime.execute(
        SubAgentTask(role="explorer", objective="Find the test command.")
    )

    assert execution.result.status == "completed"
    assert len(client.seen_conversations) == 2
    for conversation in client.seen_conversations:
        system_contents = [
            str(message["content"])
            for message in conversation
            if message["role"] == "system"
        ]
        assert any("frozen instruction version one" in item for item in system_contents)
        assert any("frozen test command version one" in item for item in system_contents)
        assert all("changed instruction version two" not in item for item in system_contents)
        assert all("changed test command version two" not in item for item in system_contents)


def test_tester_result_uses_real_validation_execution_and_confirmation_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.py").write_text("value = 1\n", encoding="utf-8")
    command = [sys.executable, "-m", "compileall", "-q", "."]
    client = RecordingToolLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation",
                        name="run_validation",
                        arguments={"command": command},
                    )
                ],
                stop_reason="tool_calls",
            ),
            _tester_submission(status="passed", summary="Compilation passed."),
        ]
    )
    confirmer = ApprovingConfirmer()
    observed_transitions = []
    runtime = _runtime(
        workspace,
        lambda: client,
        tmp_path=tmp_path,
        confirmer=confirmer,
    )

    execution = runtime.execute(
        SubAgentTask(role="tester", objective="Compile the project."),
        on_state_transition=observed_transitions.append,
    )

    assert execution.result.status == "completed"
    assert isinstance(execution.result.payload, TesterResult)
    payload = execution.result.payload
    assert payload.status == "passed"
    assert len(payload.executions) == 1
    assert payload.executions[0].command == command
    assert payload.executions[0].exit_code == 0
    assert payload.executions[0].duration_ms >= 0
    assert Path(payload.executions[0].cwd) == workspace.resolve()
    assert [transition.state for transition in execution.transitions] == [
        "running",
        "awaiting_confirmation",
        "running",
        "completed",
    ]
    assert observed_transitions == list(execution.transitions)
    assert len(confirmer.requests) == 1


def test_parallel_testers_serialize_confirmation_interaction(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model_barrier = threading.Barrier(2)
    command = [sys.executable, "-m", "pytest", "--version"]
    clients = [
        RecordingToolLLM(
            [
                AgentModelResponse(
                    tool_calls=[
                        AgentToolCall(
                            id=f"validation-{index}",
                            name="run_validation",
                            arguments={"command": command},
                        )
                    ],
                    stop_reason="tool_calls",
                ),
                _tester_submission(
                    status="passed",
                    summary=f"Tester {index} passed.",
                ),
            ],
            before_first_response=lambda: model_barrier.wait(timeout=3),
        )
        for index in range(2)
    ]
    confirmer = SlowApprovingConfirmer()
    runtime = _runtime(
        workspace,
        lambda: clients.pop(0),
        tmp_path=tmp_path,
        confirmer=confirmer,
    )
    registry = ToolRegistry.from_tools([DelegateTaskTool(runtime)])
    calls = [
        AgentToolCall(
            id=f"tester-call-{index}",
            name="delegate_task",
            arguments={
                "role": "tester",
                "objective": f"Run validator {index}.",
            },
        )
        for index in range(2)
    ]

    batch = DelegationToolBatchHandler(
        max_delegations_per_run=2,
        max_concurrent_delegations=2,
    )(registry, calls)

    assert all(execution.result.ok for execution in batch.executions)
    assert [
        execution.result.metadata["child_status"]
        for execution in batch.executions
    ] == ["completed", "completed"]
    assert [
        execution.result.metadata["validation_execution_count"]
        for execution in batch.executions
    ] == [1, 1]
    assert confirmer.max_active == 1
    assert len(confirmer.requests) == 2


def test_runtime_rejects_unproven_passed_result_then_allows_correction(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = RecordingToolLLM(
        [
            _tester_submission(status="passed", summary="Claimed pass."),
            _tester_submission(
                status="blocked",
                summary="No validation was executed.",
                blocked_reason="The required validator is unavailable.",
            ),
        ]
    )
    runtime = _runtime(workspace, lambda: client, tmp_path=tmp_path)

    execution = runtime.execute(
        SubAgentTask(role="tester", objective="Run the unavailable validator.")
    )

    assert execution.result.status == "completed"
    assert isinstance(execution.result.payload, TesterResult)
    assert execution.result.payload.status == "blocked"
    assert len(client.seen_conversations) == 2
    assert "successful execution" in str(client.seen_conversations[1])


def test_submit_result_barrier_skips_validation_from_same_model_response(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    confirmer = ApprovingConfirmer()
    client = RecordingToolLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_stale_validation",
                        name="run_validation",
                        arguments={
                            "command": [sys.executable, "-m", "compileall", "-q", "."]
                        },
                    ),
                    _tester_submit_call(
                        status="blocked",
                        summary="Validation is not needed for this task.",
                        blocked_reason="The task only requested static information.",
                    ),
                ],
                stop_reason="tool_calls",
            )
        ]
    )
    runtime = _runtime(
        workspace,
        lambda: client,
        tmp_path=tmp_path,
        confirmer=confirmer,
    )

    execution = runtime.execute(
        SubAgentTask(role="tester", objective="Report whether validation is needed.")
    )

    assert execution.result.status == "completed"
    assert isinstance(execution.result.payload, TesterResult)
    assert execution.result.payload.executions == []
    assert confirmer.requests == []
    assert execution.validation_execution_count == 0
    assert [transition.state for transition in execution.transitions] == [
        "running",
        "completed",
    ]


def test_multiple_submit_calls_are_rejected_as_a_batch_then_can_retry(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = _explorer_submit_call("first invalid batch result")
    second = _explorer_submit_call("second invalid batch result")
    client = RecordingToolLLM(
        [
            AgentModelResponse(tool_calls=[first, second], stop_reason="tool_calls"),
            _explorer_submission("single corrected result"),
        ]
    )
    runtime = _runtime(workspace, lambda: client, tmp_path=tmp_path)

    execution = runtime.execute(
        SubAgentTask(role="explorer", objective="Inspect the workspace.")
    )

    assert execution.result.status == "completed"
    assert isinstance(execution.result.payload, ExplorerResult)
    assert execution.result.payload.summary == "single corrected result"
    assert len(client.seen_conversations) == 2
    assert str(client.seen_conversations[1]).count("multiple_submission_barriers") == 2


def test_runtime_marks_keyboard_interrupt_without_resuming_old_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = RecordingToolLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_validation",
                        name="run_validation",
                        arguments={
                            "command": [sys.executable, "-m", "compileall", "-q", "."]
                        },
                    )
                ],
                stop_reason="tool_calls",
            )
        ]
    )
    runtime = _runtime(
        workspace,
        lambda: client,
        tmp_path=tmp_path,
        confirmer=InterruptingConfirmer(),
    )

    observed_transitions = []
    with pytest.raises(KeyboardInterrupt):
        runtime.execute(
            SubAgentTask(role="tester", objective="Compile the project."),
            on_state_transition=observed_transitions.append,
        )

    assert [transition.state for transition in observed_transitions] == [
        "running",
        "awaiting_confirmation",
        "interrupted",
    ]
    assert len(client.seen_conversations) == 1


def test_runtime_marks_system_exit_before_propagating_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = _runtime(
        workspace,
        lambda: SystemExitingLLM(exit_code=7),
        tmp_path=tmp_path,
    )
    observed_transitions = []

    with pytest.raises(SystemExit) as raised:
        runtime.execute(
            SubAgentTask(role="explorer", objective="Inspect the workspace."),
            on_state_transition=observed_transitions.append,
        )

    assert raised.value.code == 7
    assert [transition.state for transition in observed_transitions] == [
        "running",
        "interrupted",
    ]


def test_runtime_treats_plain_model_answer_as_invalid_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = RecordingToolLLM([AgentModelResponse(content="ordinary answer")])
    runtime = _runtime(workspace, lambda: client, tmp_path=tmp_path)

    execution = runtime.execute(
        SubAgentTask(role="reviewer", objective="Review the change.")
    )

    assert execution.result.status == "failed"
    assert execution.result.stop_reason == "invalid_result"
    assert execution.result.payload is None
    assert "ordinary answer" not in (execution.result.error or "")
    assert "sha256=" in (execution.result.error or "")


def test_runtime_distinguishes_max_turns_model_error_repetition_and_overflow(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello", encoding="utf-8")
    read_call = AgentToolCall(
        id="call_read_one",
        name="read_file",
        arguments={"path": "README.md"},
    )

    max_turns_client = RecordingToolLLM(
        [AgentModelResponse(tool_calls=[read_call], stop_reason="tool_calls")]
    )
    max_turns = _runtime(
        workspace,
        lambda: max_turns_client,
        tmp_path=tmp_path,
        max_turns=1,
        convergence_remaining_turns=None,
    ).execute(SubAgentTask(role="explorer", objective="Read the file."))
    assert max_turns.result.stop_reason == "max_turns"

    model_error_client = RecordingToolLLM(
        [AgentModelResponse(content="provider unavailable", stop_reason="model_error")]
    )
    model_error = _runtime(
        workspace,
        lambda: model_error_client,
        tmp_path=tmp_path,
    ).execute(SubAgentTask(role="reviewer", objective="Review the file."))
    assert model_error.result.stop_reason == "model_error"
    assert model_error.result.error == "provider unavailable"

    repeated_client = RecordingToolLLM(
        [
            AgentModelResponse(tool_calls=[read_call], stop_reason="tool_calls"),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_read_two",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                stop_reason="tool_calls",
            ),
        ]
    )
    repeated = _runtime(
        workspace,
        lambda: repeated_client,
        tmp_path=tmp_path,
        repeated_tool_call_limit=2,
    ).execute(SubAgentTask(role="explorer", objective="Read the file twice."))
    assert repeated.result.stop_reason == "repeated_tool_call"

    overflow_client = RecordingToolLLM([])
    overflow = _runtime(
        workspace,
        lambda: overflow_client,
        tmp_path=tmp_path,
        context_budget=ContextBudget(
            context_window_tokens=10,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
    ).execute(SubAgentTask(role="explorer", objective="Inspect the workspace."))
    assert overflow.result.stop_reason == "context_overflow"
    assert overflow.result.error is not None
    assert "estimated input context exceeds" in overflow.result.error
    assert overflow_client.seen_conversations == []


def test_runtime_rejects_memory_store_from_another_project(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other"
    workspace.mkdir()
    other_workspace.mkdir()
    other_store = MemoryStore(
        ProjectIdentity.from_workspace(other_workspace),
        base_directory=tmp_path / "memory",
    )

    with pytest.raises(ValueError, match="must match"):
        _runtime(
            workspace,
            lambda: RecordingToolLLM([]),
            tmp_path=tmp_path,
            memory_store=other_store,
        )


def test_runtime_reports_setup_failure_separately_from_model_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = _runtime(
        workspace,
        lambda: (_ for _ in ()).throw(RuntimeError("factory failed")),
        tmp_path=tmp_path,
    )

    execution = runtime.execute(
        SubAgentTask(role="explorer", objective="Inspect the workspace.")
    )

    assert execution.result.status == "failed"
    assert execution.result.stop_reason == "runtime_error"
    assert "factory failed" in (execution.result.error or "")
    assert [transition.state for transition in execution.transitions] == [
        "running",
        "failed",
    ]


def _runtime(
    workspace: Path,
    client_factory,
    *,
    tmp_path: Path,
    confirmer=None,
    memory_store: MemoryStore | None = None,
    **runtime_options,
) -> SubAgentRuntime:
    no_user_instructions = tmp_path / "no-user-instructions"
    return SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=client_factory,
        confirmer=confirmer,
        memory_store=memory_store,
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=no_user_instructions,
        ),
        **runtime_options,
    )


def _explorer_submission(summary: str) -> AgentModelResponse:
    return AgentModelResponse(
        tool_calls=[_explorer_submit_call(summary)],
        stop_reason="tool_calls",
    )


def _explorer_submit_call(summary: str) -> AgentToolCall:
    return AgentToolCall(
        id=f"call_{summary.replace(' ', '_')}",
        name="submit_result",
        arguments={
            "status": "no_match",
            "summary": summary,
            "searched_scope": ["."],
            "findings": [],
            "uncertainties": [],
        },
    )


def _tester_submission(
    *,
    status: str,
    summary: str,
    failure_summary: str | None = None,
    blocked_reason: str | None = None,
) -> AgentModelResponse:
    return AgentModelResponse(
        tool_calls=[
            _tester_submit_call(
                status=status,
                summary=summary,
                failure_summary=failure_summary,
                blocked_reason=blocked_reason,
            )
        ],
        stop_reason="tool_calls",
    )


def _tester_submit_call(
    *,
    status: str,
    summary: str,
    failure_summary: str | None = None,
    blocked_reason: str | None = None,
) -> AgentToolCall:
    arguments: dict[str, object] = {
        "status": status,
        "summary": summary,
        "uncertainties": [],
    }
    if failure_summary is not None:
        arguments["failure_summary"] = failure_summary
    if blocked_reason is not None:
        arguments["blocked_reason"] = blocked_reason
    return AgentToolCall(
        id=f"call_submit_{status}",
        name="submit_result",
        arguments=arguments,
    )


class RecordingToolLLM:
    last_token_usage = None

    def __init__(
        self,
        responses: list[AgentModelResponse],
        *,
        before_first_response=None,
    ) -> None:
        self.responses = list(responses)
        self.before_first_response = before_first_response
        self.seen_conversations: list[list[dict[str, object]]] = []
        self.seen_tools: list[list[dict[str, object]]] = []

    def complete(self, conversation: Conversation) -> Message:
        raise NotImplementedError

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        raise NotImplementedError

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        self.seen_conversations.append(conversation.to_model_messages())
        self.seen_tools.append(tools)
        if self.before_first_response is not None:
            callback = self.before_first_response
            self.before_first_response = None
            callback()
        if not self.responses:
            raise RuntimeError("No fake SubAgent response remains.")
        yield from _stream_response(self.responses.pop(0))


class RoleAwareBarrierLLM:
    last_token_usage = None

    def __init__(
        self,
        *,
        barrier: threading.Barrier,
        seen_by_role: dict[str, list[dict[str, object]]],
        seen_lock: threading.Lock,
    ) -> None:
        self.barrier = barrier
        self.seen_by_role = seen_by_role
        self.seen_lock = seen_lock

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
        names = {str(tool["name"]) for tool in tools}
        if "run_validation" in names:
            role = "tester"
            response = _tester_submission(
                status="blocked",
                summary="Tester completed its isolated response.",
                blocked_reason="No command was needed for this concurrency probe.",
            )
        elif "inspect_changes" in names:
            role = "reviewer"
            response = AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="reviewer-submit",
                        name="submit_result",
                        arguments={
                            "recommendation": "approve",
                            "summary": "Reviewer completed its isolated response.",
                            "reviewed_scope": ["."],
                            "findings": [],
                            "uncertainties": [],
                        },
                    )
                ],
                stop_reason="tool_calls",
            )
        else:
            role = "explorer"
            response = _explorer_submission(
                "Explorer completed its isolated response."
            )

        with self.seen_lock:
            self.seen_by_role[role] = conversation.to_model_messages()
        self.barrier.wait(timeout=3)
        yield from _stream_response(response)


class ApprovingConfirmer:
    def __init__(self) -> None:
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return ConfirmationResult.approved()


class SlowApprovingConfirmer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[ConfirmationRequest] = []
        self.active = 0
        self.max_active = 0

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(request)
        try:
            time.sleep(0.04)
            return ConfirmationResult.approved()
        finally:
            with self._lock:
                self.active -= 1


class InterruptingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        raise KeyboardInterrupt


class SystemExitingLLM:
    last_token_usage = None

    def __init__(self, *, exit_code: int) -> None:
        self.exit_code = exit_code

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
        raise SystemExit(self.exit_code)
        yield
