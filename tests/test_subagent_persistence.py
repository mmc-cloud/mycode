from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.conversation import Conversation
from mycode.instructions import load_instruction_bundle
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.session_deletion import SessionDeletionManager
from mycode.session_store import (
    SESSION_SCHEMA_VERSION,
    SessionLeaseLostError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.subagents.cli_observer import CliSubAgentObserver
from mycode.subagents.delegate import DelegateTaskTool
from mycode.subagents.delegation import DelegationToolBatchHandler
from mycode.subagents.observability import CompositeSubAgentObserver
from mycode.subagents.persistence import SessionSubAgentObserver
from mycode.subagents.runtime import SubAgentRuntime
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.snapshots import (
    InstructionSnapshotMetadata,
    InstructionSourceFingerprint,
    MemoryEntryFingerprint,
    MemorySnapshotMetadata,
    MemorySourceFingerprint,
    SubAgentSnapshotMetadata,
)
from mycode.tools.workspace import Workspace
from mycode.tools import ToolRegistry


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


def test_schema_v2_migrates_subagent_tables_without_losing_sessions(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    original = SessionStore(database_path)
    original.create_session(project, session_id="legacy")
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE subagent_tool_audits")
        connection.execute("DROP TABLE subagent_events")
        connection.execute("DROP TABLE subagent_runs")
        connection.execute("DROP INDEX sessions_id_project_idx")
        connection.execute("PRAGMA user_version = 2")

    migrated = SessionStore(database_path)

    assert migrated.get_session(project, "legacy") is not None
    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert version == SESSION_SCHEMA_VERSION
    assert {
        "subagent_runs",
        "subagent_events",
        "subagent_tool_audits",
    }.issubset(tables)


def test_runtime_observer_persists_safe_run_and_cli_events_without_raw_material(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "PRIVATE PROJECT INSTRUCTION BODY",
        encoding="utf-8",
    )
    (workspace / "README.md").write_text(
        "PRIVATE FILE TOOL OUTPUT",
        encoding="utf-8",
    )
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    outputs: list[str] = []
    observer = CompositeSubAgentObserver(
        observers=(
            SessionSubAgentObserver(
                store=store,
                project=project,
                parent_session_id=parent.id,
                lease_owner_id="owner",
            ),
            CliSubAgentObserver(outputs.append, mode="debug"),
        )
    )
    client = ScriptedSubAgentLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="read",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                    AgentToolCall(
                        id="unknown",
                        name="PRIVATE_TOOL_NAME",
                        arguments={"PRIVATE_ARGUMENT_KEY": "PRIVATE ARGUMENT VALUE"},
                    ),
                ],
                stop_reason="tool_calls",
            ),
            _explorer_submission("Safe persisted summary."),
        ]
    )
    runtime = _runtime(
        workspace,
        tmp_path,
        client,
        run_id="observed-run",
    )

    execution = runtime.execute(
        SubAgentTask(
            role="explorer",
            objective="PRIVATE DELEGATED TASK BODY",
            context="PRIVATE PARENT CONTEXT BODY",
            scope_paths=["README.md"],
        ),
        observer=observer,
    )

    assert execution.result.status == "completed"
    run = store.get_subagent_run(project, parent.id, "observed-run")
    assert run is not None
    assert run.status == "completed"
    assert run.stop_reason == "submitted"
    assert run.task_sha256 != ""
    assert run.objective_chars == len("PRIVATE DELEGATED TASK BODY")
    assert run.context_chars == len("PRIVATE PARENT CONTEXT BODY")
    assert run.snapshot is not None
    assert run.result is not None
    assert run.result["summary"] == "Safe persisted summary."
    events = store.list_subagent_events(project, parent.id, "observed-run")
    assert [event.event_type for event in events] == [
        "state",
        "snapshot",
        "state",
        "result",
    ]
    snapshot_event = next(event for event in events if event.event_type == "snapshot")
    assert set(snapshot_event.data) == {"combined_sha256", "snapshot_chars"}
    audits = store.list_subagent_tool_audits(project, parent.id, "observed-run")
    assert [audit.tool_name for audit in audits] == [
        "read_file",
        "unknown",
        "submit_result",
    ]
    assert audits[0].argument_summary["path_sha256"]
    assert "README.md" not in json.dumps(audits[0].argument_summary)
    assert audits[1].argument_summary["requested_tool_name_sha256"]
    assert audits[1].argument_summary["argument_keys_sha256"]
    persisted_text = _all_subagent_text(store.database_path)
    rendered_output = "\n".join(outputs)
    for secret in (
        "PRIVATE PROJECT INSTRUCTION BODY",
        "PRIVATE FILE TOOL OUTPUT",
        "PRIVATE DELEGATED TASK BODY",
        "PRIVATE PARENT CONTEXT BODY",
        "PRIVATE_TOOL_NAME",
        "PRIVATE_ARGUMENT_KEY",
        "PRIVATE ARGUMENT VALUE",
    ):
        assert secret not in persisted_text
        assert secret not in rendered_output
    assert any(line.startswith("subagent> start") for line in outputs)
    assert any(line.startswith("subagent_context>") for line in outputs)
    assert any("name=read_file" in line for line in outputs)
    assert any(line.startswith("subagent_result>") for line in outputs)


def test_parallel_runtime_persists_each_run_and_atomic_cli_events(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parallel-parent",
        lease_owner_id="owner",
    )
    outputs: list[str] = []
    observer = CompositeSubAgentObserver(
        observers=(
            SessionSubAgentObserver(
                store=store,
                project=project,
                parent_session_id=parent.id,
                lease_owner_id="owner",
            ),
            CliSubAgentObserver(outputs.append),
        )
    )
    barrier = threading.Barrier(3)
    run_numbers = itertools.count(1)
    runtime = SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=lambda: BarrierExplorerLLM(barrier),
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=tmp_path / "no-user-instructions",
        ),
        run_id_factory=lambda: f"parallel-run-{next(run_numbers)}",
    )
    registry = ToolRegistry.from_tools(
        [DelegateTaskTool(runtime, observer=observer)]
    )
    calls = [
        AgentToolCall(
            id=f"parallel-call-{index}",
            name="delegate_task",
            arguments={
                "role": "explorer",
                "objective": f"Inspect independent scope {index}.",
            },
        )
        for index in range(3)
    ]

    batch = DelegationToolBatchHandler(
        max_delegations_per_run=3,
        max_concurrent_delegations=3,
    )(registry, calls)

    assert all(execution.result.ok for execution in batch.executions)
    runs = store.list_subagent_runs(project, parent.id)
    assert len(runs) == 3
    assert {run.id for run in runs} == {
        "parallel-run-1",
        "parallel-run-2",
        "parallel-run-3",
    }
    assert {run.status for run in runs} == {"completed"}
    for run in runs:
        assert [
            event.state
            for event in store.list_subagent_events(
                project,
                parent.id,
                run.id,
            )
            if event.event_type == "state"
        ] == ["running", "completed"]
    assert sum(line == "activity> Explorer started" for line in outputs) == 3
    assert sum(
        line.startswith("activity> Explorer completed:") for line in outputs
    ) == 3
    assert not any(line.startswith("subagent_tool>") for line in outputs)
    assert all("\n" not in line for line in outputs)


def test_subagent_storage_requires_parent_lease_and_project_scope(
    tmp_path: Path,
) -> None:
    project_a_root = tmp_path / "a"
    project_b_root = tmp_path / "b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project_a,
        session_id="parent",
        lease_owner_id="owner-a",
    )

    with pytest.raises(SessionLeaseLostError, match="no longer owned"):
        _create_run(store, project_a, parent.id, "wrong-owner", owner="owner-b")
    _create_run(store, project_a, parent.id, "owned-run", owner="owner-a")

    with pytest.raises(SessionNotFoundError, match="current project"):
        store.get_subagent_run(project_b, parent.id, "owned-run")


def test_real_validation_persists_exit_code_and_duration_without_stdout(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.py").write_text("value = 1\n", encoding="utf-8")
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    observer = SessionSubAgentObserver(
        store=store,
        project=project,
        parent_session_id=parent.id,
        lease_owner_id="owner",
    )
    command = [sys.executable, "-m", "compileall", "-q", "."]
    client = ScriptedSubAgentLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="validation",
                        name="run_validation",
                        arguments={"command": command},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="submit",
                        name="submit_result",
                        arguments={
                            "status": "passed",
                            "summary": "Compilation passed.",
                            "uncertainties": [],
                        },
                    )
                ],
                stop_reason="tool_calls",
            ),
        ]
    )
    runtime = SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=lambda: client,
        confirmer=ApprovingConfirmer(),
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=tmp_path / "no-user-instructions",
        ),
        run_id_factory=lambda: "validation-run",
    )

    execution = runtime.execute(
        SubAgentTask(role="tester", objective="Compile the project."),
        observer=observer,
    )

    assert execution.result.status == "completed"
    audits = store.list_subagent_tool_audits(
        project,
        parent.id,
        "validation-run",
    )
    validation = audits[0]
    assert validation.tool_name == "run_validation"
    assert validation.exit_code == 0
    assert validation.duration_ms is not None
    assert validation.duration_ms >= 0
    assert validation.output_chars < 1000
    assert validation.argument_summary["command_parts"] == len(command)
    assert validation.argument_summary["command_sha256"]
    run = store.get_subagent_run(project, parent.id, "validation-run")
    assert run is not None
    assert run.result["payload"]["executions"][0]["exit_code"] == 0


def test_confirmation_interrupt_is_persisted_without_resuming_child(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    observer = SessionSubAgentObserver(
        store=store,
        project=project,
        parent_session_id=parent.id,
        lease_owner_id="owner",
    )
    client = ScriptedSubAgentLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="validation",
                        name="run_validation",
                        arguments={
                            "command": [
                                sys.executable,
                                "-m",
                                "compileall",
                                "-q",
                                ".",
                            ]
                        },
                    )
                ],
                stop_reason="tool_calls",
            )
        ]
    )
    runtime = SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=lambda: client,
        confirmer=InterruptingConfirmer(),
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=tmp_path / "no-user-instructions",
        ),
        run_id_factory=lambda: "interrupted-run",
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.execute(
            SubAgentTask(role="tester", objective="Compile the project."),
            observer=observer,
        )

    run = store.get_subagent_run(project, parent.id, "interrupted-run")
    assert run is not None
    assert run.status == "interrupted"
    assert run.stop_reason == "interrupted"
    assert run.result is None
    states = [
        event.state
        for event in store.list_subagent_events(
            project,
            parent.id,
            "interrupted-run",
        )
        if event.event_type == "state"
    ]
    assert states == ["running", "awaiting_confirmation", "interrupted"]
    assert store.list_subagent_tool_audits(
        project,
        parent.id,
        "interrupted-run",
    ) == []


def test_system_exit_is_persisted_as_interrupted_before_propagation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    observer = SessionSubAgentObserver(
        store=store,
        project=project,
        parent_session_id=parent.id,
        lease_owner_id="owner",
    )
    runtime = SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=lambda: SystemExitingSubAgentLLM(exit_code=7),
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=tmp_path / "no-user-instructions",
        ),
        run_id_factory=lambda: "system-exit-run",
    )

    with pytest.raises(SystemExit) as raised:
        runtime.execute(
            SubAgentTask(role="explorer", objective="Inspect the project."),
            observer=observer,
        )

    assert raised.value.code == 7
    run = store.get_subagent_run(project, parent.id, "system-exit-run")
    assert run is not None
    assert run.status == "interrupted"
    assert run.stop_reason == "interrupted"
    assert run.result is None
    states = [
        event.state
        for event in store.list_subagent_events(
            project,
            parent.id,
            "system-exit-run",
        )
        if event.event_type == "state"
    ]
    assert states == ["running", "interrupted"]
    assert store.list_subagent_tool_audits(
        project,
        parent.id,
        "system-exit-run",
    ) == []


def test_subagent_event_and_tool_audit_limits_record_omissions(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(
        tmp_path / "state.sqlite3",
        subagent_event_limit=2,
        subagent_tool_audit_limit=1,
    )
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    _create_run(store, project, parent.id, "limited", owner="owner")
    now = datetime.now(timezone.utc)
    store.update_subagent_snapshot(
        project,
        parent.id,
        "limited",
        snapshot={"combined_sha256": "b" * 64},
        occurred_at=now,
        lease_owner_id="owner",
    )
    store.append_subagent_state(
        project,
        parent.id,
        "limited",
        state="awaiting_confirmation",
        reason="permission_confirmation",
        occurred_at=now,
        lease_owner_id="owner",
    )
    for sequence in range(2):
        store.append_subagent_tool_audit(
            project,
            parent.id,
            "limited",
            tool_name="read_file",
            arguments_sha256=f"{sequence + 1:064x}",
            argument_summary={"path_sha256": "c" * 64},
            ok=True,
            exit_code=None,
            duration_ms=None,
            output_chars=10,
            truncated=False,
            reason=None,
            occurred_at=now,
            lease_owner_id="owner",
        )
    store.finalize_subagent_run(
        project,
        parent.id,
        "limited",
        status="failed",
        stop_reason="model_error",
        result={"run_id": "limited", "status": "failed"},
        context=None,
        token_usage=None,
        tool_call_count=2,
        validation_execution_count=0,
        occurred_at=now,
        lease_owner_id="owner",
    )

    run = store.get_subagent_run(project, parent.id, "limited")
    assert run is not None
    assert len(store.list_subagent_events(project, parent.id, "limited")) == 2
    assert len(store.list_subagent_tool_audits(project, parent.id, "limited")) == 1
    assert run.omitted_event_count == 2
    assert run.omitted_tool_audit_count == 1


def test_large_snapshot_metadata_is_hash_bounded_before_storage(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    _create_run(store, project, parent.id, "snapshot-run", owner="owner")
    now = datetime.now(timezone.utc)
    source_hashes = tuple(f"{index + 1:064x}" for index in range(25))
    snapshot = SubAgentSnapshotMetadata(
        loaded_at=now,
        combined_sha256="f" * 64,
        instructions=InstructionSnapshotMetadata(
            loaded_at=now,
            total_chars=25,
            combined_sha256="e" * 64,
            sources=tuple(
                InstructionSourceFingerprint(
                    scope="project",
                    path=tmp_path / f"instruction-{index}.md",
                    content_bytes=1,
                    content_chars=1,
                    sha256=digest,
                )
                for index, digest in enumerate(source_hashes)
            ),
        ),
        memory=MemorySnapshotMetadata(
            enabled=True,
            loaded_at=now,
            combined_sha256="d" * 64,
            sources=tuple(
                MemorySourceFingerprint(
                    scope="project",
                    path=tmp_path / f"memory-{index}.md",
                    content_bytes=1,
                    content_chars=1,
                    sha256=digest,
                )
                for index, digest in enumerate(source_hashes)
            ),
            selected_entries=tuple(
                MemoryEntryFingerprint(
                    scope="project",
                    kind="fact",
                    content_chars=1,
                    sha256=digest,
                )
                for digest in source_hashes
            ),
        ),
    )
    observer = SessionSubAgentObserver(
        store=store,
        project=project,
        parent_session_id=parent.id,
        lease_owner_id="owner",
    )

    observer.on_snapshot(
        SubAgentTask(role="explorer", objective="Inspect."),
        "snapshot-run",
        snapshot,
        now,
    )

    run = store.get_subagent_run(project, parent.id, "snapshot-run")
    assert run is not None
    assert len(run.snapshot["instructions"]["source_sha256"]) == 12
    assert run.snapshot["instructions"]["source_hashes_omitted"] == 13
    assert len(run.snapshot["memory"]["source_sha256"]) == 12
    assert len(run.snapshot["memory"]["selected_entry_sha256"]) == 12
    assert "instruction-" not in json.dumps(run.snapshot)
    assert "memory-" not in json.dumps(run.snapshot)


def test_subagent_retention_removes_only_old_terminal_runs(tmp_path: Path) -> None:
    current = datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc)
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(
        tmp_path / "state.sqlite3",
        now=lambda: current,
        subagent_run_retention=2,
    )
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
        lease_duration_seconds=3600,
    )
    for index in range(3):
        occurred = current + timedelta(seconds=index)
        _create_run(
            store,
            project,
            parent.id,
            f"run-{index}",
            owner="owner",
            occurred_at=occurred,
        )
        store.finalize_subagent_run(
            project,
            parent.id,
            f"run-{index}",
            status="failed",
            stop_reason="model_error",
            result={"run_id": f"run-{index}", "status": "failed"},
            context=None,
            token_usage=None,
            tool_call_count=0,
            validation_execution_count=0,
            occurred_at=occurred,
            lease_owner_id="owner",
        )

    assert [run.id for run in store.list_subagent_runs(project, parent.id)] == [
        "run-2",
        "run-1",
    ]


def test_parent_release_interrupts_unfinished_child_without_resuming_it(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    _create_run(store, project, parent.id, "unfinished", owner="owner")

    store.release_session_lease(project, parent.id, "owner", "interrupted")
    store.acquire_session_lease(project, parent.id, "new-owner")

    run = store.get_subagent_run(project, parent.id, "unfinished")
    assert run is not None
    assert run.status == "interrupted"
    assert run.stop_reason == "interrupted"
    assert run.result is None
    assert store.list_subagent_events(project, parent.id, "unfinished")[-1].reason == (
        "parent_session_interrupted"
    )


def test_expired_parent_lease_cleanup_is_project_scoped_for_subagents(
    tmp_path: Path,
) -> None:
    project_a_root = tmp_path / "a"
    project_b_root = tmp_path / "b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    current = [datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)]
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    parent_a = store.create_session(
        project_a,
        session_id="a-parent",
        lease_owner_id="a-owner",
        lease_duration_seconds=10,
    )
    parent_b = store.create_session(
        project_b,
        session_id="b-parent",
        lease_owner_id="b-owner",
        lease_duration_seconds=10,
    )
    _create_run(store, project_a, parent_a.id, "a-run", owner="a-owner")
    _create_run(store, project_b, parent_b.id, "b-run", owner="b-owner")
    current[0] += timedelta(seconds=11)

    assert store.expire_session_leases(project_a) == 1

    assert store.get_subagent_run(project_a, parent_a.id, "a-run").status == (
        "interrupted"
    )
    assert store.get_subagent_run(project_b, parent_b.id, "b-run").status == "running"


def test_resume_after_expired_lease_interrupts_but_does_not_restore_child(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    current = [datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc)]
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="old-owner",
        lease_duration_seconds=10,
    )
    _create_run(store, project, parent.id, "orphan", owner="old-owner")
    current[0] += timedelta(seconds=11)

    resumed = store.acquire_session_lease(project, parent.id, "new-owner")

    assert resumed.status == "active"
    run = store.get_subagent_run(project, parent.id, "orphan")
    assert run is not None
    assert run.status == "interrupted"
    assert run.result is None
    assert store.list_subagent_events(project, parent.id, "orphan")[-1].reason == (
        "parent_session_resumed"
    )
    with pytest.raises(SessionStoreError, match="already terminal"):
        store.finalize_subagent_run(
            project,
            parent.id,
            "orphan",
            status="completed",
            stop_reason="submitted",
            result={"run_id": "orphan", "status": "completed"},
            context=None,
            token_usage=None,
            tool_call_count=0,
            validation_execution_count=0,
            occurred_at=current[0],
            lease_owner_id="new-owner",
        )
    assert store.get_subagent_run(project, parent.id, "orphan").result is None


def test_parent_resume_applies_retention_after_interrupting_orphans(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    current = [datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)]
    store = SessionStore(
        tmp_path / "state.sqlite3",
        now=lambda: current[0],
        subagent_run_retention=2,
    )
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="old-owner",
        lease_duration_seconds=10,
    )
    for index in range(3):
        _create_run(
            store,
            project,
            parent.id,
            f"orphan-{index}",
            owner="old-owner",
            occurred_at=current[0] + timedelta(seconds=index),
        )
    current[0] += timedelta(seconds=11)

    store.acquire_session_lease(project, parent.id, "new-owner")

    runs = store.list_subagent_runs(project, parent.id)
    assert len(runs) == 2
    assert all(run.status == "interrupted" for run in runs)


def test_subagent_records_cascade_when_parent_session_is_deleted(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    parent = store.create_session(
        project,
        session_id="parent",
        lease_owner_id="owner",
    )
    _create_run(store, project, parent.id, "child", owner="owner")
    store.append_subagent_tool_audit(
        project,
        parent.id,
        "child",
        tool_name="read_file",
        arguments_sha256="a" * 64,
        argument_summary={"path_sha256": "b" * 64},
        ok=True,
        exit_code=None,
        duration_ms=1,
        output_chars=10,
        truncated=False,
        reason=None,
        occurred_at=datetime.now(timezone.utc),
        lease_owner_id="owner",
    )
    store.release_session_lease(project, parent.id, "owner", "closed")

    assert (
        SessionDeletionManager(store)
        .request_and_process(project, parent.id)
        .completed
        is True
    )

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM subagent_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM subagent_events").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM subagent_tool_audits"
            ).fetchone()[0]
            == 0
        )


def _create_run(
    store: SessionStore,
    project: ProjectIdentity,
    parent_session_id: str,
    run_id: str,
    *,
    owner: str,
    occurred_at: datetime | None = None,
) -> None:
    store.create_subagent_run(
        project,
        parent_session_id,
        run_id=run_id,
        role="explorer",
        task_sha256="a" * 64,
        objective_chars=10,
        context_chars=0,
        scope_path_count=1,
        reason="run_started",
        occurred_at=(
            datetime.now(timezone.utc) if occurred_at is None else occurred_at
        ),
        lease_owner_id=owner,
    )


def _runtime(
    workspace: Path,
    tmp_path: Path,
    client,
    *,
    run_id: str,
) -> SubAgentRuntime:
    return SubAgentRuntime(
        workspace=Workspace(workspace),
        llm_client_factory=lambda: client,
        instruction_loader=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
            user_instruction_directory=tmp_path / "no-user-instructions",
        ),
        run_id_factory=lambda: run_id,
    )


def _explorer_submission(summary: str) -> AgentModelResponse:
    return AgentModelResponse(
        tool_calls=[
            AgentToolCall(
                id="submit",
                name="submit_result",
                arguments={
                    "status": "no_match",
                    "summary": summary,
                    "searched_scope": ["README.md"],
                    "findings": [],
                    "uncertainties": [],
                },
            )
        ],
        stop_reason="tool_calls",
    )


def _all_subagent_text(database_path: Path) -> str:
    with sqlite3.connect(database_path) as connection:
        run_rows = connection.execute(
            """
            SELECT task_sha256, snapshot_json, context_json, token_usage_json,
                   result_json FROM subagent_runs
            """
        ).fetchall()
        event_rows = connection.execute(
            "SELECT state, reason, data_json FROM subagent_events"
        ).fetchall()
        audit_rows = connection.execute(
            """
            SELECT tool_name, arguments_sha256, argument_summary_json, reason
            FROM subagent_tool_audits
            """
        ).fetchall()
    return repr([*run_rows, *event_rows, *audit_rows])


class ScriptedSubAgentLLM:
    last_token_usage = None

    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = list(responses)

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
        yield from _stream_response(self.responses.pop(0))


class BarrierExplorerLLM:
    last_token_usage = None

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier

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
        self.barrier.wait(timeout=3)
        yield from _stream_response(_explorer_submission("Parallel persisted summary."))


class SystemExitingSubAgentLLM:
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


class ApprovingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.approved()


class InterruptingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        raise KeyboardInterrupt
