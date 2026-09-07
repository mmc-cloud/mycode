from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
import sys
import threading

import pytest

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.conversation import Conversation
from mycode.instructions import load_instruction_bundle
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.session_store import SessionStore
from mycode.filesystem import read_jsonl_records
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


def test_observer_runtime_jsonl_safe_audit_and_multiple_runs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("PRIVATE FILE TOOL OUTPUT")
    (workspace / "AGENTS.md").write_text("PRIVATE PROJECT INSTRUCTION BODY")
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "projects")
    outputs = []
    with store.open_session(project, "parent", create=True) as session:
        observer = CompositeSubAgentObserver(observers=(
            SessionSubAgentObserver(session), CliSubAgentObserver(outputs.append, mode="debug"),
        ))
        for run_id in ("first", "second"):
            client = ScriptedSubAgentLLM([
                AgentModelResponse(tool_calls=[
                    AgentToolCall(id="read", name="read_file", arguments={"path": "README.md"}),
                    AgentToolCall(id="unknown", name="PRIVATE_TOOL_NAME", arguments={"PRIVATE_KEY": "PRIVATE ARGUMENT VALUE"}),
                ], stop_reason="tool_calls"),
                _explorer_submission("Safe persisted summary."),
            ])
            result = _runtime(workspace, tmp_path, client, run_id=run_id).execute(
                SubAgentTask(role="explorer", objective="PRIVATE TASK", context="PRIVATE CONTEXT", scope_paths=["README.md"]),
                observer=observer,
            )
            assert result.result.status == "completed"
            events = read_jsonl_records(store.project_storage(project).projects_root, session.layout.subagent_log_path(run_id))
            assert all(e["version"] == 1 for e in events)
            assert {e["type"] for e in events} == {"state", "snapshot", "tool_audit", "result"}
            assert events[0]["task_sha256"]
            assert events[-1]["result"]["summary"] == "Safe persisted summary."
            audits = [e for e in events if e["type"] == "tool_audit"]
            assert [e["tool_name"] for e in audits] == ["read_file", "unknown", "submit_result"]
            assert audits[0]["argument_summary"]["path_sha256"]
            raw = json.dumps(events)
            for secret in ("PRIVATE FILE TOOL OUTPUT", "PRIVATE PROJECT INSTRUCTION BODY", "PRIVATE TASK", "PRIVATE CONTEXT", "PRIVATE_TOOL_NAME", "PRIVATE_KEY", "PRIVATE ARGUMENT VALUE"):
                assert secret not in raw
        assert session.layout.transcript_path.read_bytes() == b""
        assert len(list(session.layout.subagents_directory.iterdir())) == 2


def test_parallel_runtime_jsonl_events_remain_isolated(tmp_path):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    barrier = threading.Barrier(3)
    run_numbers = itertools.count(1)
    runtime = SubAgentRuntime(
        workspace=Workspace(tmp_path),
        llm_client_factory=lambda: BarrierExplorerLLM(barrier),
        instruction_loader=lambda root, working: load_instruction_bundle(
            root, working_directory=working, user_instruction_directory=tmp_path / "no-instructions",
        ),
        run_id_factory=lambda: f"parallel-{next(run_numbers)}",
    )
    with store.open_session(project, "parent", create=True) as session:
        registry = ToolRegistry.from_tools([DelegateTaskTool(runtime, observer=SessionSubAgentObserver(session))])
        calls = [AgentToolCall(id=f"call-{i}", name="delegate_task", arguments={
            "role": "explorer", "objective": f"Inspect scope {i}",
        }) for i in range(3)]
        batch = DelegationToolBatchHandler(max_delegations_per_run=3, max_concurrent_delegations=3)(registry, calls)
        assert all(e.result.ok for e in batch.executions)
        for run_id in ("parallel-1", "parallel-2", "parallel-3"):
            events = read_jsonl_records(session.storage.projects_root, session.layout.subagent_log_path(run_id))
            assert [e["state"] for e in events if e["type"] == "state"] == ["running", "completed"]


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
