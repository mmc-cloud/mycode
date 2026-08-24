from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TYPE_CHECKING

from mycode.subagents.concurrency import SubAgentInteractionGate
from mycode.subagents.audit import SubAgentToolAudit
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.lifecycle import SubAgentStateTransition
from mycode.subagents.snapshots import SubAgentSnapshotMetadata

if TYPE_CHECKING:
    from mycode.subagents.runtime import SubAgentExecution


class SubAgentObserver(Protocol):
    def on_state(
        self,
        task: SubAgentTask,
        transition: SubAgentStateTransition,
    ) -> None:
        pass

    def on_snapshot(
        self,
        task: SubAgentTask,
        run_id: str,
        snapshot: SubAgentSnapshotMetadata,
        occurred_at: datetime,
    ) -> None:
        pass

    def on_tool_audit(
        self,
        task: SubAgentTask,
        run_id: str,
        audit: SubAgentToolAudit,
        occurred_at: datetime,
    ) -> None:
        pass

    def on_result(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
        occurred_at: datetime,
    ) -> None:
        pass


@dataclass(frozen=True)
class CompositeSubAgentObserver:
    observers: tuple[SubAgentObserver, ...]

    def on_state(
        self,
        task: SubAgentTask,
        transition: SubAgentStateTransition,
    ) -> None:
        for observer in self.observers:
            observer.on_state(task, transition)

    def on_snapshot(
        self,
        task: SubAgentTask,
        run_id: str,
        snapshot: SubAgentSnapshotMetadata,
        occurred_at: datetime,
    ) -> None:
        for observer in self.observers:
            observer.on_snapshot(task, run_id, snapshot, occurred_at)

    def on_tool_audit(
        self,
        task: SubAgentTask,
        run_id: str,
        audit: SubAgentToolAudit,
        occurred_at: datetime,
    ) -> None:
        for observer in self.observers:
            observer.on_tool_audit(task, run_id, audit, occurred_at)

    def on_result(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
        occurred_at: datetime,
    ) -> None:
        for observer in self.observers:
            observer.on_result(task, execution, occurred_at)


@dataclass(frozen=True)
class SynchronizedSubAgentObserver:
    """Keep one observer callback atomic with confirmation interaction."""

    delegate: SubAgentObserver
    interaction_gate: SubAgentInteractionGate

    def on_state(
        self,
        task: SubAgentTask,
        transition: SubAgentStateTransition,
    ) -> None:
        self.interaction_gate.run(
            lambda: self.delegate.on_state(task, transition)
        )

    def on_snapshot(
        self,
        task: SubAgentTask,
        run_id: str,
        snapshot: SubAgentSnapshotMetadata,
        occurred_at: datetime,
    ) -> None:
        self.interaction_gate.run(
            lambda: self.delegate.on_snapshot(
                task,
                run_id,
                snapshot,
                occurred_at,
            )
        )

    def on_tool_audit(
        self,
        task: SubAgentTask,
        run_id: str,
        audit: SubAgentToolAudit,
        occurred_at: datetime,
    ) -> None:
        self.interaction_gate.run(
            lambda: self.delegate.on_tool_audit(
                task,
                run_id,
                audit,
                occurred_at,
            )
        )

    def on_result(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
        occurred_at: datetime,
    ) -> None:
        self.interaction_gate.run(
            lambda: self.delegate.on_result(task, execution, occurred_at)
        )
