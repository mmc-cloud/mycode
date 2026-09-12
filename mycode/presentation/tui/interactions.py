"""Thread-to-UI bridges for synchronous presentation interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Lock
from typing import TYPE_CHECKING, Callable

from textual.message import Message

from mycode.mcp.trust import MCPTrustRequest, MCPTrustWarning
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.lifecycle import SubAgentStateTransition

if TYPE_CHECKING:
    from mycode.subagents.runtime import SubAgentExecution


@dataclass(eq=False)
class MCPTrustResponseHandle:
    request: MCPTrustRequest
    _event: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _approved: bool = field(default=False, init=False, repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

    def resolve(self, approved: bool) -> None:
        with self._lock:
            if self._resolved:
                return
            self._approved = approved
            self._resolved = True
            self._event.set()

    def wait(self) -> bool:
        self._event.wait()
        return self._approved


class MCPTrustRequestMessage(Message):
    def __init__(self, handle: MCPTrustResponseHandle) -> None:
        self.handle = handle
        super().__init__()


class MCPTrustWarningMessage(Message):
    def __init__(self, warning: MCPTrustWarning) -> None:
        self.warning = warning
        super().__init__()


@dataclass(eq=False)
class PermissionResponseHandle:
    request: ConfirmationRequest
    _event: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _result: ConfirmationResult = field(
        default_factory=ConfirmationResult.rejected,
        init=False,
        repr=False,
    )
    _resolved: bool = field(default=False, init=False, repr=False)

    def resolve(self, result: ConfirmationResult) -> None:
        with self._lock:
            if self._resolved:
                return
            self._result = result
            self._resolved = True
            self._event.set()

    def wait(self) -> ConfirmationResult:
        self._event.wait()
        return self._result


class PermissionRequestMessage(Message):
    def __init__(self, handle: PermissionResponseHandle) -> None:
        self.handle = handle
        super().__init__()


class TuiMCPTrustConfirmer:
    """Adapt synchronous MCP trust confirmation to Textual messages."""

    def __init__(self, post_message: Callable[[Message], bool]) -> None:
        self._post_message = post_message
        self._lock = Lock()
        self._pending: set[MCPTrustResponseHandle] = set()
        self._closed = False

    def confirm(self, request: MCPTrustRequest) -> bool:
        handle = MCPTrustResponseHandle(request)
        with self._lock:
            if self._closed:
                return False
            self._pending.add(handle)
        try:
            if not self._post_message(MCPTrustRequestMessage(handle)):
                handle.resolve(False)
            return handle.wait()
        finally:
            with self._lock:
                self._pending.discard(handle)

    def report_warning(self, warning: MCPTrustWarning) -> None:
        self._post_message(MCPTrustWarningMessage(warning))

    def reject_all(self) -> None:
        with self._lock:
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
        for handle in pending:
            handle.resolve(False)


class TuiConfirmer:
    """Bridge synchronous Permission confirmation to the Textual UI."""

    def __init__(self, post_message: Callable[[Message], bool]) -> None:
        self._post_message = post_message
        self._lock = Lock()
        self._pending: set[PermissionResponseHandle] = set()
        self._closed = False

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        handle = PermissionResponseHandle(request)
        with self._lock:
            if self._closed:
                return ConfirmationResult.rejected(
                    message="Permission confirmation unavailable.",
                )
            self._pending.add(handle)
        try:
            if not self._post_message(PermissionRequestMessage(handle)):
                handle.resolve(
                    ConfirmationResult.rejected(
                        message="Permission confirmation unavailable.",
                    )
                )
            return handle.wait()
        finally:
            with self._lock:
                self._pending.discard(handle)

    def reject_all(self) -> None:
        with self._lock:
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
        for handle in pending:
            handle.resolve(
                ConfirmationResult.rejected(
                    message="Permission confirmation unavailable.",
                )
            )


class SubAgentStateMessage(Message):
    """Bridge a SubAgent state transition into the UI thread."""

    def __init__(self, transition: SubAgentStateTransition) -> None:
        self.transition = transition
        super().__init__()


class SubAgentResultMessage(Message):
    """Bridge a final SubAgent execution result into the UI thread."""

    def __init__(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
    ) -> None:
        self.task = task
        self.execution = execution
        super().__init__()


class TuiSubAgentObserver:
    """Bridge SubAgent lifecycle events into the Textual message queue.

    Callbacks run on the SubAgent worker thread. They only post Textual
    Messages; the UI thread owns every widget update. Snapshot and tool-audit
    events are never rendered in the single-conversation timeline.
    """

    def __init__(self, post_message: Callable[[Message], bool]) -> None:
        self._post_message = post_message
        self._closed = False

    def on_state(
        self,
        task: SubAgentTask,
        transition: SubAgentStateTransition,
    ) -> None:
        del task
        self._safe_post(SubAgentStateMessage(transition))

    def on_snapshot(
        self,
        task: SubAgentTask,
        run_id: str,
        snapshot,
        occurred_at: datetime,
    ) -> None:
        del task, run_id, snapshot, occurred_at

    def on_tool_audit(
        self,
        task: SubAgentTask,
        run_id: str,
        audit,
        occurred_at: datetime,
    ) -> None:
        del task, run_id, audit, occurred_at

    def on_result(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
        occurred_at: datetime,
    ) -> None:
        del occurred_at
        self._safe_post(SubAgentResultMessage(task, execution))

    def close(self) -> None:
        self._closed = True

    def _safe_post(self, message: Message) -> None:
        if self._closed:
            return
        try:
            self._post_message(message)
        except Exception:  # noqa: BLE001 - app may be closing; never raise into SubAgent thread
            pass


__all__ = [
    "MCPTrustRequestMessage",
    "MCPTrustResponseHandle",
    "MCPTrustWarningMessage",
    "PermissionRequestMessage",
    "PermissionResponseHandle",
    "SubAgentResultMessage",
    "SubAgentStateMessage",
    "TuiConfirmer",
    "TuiMCPTrustConfirmer",
    "TuiSubAgentObserver",
]
