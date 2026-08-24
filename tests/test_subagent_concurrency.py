from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import threading
import time

from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    PermissionDecision,
    PermissionRequest,
)
from mycode.subagents.concurrency import SubAgentInteractionGate
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.lifecycle import (
    RunTracker,
    SubAgentStateTransition,
    TrackingConfirmer,
)
from mycode.subagents.observability import SynchronizedSubAgentObserver


def test_interaction_gate_is_reentrant_for_tracking_observer() -> None:
    gate = SubAgentInteractionGate()
    task = SubAgentTask(role="tester", objective="Run validation.")
    observer = RecordingObserver()
    synchronized = SynchronizedSubAgentObserver(observer, gate)
    tracker = RunTracker(
        run_id="run-reentrant",
        role="tester",
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        handler=lambda transition: synchronized.on_state(task, transition),
    )
    confirmer = TrackingConfirmer(
        delegate=ApprovingConfirmer(),
        tracker=tracker,
        interaction_gate=gate,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "approved"
    assert [transition.state for transition in tracker.transitions] == [
        "awaiting_confirmation",
        "running",
    ]
    assert observer.states == ["awaiting_confirmation", "running"]


def test_confirmation_serializes_other_observer_callbacks() -> None:
    gate = SubAgentInteractionGate()
    blocking_confirmer = BlockingConfirmer()
    tracker = RunTracker(
        run_id="run-confirming",
        role="tester",
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    confirmer = TrackingConfirmer(
        delegate=blocking_confirmer,
        tracker=tracker,
        interaction_gate=gate,
    )
    task = SubAgentTask(role="explorer", objective="Inspect code.")
    observer = RecordingObserver()
    synchronized = SynchronizedSubAgentObserver(observer, gate)
    transition = SubAgentStateTransition(
        run_id="run-observed",
        role="explorer",
        state="running",
        occurred_at=datetime(2026, 7, 28, tzinfo=UTC),
        reason="run_started",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        confirmation = pool.submit(confirmer.confirm, _confirmation_request())
        assert blocking_confirmer.started.wait(timeout=1)
        observation = pool.submit(synchronized.on_state, task, transition)
        time.sleep(0.02)
        assert observer.states == []
        blocking_confirmer.release.set()
        assert confirmation.result(timeout=1).status == "approved"
        observation.result(timeout=1)

    assert observer.states == ["running"]


def _confirmation_request() -> ConfirmationRequest:
    permission_request = PermissionRequest(
        tool_name="run_validation",
        capability="command",
        action="run_validation",
    )
    decision = PermissionDecision.ask(
        message="Validation requires confirmation."
    )
    return ConfirmationRequest(
        permission_request=permission_request,
        permission_decision=decision,
        prompt=decision.message,
    )


class ApprovingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.approved()


class BlockingConfirmer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.started.set()
        if not self.release.wait(timeout=1):
            raise TimeoutError("confirmation release was not signaled")
        return ConfirmationResult.approved()


class RecordingObserver:
    def __init__(self) -> None:
        self.states: list[str] = []

    def on_state(self, task, transition) -> None:
        self.states.append(transition.state)

    def on_snapshot(self, task, run_id, snapshot, occurred_at) -> None:
        pass

    def on_tool_audit(self, task, run_id, audit, occurred_at) -> None:
        pass

    def on_result(self, task, execution, occurred_at) -> None:
        pass
