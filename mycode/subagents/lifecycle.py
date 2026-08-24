from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from mycode.permissions import ConfirmationRequest, ConfirmationResult, Confirmer
from mycode.subagents.concurrency import SubAgentInteractionGate
from mycode.subagents.contracts import SubAgentRole


SubAgentRunState = Literal[
    "running",
    "awaiting_confirmation",
    "completed",
    "failed",
    "interrupted",
]


@dataclass(frozen=True)
class SubAgentStateTransition:
    run_id: str
    role: SubAgentRole
    state: SubAgentRunState
    occurred_at: datetime
    reason: str


StateTransitionHandler = Callable[[SubAgentStateTransition], None]


@dataclass
class RunTracker:
    run_id: str
    role: SubAgentRole
    clock: Callable[[], datetime]
    handler: StateTransitionHandler | None = None
    transitions: list[SubAgentStateTransition] = field(default_factory=list)

    def transition(self, state: SubAgentRunState, reason: str) -> None:
        transition = SubAgentStateTransition(
            run_id=self.run_id,
            role=self.role,
            state=state,
            occurred_at=self.clock(),
            reason=reason,
        )
        self.transitions.append(transition)
        if self.handler is not None:
            handler = self.handler
            try:
                handler(transition)
            except BaseException:
                self.handler = None
                raise

    def transition_once(self, state: SubAgentRunState, reason: str) -> None:
        if self.transitions and self.transitions[-1].state == state:
            return
        self.transition(state, reason)


@dataclass(frozen=True)
class TrackingConfirmer:
    delegate: Confirmer
    tracker: RunTracker
    interaction_gate: SubAgentInteractionGate

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return self.interaction_gate.run(lambda: self._confirm(request))

    def _confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.tracker.transition("awaiting_confirmation", "permission_confirmation")
        try:
            result = self.delegate.confirm(request)
        except KeyboardInterrupt:
            self.tracker.transition_once("interrupted", "confirmation_interrupted")
            raise
        except Exception:
            self.tracker.transition("running", "confirmation_error")
            raise
        self.tracker.transition(
            "running",
            f"confirmation_{result.status}",
        )
        return result
