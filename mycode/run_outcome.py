"""Structured status returned by one top-level Agent run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal, cast

from mycode.agent import AgentStopReason


AgentRunStatus = Literal["completed", "task_incomplete", "runtime_failure"]

_COMPLETED_REASONS = frozenset({"final_answer", "control_tool"})
_TASK_INCOMPLETE_REASONS = frozenset(
    {"max_turns", "repeated_tool_call"}
)
_KNOWN_STOP_REASONS = frozenset(
    {
        "final_answer",
        "tool_calls",
        "max_turns",
        "tool_error",
        "repeated_tool_call",
        "context_overflow",
        "model_error",
        "control_tool",
    }
)
_EXIT_CODES: dict[AgentRunStatus, int] = {
    "completed": 0,
    "runtime_failure": 1,
    "task_incomplete": 2,
}


@dataclass(frozen=True)
class AgentRunOutcome:
    """Loop termination result; completed does not assert semantic correctness."""

    status: AgentRunStatus
    stop_reason: AgentStopReason | None

    def __post_init__(self) -> None:
        if self.status not in _EXIT_CODES:
            raise ValueError(f"Unknown Agent run status: {self.status!r}")
        if self.stop_reason is not None and self.stop_reason not in _KNOWN_STOP_REASONS:
            raise ValueError(f"Unknown Agent stop reason: {self.stop_reason!r}")
        expected = _status_for_stop_reason(self.stop_reason)
        if self.status != expected:
            raise ValueError(
                "Agent run status does not match stop reason: "
                f"status={self.status!r}, stop_reason={self.stop_reason!r}"
            )

    @classmethod
    def from_stop_reason(
        cls,
        stop_reason: AgentStopReason | None,
    ) -> AgentRunOutcome:
        return cls(status=_status_for_stop_reason(stop_reason), stop_reason=stop_reason)

    @classmethod
    def from_json(cls, payload: str) -> AgentRunOutcome:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid Agent run outcome JSON") from error
        if not isinstance(raw, dict) or set(raw) != {"status", "stop_reason"}:
            raise ValueError("Agent run outcome must contain only status and stop_reason")

        status = raw["status"]
        stop_reason = raw["stop_reason"]
        if not isinstance(status, str) or status not in _EXIT_CODES:
            raise ValueError("Invalid Agent run status")
        if stop_reason is not None and (
            not isinstance(stop_reason, str) or stop_reason not in _KNOWN_STOP_REASONS
        ):
            raise ValueError("Invalid Agent stop reason")
        return cls(
            status=cast(AgentRunStatus, status),
            stop_reason=cast(AgentStopReason | None, stop_reason),
        )

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES[self.status]

    def to_json(self) -> str:
        return json.dumps(
            {"status": self.status, "stop_reason": self.stop_reason},
            sort_keys=True,
        )


def _status_for_stop_reason(
    stop_reason: AgentStopReason | None,
) -> AgentRunStatus:
    if stop_reason in _COMPLETED_REASONS:
        return "completed"
    if stop_reason in _TASK_INCOMPLETE_REASONS:
        return "task_incomplete"
    return "runtime_failure"
