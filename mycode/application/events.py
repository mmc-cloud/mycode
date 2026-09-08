"""Application-facing runtime events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mycode.agent.events import AgentEvent
from mycode.agent.outcome import AgentRunOutcome
from mycode.mcp.models import MCPServerStatus


RuntimeEventType = Literal[
    "runtime_ready",
    "mcp_status",
    "agent",
    "turn_finished",
]


@dataclass(frozen=True)
class RuntimeEvent:
    """Small application boundary wrapper around existing domain events."""

    type: RuntimeEventType
    turn_id: str | None = None
    agent_event: AgentEvent | None = None
    outcome: AgentRunOutcome | None = None
    mcp_status: MCPServerStatus | None = None
    session_id: str | None = None
    session_title: str | None = None
    session_created: bool | None = None
    compact_state_recovered: bool | None = None
    instruction_sources: tuple[str, ...] = ()
    instruction_warnings: tuple[str, ...] = ()
    skill_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in {
            "runtime_ready",
            "mcp_status",
            "agent",
            "turn_finished",
        }:
            raise ValueError(f"Unsupported runtime event type: {self.type}")
        object.__setattr__(self, "instruction_sources", tuple(self.instruction_sources))
        object.__setattr__(self, "instruction_warnings", tuple(self.instruction_warnings))
        object.__setattr__(self, "skill_warnings", tuple(self.skill_warnings))

        if self.type == "agent" and not isinstance(self.agent_event, AgentEvent):
            raise ValueError("agent runtime events require agent_event")
        if self.type == "mcp_status" and not isinstance(
            self.mcp_status, MCPServerStatus
        ):
            raise ValueError("mcp_status runtime events require mcp_status")
        if self.type == "turn_finished" and not isinstance(
            self.outcome, AgentRunOutcome
        ):
            raise ValueError("turn_finished runtime events require outcome")
