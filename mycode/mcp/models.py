from dataclasses import dataclass
from typing import Literal

MCPServerStatusValue = Literal["connected", "failed"]
MCPShutdownStatusValue = Literal["not_started", "completed", "timeout"]


@dataclass(frozen=True)
class MCPServerStatus:
    alias: str
    status: MCPServerStatusValue
    tool_count: int = 0
    error_type: str | None = None
    error_summary: str | None = None


@dataclass(frozen=True)
class MCPShutdownStatus:
    status: MCPShutdownStatusValue = "not_started"
    error: str | None = None
