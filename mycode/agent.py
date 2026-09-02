from dataclasses import dataclass, field
from typing import Literal

from mycode.reasoning import ReasoningState, normalize_reasoning
from mycode.tools import ToolResult


AgentStopReason = Literal[
    "final_answer",
    "tool_calls",
    "max_turns",
    "tool_error",
    "repeated_tool_call",
    "context_overflow",
    "model_error",
    "control_tool",
]

AgentEventType = Literal[
    "context",
    "turn",
    "progress",
    "artifact_warning",
    "model_start",
    "reasoning_delta",
    "reasoning_state",
    "text_delta",
    "tool_call",
    "tool_result",
    "stop",
    "error",
]

AgentWarningType = Literal["artifact_externalization"]


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AgentWarning:
    type: AgentWarningType
    content: str


@dataclass(frozen=True)
class AgentProgressSnapshot:
    stagnation_turns: int = 0
    same_tool_repeat: int = 0
    same_result_repeat: int = 0
    resource_repeat: int = 0
    convergence_guided: bool = False
    reason: str = "run_started"


@dataclass(frozen=True)
class AgentModelResponse:
    content: str = ""
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    stop_reason: AgentStopReason = "final_answer"
    warnings: tuple[AgentWarning, ...] = ()
    reasoning_content: str | None = field(default=None, repr=False)
    reasoning_state: ReasoningState = field(default="absent", repr=False)

    def __post_init__(self) -> None:
        reasoning_state, reasoning_content = normalize_reasoning(
            self.reasoning_state,
            self.reasoning_content,
        )
        object.__setattr__(self, "reasoning_state", reasoning_state)
        object.__setattr__(self, "reasoning_content", reasoning_content)
        if reasoning_state != "absent" and (
            self.stop_reason != "tool_calls" or not self.tool_calls
        ):
            raise ValueError(
                "reasoning_content is retained only for assistant tool-call responses."
            )


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    content: str = ""
    turn_number: int | None = None
    max_turns: int | None = None
    progress: AgentProgressSnapshot | None = None
    tool_call: AgentToolCall | None = None
    tool_result: ToolResult | None = None
    stop_reason: AgentStopReason | None = None
    error: str | None = None
    reasoning_content: str = field(default="", repr=False)
    reasoning_state: ReasoningState = field(default="absent", repr=False)
