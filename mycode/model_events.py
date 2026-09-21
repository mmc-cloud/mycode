"""Provider-neutral values exchanged by model clients and their callers."""

from dataclasses import dataclass, field
from typing import Literal

from mycode.reasoning import ReasoningState, normalize_reasoning


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, object]


ModelStreamEventType = Literal[
    "reasoning_delta", "reasoning_state", "text_delta", "tool_call", "error"
]


@dataclass(frozen=True)
class ModelStreamEvent:
    type: ModelStreamEventType
    content: str = ""
    tool_call: ModelToolCall | None = None
    error: str | None = None
    reasoning_content: str = field(default="", repr=False)
    reasoning_state: ReasoningState = field(default="absent", repr=False)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by a provider for one model call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    reasoning_content: str | None = field(default=None, repr=False)
    reasoning_state: ReasoningState = field(default="absent", repr=False)

    def __post_init__(self) -> None:
        state, content = normalize_reasoning(self.reasoning_state, self.reasoning_content)
        object.__setattr__(self, "reasoning_state", state)
        object.__setattr__(self, "reasoning_content", content)
        if state != "absent" and not self.tool_calls:
            raise ValueError(
                "reasoning_content is retained only for assistant tool-call responses."
            )
