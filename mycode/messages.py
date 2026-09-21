from dataclasses import dataclass, field
from typing import Literal

from mycode.model_events import ModelToolCall
from mycode.reasoning import ReasoningState, normalize_reasoning


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None
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
            self.role != "assistant" or not self.tool_calls
        ):
            raise ValueError(
                "reasoning_content is retained only on assistant tool-call messages."
            )
