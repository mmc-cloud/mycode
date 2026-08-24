from dataclasses import dataclass, field
import json
from typing import Literal

from mycode.agent import AgentToolCall
from mycode.reasoning import ReasoningState, normalize_reasoning


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    tool_calls: tuple[AgentToolCall, ...] = ()
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

    def to_model_dict(self) -> dict[str, object]:
        model_dict: dict[str, object] = {
            "role": self.role,
            "content": self.content,
        }

        if self.tool_calls:
            model_dict["tool_calls"] = [
                _tool_call_to_model_dict(tool_call) for tool_call in self.tool_calls
            ]

        if self.tool_call_id is not None:
            model_dict["tool_call_id"] = self.tool_call_id

        if self.reasoning_state != "absent":
            model_dict["reasoning_content"] = self.reasoning_content

        return model_dict


def _tool_call_to_model_dict(tool_call: AgentToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
        },
    }
