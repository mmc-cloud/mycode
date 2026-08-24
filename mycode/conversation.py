from collections.abc import Callable
from dataclasses import dataclass, field

from mycode.agent import AgentToolCall
from mycode.messages import Message
from mycode.reasoning import ReasoningState


@dataclass
class Conversation:
    _messages: list[Message] = field(default_factory=list)
    _on_message_added: Callable[[Message], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_messages(
        cls,
        messages: list[Message],
        *,
        on_message_added: Callable[[Message], None] | None = None,
    ) -> "Conversation":
        return cls(
            _messages=list(messages),
            _on_message_added=on_message_added,
        )

    def add_message(self, message: Message) -> None:
        if self._on_message_added is not None:
            self._on_message_added(message)
        self._messages.append(message)

    def add_system_message(self, content: str) -> None:
        self.add_message(Message(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        self.add_message(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.add_message(Message(role="assistant", content=content))

    def add_assistant_tool_calls(
        self,
        content: str,
        tool_calls: list[AgentToolCall],
        *,
        reasoning_content: str | None = None,
        reasoning_state: ReasoningState = "absent",
    ) -> None:
        self.add_message(
            Message(
                role="assistant",
                content=content,
                tool_calls=tuple(tool_calls),
                reasoning_content=reasoning_content,
                reasoning_state=reasoning_state,
            )
        )

    def add_tool_result_message(self, tool_call_id: str, content: str) -> None:
        self.add_message(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )
        )

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def to_model_messages(self) -> list[dict[str, object]]:
        return [message.to_model_dict() for message in self._messages]

    def clear(self) -> None:
        self._messages.clear()
