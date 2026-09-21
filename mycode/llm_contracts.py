from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.model_events import ModelResponse, ModelStreamEvent, TokenUsage


class LLMClient(Protocol):
    """Provider-neutral contract consumed by the agent runtime."""

    last_token_usage: TokenUsage | None
    last_reasoning_char_count: int
    last_model_response: dict[str, object] | None

    def complete(self, conversation: Conversation) -> Message:
        pass

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        pass

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[ModelStreamEvent]:
        pass


@dataclass
class FakeLLMClient:
    """Deterministic provider-neutral client used by tests and local callers."""

    responses: list[str]
    stream_chunk_size: int | None = None
    tool_responses: list[ModelResponse] = field(default_factory=list)
    token_usages: list[TokenUsage | None] = field(default_factory=list)
    last_token_usage: TokenUsage | None = field(default=None, init=False)
    last_reasoning_char_count: int = field(default=0, init=False)
    last_model_response: dict[str, object] | None = field(default=None, init=False)

    def complete(self, conversation: Conversation) -> Message:
        self._consume_token_usage()
        self.last_reasoning_char_count = 0
        self.last_model_response = None
        if not self.responses:
            raise RuntimeError("FakeLLMClient has no responses left")

        return Message(role="assistant", content=self.responses.pop(0))

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        content = self.complete(conversation).content

        if self.stream_chunk_size is None:
            yield content
            return

        for start in range(0, len(content), self.stream_chunk_size):
            yield content[start : start + self.stream_chunk_size]

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[ModelStreamEvent]:
        self._consume_token_usage()
        if not self.tool_responses:
            raise RuntimeError("FakeLLMClient has no tool responses left")

        response = self.tool_responses.pop(0)
        self.last_reasoning_char_count = len(response.reasoning_content or "")
        usage = self.last_token_usage
        self.last_model_response = {
            "model": None,
            "request_id": None,
            "provider_request_id": None,
            "finish_reason": None,
            "stop_reason": "tool_calls" if response.tool_calls else "final_answer",
            "content_chars": len(response.content),
            "content_non_whitespace_chars": sum(
                not character.isspace() for character in response.content
            ),
            "tool_call_count": len(response.tool_calls),
            "tool_names": [call.name for call in response.tool_calls],
            "reasoning_field_present": response.reasoning_state != "absent",
            "reasoning_chars": len(response.reasoning_content or ""),
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": None if usage is None else usage.completion_tokens,
            "total_tokens": None if usage is None else usage.total_tokens,
            "latency_ms": None,
            "first_token_latency_ms": None,
            "stream_chunk_count": None,
            "retry_count": None,
            "error_type": None,
            "http_status": None,
            "empty_response": (
                not response.content.strip() and not response.tool_calls
            ),
        }

        if response.reasoning_content is not None:
            chunks = (
                [response.reasoning_content]
                if self.stream_chunk_size is None
                else [
                    response.reasoning_content[start : start + self.stream_chunk_size]
                    for start in range(
                        0,
                        len(response.reasoning_content),
                        self.stream_chunk_size,
                    )
                ]
            )
            for chunk in chunks:
                yield ModelStreamEvent(
                    type="reasoning_delta",
                    reasoning_content=chunk,
                )

        if response.tool_calls and response.reasoning_state != "absent":
            yield ModelStreamEvent(
                type="reasoning_state",
                reasoning_state=response.reasoning_state,
            )

        if response.content != "":
            if self.stream_chunk_size is None:
                yield ModelStreamEvent(type="text_delta", content=response.content)
            else:
                for start in range(0, len(response.content), self.stream_chunk_size):
                    yield ModelStreamEvent(
                        type="text_delta",
                        content=response.content[start : start + self.stream_chunk_size],
                    )

        for tool_call in response.tool_calls:
            yield ModelStreamEvent(type="tool_call", tool_call=tool_call)

    def _consume_token_usage(self) -> None:
        self.last_token_usage = self.token_usages.pop(0) if self.token_usages else None
