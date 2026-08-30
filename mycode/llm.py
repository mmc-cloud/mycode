from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any, Literal, Protocol

from openai import OpenAI

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.config import LLMConfig, ReasoningEffort
from mycode.context_budget import TokenUsage
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.reasoning import ReasoningState


class LLMClient(Protocol):
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
    ) -> Iterator[AgentEvent]:
        pass


@dataclass
class OpenAICompatibleLLMClient:
    config: LLMConfig
    model: str | None = None
    thinking_enabled: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    _client: Any = field(default=None, repr=False)
    last_token_usage: TokenUsage | None = field(default=None, init=False)
    last_reasoning_char_count: int = field(default=0, init=False)
    last_model_response: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = self.config.model if self.model is None else self.model.strip()
        if self.model == "":
            raise ValueError("LLM client model must not be empty.")
        self.thinking_enabled = (
            self.config.thinking_enabled
            if self.thinking_enabled is None
            else self.thinking_enabled
        )
        if self.thinking_enabled is True:
            self.reasoning_effort = (
                self.config.reasoning_effort
                if self.reasoning_effort is None
                else self.reasoning_effort
            )
            if self.reasoning_effort is None:
                self.reasoning_effort = "high"
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort requires thinking_enabled=True.")
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

    def complete(self, conversation: Conversation) -> Message:
        self.last_token_usage = None
        self.last_reasoning_char_count = 0
        self.last_model_response = None
        observation = _ModelResponseAccumulator(model=self.model, stream=False)
        try:
            response = self._client.chat.completions.create(
                **self._request(conversation, stream=False)
            )
        except Exception as error:
            self.last_model_response = observation.finish(error=error)
            raise
        observation.observe_response(response)
        self.last_token_usage = _extract_token_usage(response)

        choice = _first_choice(response)
        if choice is None:
            self.last_model_response = observation.finish(usage=self.last_token_usage)
            return Message(role="assistant", content="")

        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) or ""
        reasoning = _extract_reasoning_field(message)
        self.last_reasoning_char_count = len(reasoning.content or "")
        observation.observe_content(content)
        observation.observe_reasoning(reasoning)
        self.last_model_response = observation.finish(usage=self.last_token_usage)

        return Message(role="assistant", content=content)

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        self.last_token_usage = None
        self.last_reasoning_char_count = 0
        self.last_model_response = None
        observation = _ModelResponseAccumulator(model=self.model, stream=True)
        try:
            response = self._client.chat.completions.create(
                **self._request(conversation, stream=True)
            )
            observation.observe_response(response)

            for chunk in response:
                observation.observe_chunk(chunk)
                usage = _extract_token_usage(chunk)
                if usage is not None:
                    self.last_token_usage = usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                reasoning = _extract_reasoning_field(delta)
                self.last_reasoning_char_count += len(reasoning.content or "")
                observation.observe_reasoning(reasoning)
                content = getattr(delta, "content", None) or ""
                observation.observe_content(content)

                if content != "":
                    yield content
        except Exception as error:
            self.last_model_response = observation.finish(
                usage=self.last_token_usage,
                error=error,
            )
            raise
        self.last_model_response = observation.finish(usage=self.last_token_usage)

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[AgentEvent]:
        self.last_token_usage = None
        self.last_reasoning_char_count = 0
        self.last_model_response = None
        observation = _ModelResponseAccumulator(model=self.model, stream=True)
        tool_call_buffers: dict[int, _ToolCallBuffer] = {}
        reasoning_state: _RawReasoningState = "absent"
        try:
            response = self._client.chat.completions.create(
                **self._request(conversation, tools=tools, stream=True)
            )
            observation.observe_response(response)

            for chunk in response:
                observation.observe_chunk(chunk)
                usage = _extract_token_usage(chunk)
                if usage is not None:
                    self.last_token_usage = usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                reasoning = _extract_reasoning_field(delta)
                reasoning_state = _merge_reasoning_state(
                    reasoning_state,
                    reasoning.state,
                )
                observation.observe_reasoning(reasoning)
                if reasoning.content is not None:
                    self.last_reasoning_char_count += len(reasoning.content)
                    yield AgentEvent(
                        type="reasoning_delta",
                        reasoning_content=reasoning.content,
                    )

                content = getattr(delta, "content", None) or ""
                observation.observe_content(content)
                if content != "":
                    yield AgentEvent(type="text_delta", content=content)

                tool_call_deltas = getattr(delta, "tool_calls", None) or []
                if tool_call_deltas:
                    observation.observe_meaningful_delta()
                for tool_call_delta in tool_call_deltas:
                    _accumulate_tool_call_delta(tool_call_buffers, tool_call_delta)
        except Exception as error:
            self.last_model_response = observation.finish(
                usage=self.last_token_usage,
                error=error,
            )
            raise

        parsed_tool_calls = _parse_tool_call_buffers(tool_call_buffers)
        if isinstance(parsed_tool_calls, str):
            observation.error_type = "ToolCallParseError"
            self.last_model_response = observation.finish(usage=self.last_token_usage)
            yield AgentEvent(type="error", error=parsed_tool_calls)
            return

        observation.tool_names = tuple(call.name for call in parsed_tool_calls)

        if (
            parsed_tool_calls
            and self.thinking_enabled is True
            and reasoning_state == "absent"
        ):
            observation.error_type = "ReasoningProtocolError"
            self.last_model_response = observation.finish(usage=self.last_token_usage)
            yield AgentEvent(
                type="error",
                error=(
                    "Thinking tool-call response omitted the "
                    "reasoning_content field."
                ),
            )
            return

        if parsed_tool_calls and reasoning_state != "absent":
            yield AgentEvent(
                type="reasoning_state",
                reasoning_state=_message_reasoning_state(reasoning_state),
            )

        for tool_call in parsed_tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tool_call)
        self.last_model_response = observation.finish(usage=self.last_token_usage)

    def _request(
        self,
        conversation: Conversation,
        *,
        stream: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self.model,
            "messages": conversation.to_model_messages(),
            "stream": stream,
        }
        if tools:
            request["tools"] = _format_openai_tools(tools)
        if stream and self.config.stream_include_usage:
            request["stream_options"] = {"include_usage": True}
        if self.config.max_output_tokens is not None:
            request["max_tokens"] = self.config.max_output_tokens
        if self.thinking_enabled is not None:
            request["extra_body"] = {
                "thinking": {
                    "type": "enabled" if self.thinking_enabled else "disabled"
                }
            }
        if self.thinking_enabled is True:
            request["reasoning_effort"] = self.reasoning_effort
        return request


@dataclass
class FakeLLMClient:
    responses: list[str]
    stream_chunk_size: int | None = None
    tool_responses: list[AgentModelResponse] = field(default_factory=list)
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
    ) -> Iterator[AgentEvent]:
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
            "stop_reason": response.stop_reason,
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
                yield AgentEvent(
                    type="reasoning_delta",
                    reasoning_content=chunk,
                )

        if response.tool_calls and response.reasoning_state != "absent":
            yield AgentEvent(
                type="reasoning_state",
                reasoning_state=response.reasoning_state,
            )

        if response.content != "":
            if self.stream_chunk_size is None:
                yield AgentEvent(type="text_delta", content=response.content)
            else:
                for start in range(0, len(response.content), self.stream_chunk_size):
                    yield AgentEvent(
                        type="text_delta",
                        content=response.content[start : start + self.stream_chunk_size],
                    )

        for tool_call in response.tool_calls:
            yield AgentEvent(type="tool_call", tool_call=tool_call)

    def _consume_token_usage(self) -> None:
        self.last_token_usage = self.token_usages.pop(0) if self.token_usages else None


@dataclass
class _ModelResponseAccumulator:
    model: str | None
    stream: bool
    started_at: float = field(default_factory=perf_counter)
    request_id: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    content_chars: int = 0
    content_non_whitespace_chars: int = 0
    reasoning_field_present: bool = False
    reasoning_chars: int = 0
    stream_chunk_count: int = 0
    first_token_at: float | None = None
    tool_names: tuple[str, ...] = ()
    error_type: str | None = None

    def observe_response(self, response: Any) -> None:
        self.request_id = self.request_id or _optional_string(
            _value(response, "id")
        )
        self.provider_request_id = self.provider_request_id or _optional_string(
            _value(response, "_request_id")
        )
        choices = _value(response, "choices") or []
        if choices:
            self._observe_choice(choices[0])

    def observe_chunk(self, chunk: Any) -> None:
        self.stream_chunk_count += 1
        self.observe_response(chunk)

    def observe_content(self, content: str) -> None:
        self.content_chars += len(content)
        self.content_non_whitespace_chars += sum(
            not character.isspace() for character in content
        )
        if content:
            self.observe_meaningful_delta()

    def observe_reasoning(self, reasoning: "_ReasoningField") -> None:
        if reasoning.state != "absent":
            self.reasoning_field_present = True
        if reasoning.content is not None:
            self.reasoning_chars += len(reasoning.content)
            if reasoning.content:
                self.observe_meaningful_delta()

    def observe_meaningful_delta(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = perf_counter()

    def finish(
        self,
        *,
        usage: TokenUsage | None = None,
        error: Exception | None = None,
    ) -> dict[str, object]:
        finished_at = perf_counter()
        if error is not None:
            self.error_type = type(error).__name__
        return {
            "model": self.model,
            "request_id": self.request_id,
            "provider_request_id": self.provider_request_id,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "content_chars": self.content_chars,
            "content_non_whitespace_chars": self.content_non_whitespace_chars,
            "tool_call_count": len(self.tool_names),
            "tool_names": list(self.tool_names),
            "reasoning_field_present": self.reasoning_field_present,
            "reasoning_chars": self.reasoning_chars,
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": None if usage is None else usage.completion_tokens,
            "total_tokens": None if usage is None else usage.total_tokens,
            "latency_ms": round((finished_at - self.started_at) * 1000),
            "first_token_latency_ms": (
                None
                if self.first_token_at is None
                else round((self.first_token_at - self.started_at) * 1000)
            ),
            "stream_chunk_count": self.stream_chunk_count if self.stream else None,
            "retry_count": _retry_count(error),
            "error_type": self.error_type,
            "http_status": _http_status(error),
            "empty_response": (
                self.content_non_whitespace_chars == 0 and not self.tool_names
            ),
        }

    def _observe_choice(self, choice: Any) -> None:
        finish_reason = _optional_string(_value(choice, "finish_reason"))
        stop_reason = _optional_string(_value(choice, "stop_reason"))
        if finish_reason is not None:
            self.finish_reason = finish_reason
        if stop_reason is not None:
            self.stop_reason = stop_reason


@dataclass
class _ToolCallBuffer:
    id: str = ""
    name: str = ""
    arguments: str = ""


_RawReasoningState = Literal["absent", "null", "empty", "nonempty"]


@dataclass(frozen=True)
class _ReasoningField:
    state: _RawReasoningState
    content: str | None = None

    @property
    def message_state(self) -> ReasoningState:
        return _message_reasoning_state(self.state)


def _format_openai_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"type": "function", "function": dict(tool)} for tool in tools]


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    return model_extra.get(name) if isinstance(model_extra, Mapping) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _http_status(error: Exception | None) -> int | None:
    if error is None:
        return None
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_count(error: Exception | None) -> int | None:
    if error is None:
        return None
    request = getattr(error, "request", None)
    headers = getattr(request, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("x-stainless-retry-count")
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _first_choice(response: Any) -> Any | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None

    return choices[0]


def _extract_reasoning_field(value: Any) -> _ReasoningField:
    if value is None:
        return _ReasoningField(state="absent")

    found = False
    reasoning_content: object = None
    if isinstance(value, Mapping):
        if "reasoning_content" in value:
            found = True
            reasoning_content = value["reasoning_content"]
    else:
        model_extra = getattr(value, "model_extra", None)
        if isinstance(model_extra, Mapping) and "reasoning_content" in model_extra:
            found = True
            reasoning_content = model_extra["reasoning_content"]
        else:
            model_fields_set = getattr(value, "model_fields_set", None)
            if (
                isinstance(model_fields_set, (set, frozenset))
                and "reasoning_content" in model_fields_set
            ):
                found = True
                reasoning_content = getattr(value, "reasoning_content", None)
            else:
                instance_values = getattr(value, "__dict__", None)
                if (
                    isinstance(instance_values, Mapping)
                    and "reasoning_content" in instance_values
                ):
                    found = True
                    reasoning_content = instance_values["reasoning_content"]

    if not found:
        return _ReasoningField(state="absent")
    if reasoning_content is None:
        return _ReasoningField(state="null")
    if reasoning_content == "":
        return _ReasoningField(state="empty")
    if not isinstance(reasoning_content, str):
        raise TypeError("Model reasoning_content must be a string when provided.")
    return _ReasoningField(state="nonempty", content=reasoning_content)


def _merge_reasoning_state(
    current: _RawReasoningState,
    incoming: _RawReasoningState,
) -> _RawReasoningState:
    priority = {"absent": 0, "null": 1, "empty": 2, "nonempty": 3}
    return incoming if priority[incoming] > priority[current] else current


def _message_reasoning_state(state: _RawReasoningState) -> ReasoningState:
    if state == "absent":
        return "absent"
    if state == "nonempty":
        return "present_nonempty"
    return "present_empty"


def _extract_token_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return None

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if prompt_tokens is None:
        return None

    completion_tokens = 0 if completion_tokens is None else completion_tokens
    total_tokens = (
        prompt_tokens + completion_tokens if total_tokens is None else total_tokens
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _usage_value(usage: Any, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
    return value if isinstance(value, int) and value >= 0 else None


def _parse_tool_calls(tool_calls: list[Any]) -> list[AgentToolCall] | str:
    parsed_tool_calls: list[AgentToolCall] = []

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        parsed_tool_call = _parse_tool_call(
            id=getattr(tool_call, "id", ""),
            name=getattr(function, "name", ""),
            arguments=getattr(function, "arguments", None) or "{}",
        )
        if isinstance(parsed_tool_call, str):
            return parsed_tool_call

        parsed_tool_calls.append(parsed_tool_call)

    return parsed_tool_calls


def _parse_tool_call_buffers(
    tool_call_buffers: dict[int, _ToolCallBuffer],
) -> list[AgentToolCall] | str:
    parsed_tool_calls: list[AgentToolCall] = []

    for index in sorted(tool_call_buffers):
        buffer = tool_call_buffers[index]
        parsed_tool_call = _parse_tool_call(
            id=buffer.id,
            name=buffer.name,
            arguments=buffer.arguments or "{}",
        )
        if isinstance(parsed_tool_call, str):
            return parsed_tool_call

        parsed_tool_calls.append(parsed_tool_call)

    return parsed_tool_calls


def _parse_tool_call(*, id: str, name: str, arguments: str) -> AgentToolCall | str:
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as error:
        return f"Invalid tool call arguments for {name}: {error.msg}"

    if not isinstance(parsed_arguments, dict):
        return f"Invalid tool call arguments for {name}: expected a JSON object"

    return AgentToolCall(
        id=id,
        name=name,
        arguments=parsed_arguments,
    )


def _accumulate_tool_call_delta(
    tool_call_buffers: dict[int, _ToolCallBuffer],
    tool_call_delta: Any,
) -> None:
    index = getattr(tool_call_delta, "index", len(tool_call_buffers))
    buffer = tool_call_buffers.setdefault(index, _ToolCallBuffer())

    tool_call_id = getattr(tool_call_delta, "id", None)
    if tool_call_id:
        buffer.id = tool_call_id

    function = getattr(tool_call_delta, "function", None)
    if function is None:
        return

    function_name = getattr(function, "name", None)
    if function_name:
        buffer.name += function_name

    function_arguments = getattr(function, "arguments", None)
    if function_arguments:
        buffer.arguments += function_arguments
