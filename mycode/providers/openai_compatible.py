from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from mycode.config import LLMConfig, ReasoningEffort
from mycode.conversation import Conversation
from mycode.model_errors import extract_provider_diagnostic
from mycode.model_events import ModelStreamEvent, ModelToolCall, TokenUsage
from mycode.messages import Message
from mycode.providers.openai_errors import normalize_openai_error
from mycode.reasoning import ReasoningState


SDK_MAX_RETRIES = 2
SDK_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=120.0,
    write=30.0,
    pool=10.0,
)
_OPENCODE_GO_HOST = "opencode.ai"
_OPENCODE_GO_PATHS = frozenset({"/zen/go", "/zen/go/v1"})


def _raise_provider_error(error: Exception) -> NoReturn:
    normalized = normalize_openai_error(error)
    if normalized is error:
        raise error
    raise normalized from error


def _is_opencode_go_base_url(base_url: str) -> bool:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == _OPENCODE_GO_HOST
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
        and parsed.path.rstrip("/") in _OPENCODE_GO_PATHS
        and parsed.query == ""
        and parsed.fragment == ""
    )


@dataclass
class OpenAICompatibleLLMClient:
    config: LLMConfig
    model: str | None = None
    thinking_enabled: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    session_id: str | None = field(default=None, repr=False)
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
                max_retries=SDK_MAX_RETRIES,
                timeout=SDK_TIMEOUT,
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
            _raise_provider_error(error)
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
            _raise_provider_error(error)
        self.last_model_response = observation.finish(usage=self.last_token_usage)

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ) -> Iterator[ModelStreamEvent]:
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
                    yield ModelStreamEvent(
                        type="reasoning_delta",
                        reasoning_content=reasoning.content,
                    )

                content = getattr(delta, "content", None) or ""
                observation.observe_content(content)
                if content != "":
                    yield ModelStreamEvent(type="text_delta", content=content)

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
            _raise_provider_error(error)

        parsed_tool_calls = _parse_tool_call_buffers(tool_call_buffers)
        if isinstance(parsed_tool_calls, str):
            observation.error_type = "ToolCallParseError"
            self.last_model_response = observation.finish(usage=self.last_token_usage)
            yield ModelStreamEvent(type="error", error=parsed_tool_calls)
            return

        observation.tool_names = tuple(call.name for call in parsed_tool_calls)

        if (
            parsed_tool_calls
            and self.thinking_enabled is True
            and reasoning_state == "absent"
        ):
            observation.error_type = "ReasoningProtocolError"
            self.last_model_response = observation.finish(usage=self.last_token_usage)
            yield ModelStreamEvent(
                type="error",
                error=(
                    "Thinking tool-call response omitted the "
                    "reasoning_content field."
                ),
            )
            return

        if parsed_tool_calls and reasoning_state != "absent":
            yield ModelStreamEvent(
                type="reasoning_state",
                reasoning_state=_message_reasoning_state(reasoning_state),
            )

        for tool_call in parsed_tool_calls:
            yield ModelStreamEvent(type="tool_call", tool_call=tool_call)
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
            "messages": _format_openai_messages(conversation.get_messages()),
            "stream": stream,
        }
        if _is_opencode_go_base_url(self.config.base_url) and self.session_id:
            request["extra_headers"] = {
                "User-Agent": "mycode-agent",
                "x-opencode-session": self.session_id,
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
        provider_diagnostic = (
            None if error is None else extract_provider_diagnostic(error)
        )
        if error is not None:
            self.error_type = type(error).__name__
        return {
            "model": self.model,
            "request_id": self.request_id,
            "provider_request_id": (
                self.provider_request_id
                if provider_diagnostic is None
                else self.provider_request_id or provider_diagnostic.request_id
            ),
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
            "provider_error_code": (
                None if provider_diagnostic is None else provider_diagnostic.code
            ),
            "provider_error_type": (
                None if provider_diagnostic is None else provider_diagnostic.error_type
            ),
            "provider_error_message": (
                None if provider_diagnostic is None else provider_diagnostic.message
            ),
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


def _format_openai_messages(messages: list[Message]) -> list[dict[str, object]]:
    return [_format_openai_message(message) for message in messages]


def _format_openai_message(message: Message) -> dict[str, object]:
    formatted: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        formatted["tool_calls"] = [
            _format_openai_tool_call(tool_call) for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        formatted["tool_call_id"] = message.tool_call_id
    if message.reasoning_state != "absent":
        formatted["reasoning_content"] = message.reasoning_content
    return formatted


def _format_openai_tool_call(tool_call: ModelToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
        },
    }


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


def _parse_tool_calls(tool_calls: list[Any]) -> list[ModelToolCall] | str:
    parsed_tool_calls: list[ModelToolCall] = []

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
) -> list[ModelToolCall] | str:
    parsed_tool_calls: list[ModelToolCall] = []

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


def _parse_tool_call(*, id: str, name: str, arguments: str) -> ModelToolCall | str:
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as error:
        return f"Invalid tool call arguments for {name}: {error.msg}"

    if not isinstance(parsed_arguments, dict):
        return f"Invalid tool call arguments for {name}: expected a JSON object"

    return ModelToolCall(
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
