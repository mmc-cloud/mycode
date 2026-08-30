from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from mycode.agent import AgentEvent, AgentModelResponse, AgentToolCall
from mycode.config import LLMConfig
from mycode.context_budget import TokenUsage
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient, OpenAICompatibleLLMClient
from mycode.messages import Message


def test_fake_llm_client_returns_configured_responses() -> None:
    client = FakeLLMClient(responses=["first", "second"])
    conversation = Conversation()

    assert client.complete(conversation) == Message(role="assistant", content="first")
    assert client.complete(conversation) == Message(role="assistant", content="second")


def test_fake_llm_client_streams_configured_response() -> None:
    client = FakeLLMClient(responses=["streaming"], stream_chunk_size=3)
    conversation = Conversation()

    assert list(client.stream_complete(conversation)) == ["str", "eam", "ing"]


def test_fake_llm_client_raises_when_no_responses_left() -> None:
    client = FakeLLMClient(responses=[])

    with pytest.raises(RuntimeError, match="no responses left"):
        client.complete(Conversation())


def test_fake_llm_client_streams_tool_response_events() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="grep",
        arguments={"query": "main"},
    )
    client = FakeLLMClient(
        responses=[],
        stream_chunk_size=3,
        tool_responses=[
            AgentModelResponse(
                content="hello",
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            )
        ],
    )

    assert list(client.stream_with_tools(Conversation(), [])) == [
        AgentEvent(type="text_delta", content="hel"),
        AgentEvent(type="text_delta", content="lo"),
        AgentEvent(type="tool_call", tool_call=tool_call),
    ]


def test_openai_compatible_client_sends_model_and_messages() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="assistant reply")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)
    conversation = Conversation()
    conversation.add_system_message("system rules")
    conversation.add_user_message("hello")

    response = client.complete(conversation)

    assert response == Message(role="assistant", content="assistant reply")
    assert fake_sdk_client.chat.completions.last_request == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
    }


def test_openai_compatible_client_can_override_configured_model() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="assistant reply")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="main-model",
    )
    client = OpenAICompatibleLLMClient(
        config=config,
        model="role-model",
        _client=fake_sdk_client,
    )

    client.complete(Conversation())

    assert fake_sdk_client.chat.completions.last_request["model"] == "role-model"


def test_openai_compatible_client_sends_deepseek_thinking_request_fields() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="assistant reply")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        thinking_enabled=True,
        reasoning_effort="max",
        max_output_tokens=4096,
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    client.complete(Conversation())

    request = fake_sdk_client.chat.completions.last_request
    assert request["reasoning_effort"] == "max"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["max_tokens"] == 4096


def test_openai_compatible_client_can_explicitly_disable_thinking() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="summary")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        thinking_enabled=True,
        reasoning_effort="max",
    )
    client = OpenAICompatibleLLMClient(
        config=config,
        thinking_enabled=False,
        _client=fake_sdk_client,
    )

    client.complete(Conversation())

    request = fake_sdk_client.chat.completions.last_request
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in request


def test_openai_compatible_client_complete_handles_empty_choices() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        response_choices=[],
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    response = client.complete(Conversation())

    assert response == Message(role="assistant", content="")


def test_openai_compatible_client_streams_response_chunks() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="hello")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)
    conversation = Conversation()
    conversation.add_user_message("hello")

    chunks = list(client.stream_complete(conversation))

    assert chunks == ["he", "llo"]
    assert fake_sdk_client.chat.completions.last_request == {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_openai_compatible_client_captures_non_stream_token_usage() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="hello",
        response_usage=fake_usage(prompt_tokens=120, completion_tokens=8),
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com",
            model="test-model",
        ),
        _client=fake_sdk_client,
    )

    client.complete(Conversation())

    assert client.last_token_usage == TokenUsage(
        prompt_tokens=120,
        completion_tokens=8,
        total_tokens=128,
    )


def test_openai_compatible_client_captures_stream_usage_from_empty_choice_chunk() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(content="hello"),
            fake_empty_stream_chunk(
                usage=fake_usage(prompt_tokens=90, completion_tokens=5)
            ),
        ],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com",
            model="test-model",
        ),
        _client=fake_sdk_client,
    )

    assert list(client.stream_complete(Conversation())) == ["hello"]
    assert client.last_token_usage == TokenUsage(
        prompt_tokens=90,
        completion_tokens=5,
        total_tokens=95,
    )


def test_openai_compatible_client_can_disable_stream_usage_request() -> None:
    fake_sdk_client = FakeOpenAIClient(response_content="hello")
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com",
            model="test-model",
            stream_include_usage=False,
        ),
        _client=fake_sdk_client,
    )

    list(client.stream_complete(Conversation()))

    assert "stream_options" not in fake_sdk_client.chat.completions.last_request


def test_openai_compatible_client_stream_complete_ignores_empty_choice_chunks() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_empty_stream_chunk(),
            fake_stream_chunk(content="hello"),
        ],
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    chunks = list(client.stream_complete(Conversation()))

    assert chunks == ["hello"]


def test_openai_compatible_client_streams_with_tools_text_deltas() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(content="he"),
            fake_stream_chunk(content="llo"),
        ],
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events == [
        AgentEvent(type="text_delta", content="he"),
        AgentEvent(type="text_delta", content="llo"),
    ]
    assert fake_sdk_client.chat.completions.last_request["stream"] is True
    assert fake_sdk_client.chat.completions.last_request["tools"] == [
        {
            "type": "function",
            "function": fake_tool_schema(),
        }
    ]


def test_stream_observation_marks_provider_empty_response() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[fake_stream_chunk(finish_reason="stop")],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com",
            model="test-model",
        ),
        _client=fake_sdk_client,
    )

    assert list(client.stream_with_tools(Conversation(), [fake_tool_schema()])) == []

    observation = client.last_model_response
    assert observation is not None
    assert observation["finish_reason"] == "stop"
    assert observation["content_chars"] == 0
    assert observation["content_non_whitespace_chars"] == 0
    assert observation["tool_call_count"] == 0
    assert observation["stream_chunk_count"] == 1
    assert observation["empty_response"] is True


def test_openai_compatible_client_streams_with_tools_ignores_empty_choice_chunks() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_empty_stream_chunk(),
            fake_stream_chunk(content="hello"),
        ],
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events == [AgentEvent(type="text_delta", content="hello")]


def test_openai_compatible_client_streams_with_tools_accumulates_tool_calls() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(
                tool_calls=[
                    fake_stream_tool_call_delta(
                        index=0,
                        id="call_123",
                        name="read_file",
                        arguments='{"pa',
                    )
                ]
            ),
            fake_stream_chunk(
                tool_calls=[
                    fake_stream_tool_call_delta(
                        index=0,
                        arguments='th": "README.md"}',
                    )
                ]
            ),
        ],
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
    )
    client = OpenAICompatibleLLMClient(config=config, _client=fake_sdk_client)

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events == [
        AgentEvent(
            type="tool_call",
            tool_call=AgentToolCall(
                id="call_123",
                name="read_file",
                arguments={"path": "README.md"},
            ),
        )
    ]


def test_openai_compatible_client_streams_reasoning_separately_from_tool_call() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(reasoning_content="private "),
            fake_stream_chunk(reasoning_content="reasoning"),
            fake_stream_chunk(
                tool_calls=[
                    fake_stream_tool_call_delta(
                        index=0,
                        id="call_stream_reasoning",
                        name="read_file",
                        arguments='{"path": "README.md"}',
                    )
                ]
            ),
        ],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            thinking_enabled=True,
            reasoning_effort="max",
        ),
        _client=fake_sdk_client,
    )

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events[:2] == [
        AgentEvent(type="reasoning_delta", reasoning_content="private "),
        AgentEvent(type="reasoning_delta", reasoning_content="reasoning"),
    ]
    assert events[-1].type == "tool_call"
    assert client.last_reasoning_char_count == len("private reasoning")
    assert "private reasoning" not in repr(events)


def test_thinking_stream_accepts_null_and_empty_reasoning_chunks() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(reasoning_content=None),
            fake_stream_chunk(
                reasoning_content="",
                tool_calls=[
                    fake_stream_tool_call_delta(
                        index=0,
                        id="call_stream_empty_reasoning",
                        name="read_file",
                        arguments='{"path": "README.md"}',
                    )
                ],
            ),
        ],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            thinking_enabled=True,
        ),
        _client=fake_sdk_client,
    )

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events[0] == AgentEvent(
        type="reasoning_state",
        reasoning_state="present_empty",
    )
    assert events[1].type == "tool_call"
    assert all(event.type != "error" for event in events)
    assert client.last_reasoning_char_count == 0


def test_openai_sdk_stream_preserves_present_null_reasoning_field() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chunk-null-reasoning",
            "created": 0,
            "model": "deepseek-v4-flash",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {
                        "reasoning_content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_sdk_stream_null",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[chunk],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            thinking_enabled=True,
        ),
        _client=fake_sdk_client,
    )

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events[0] == AgentEvent(
        type="reasoning_state",
        reasoning_state="present_empty",
    )
    assert events[1].type == "tool_call"


def test_thinking_stream_without_reasoning_field_becomes_model_error() -> None:
    fake_sdk_client = FakeOpenAIClient(
        response_content="",
        stream_chunks=[
            fake_stream_chunk(
                tool_calls=[
                    fake_stream_tool_call_delta(
                        index=0,
                        id="call_stream_missing_reasoning",
                        name="read_file",
                        arguments='{"path": "README.md"}',
                    )
                ]
            )
        ],
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            thinking_enabled=True,
        ),
        _client=fake_sdk_client,
    )

    events = list(client.stream_with_tools(Conversation(), [fake_tool_schema()]))

    assert events == [
        AgentEvent(
            type="error",
            error=(
                "Thinking tool-call response omitted the reasoning_content field."
            ),
        )
    ]


_REASONING_UNSET = object()


@dataclass
class FakeOpenAIClient:
    response_content: str
    response_reasoning_content: object = _REASONING_UNSET
    tool_calls: list[SimpleNamespace] | None = None
    stream_chunks: list[SimpleNamespace] | None = None
    response_choices: list[SimpleNamespace] | None = None
    response_usage: SimpleNamespace | None = None

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(
                response_content=self.response_content,
                response_reasoning_content=self.response_reasoning_content,
                tool_calls=self.tool_calls,
                stream_chunks=self.stream_chunks,
                response_choices=self.response_choices,
                response_usage=self.response_usage,
            )
        )


@dataclass
class FakeChatCompletions:
    response_content: str
    response_reasoning_content: object = _REASONING_UNSET
    tool_calls: list[SimpleNamespace] | None = None
    stream_chunks: list[SimpleNamespace] | None = None
    response_choices: list[SimpleNamespace] | None = None
    response_usage: SimpleNamespace | None = None
    last_request: dict | None = None

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        stream: bool = False,
        stream_options: dict[str, bool] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, object] | None = None,
    ):
        self.last_request = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools is not None:
            self.last_request["tools"] = tools
        if stream_options is not None:
            self.last_request["stream_options"] = stream_options
        if reasoning_effort is not None:
            self.last_request["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            self.last_request["max_tokens"] = max_tokens
        if extra_body is not None:
            self.last_request["extra_body"] = extra_body

        if stream:
            if self.stream_chunks is not None:
                return self.stream_chunks

            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=self.response_content[:2]),
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=self.response_content[2:]),
                        )
                    ]
                ),
            ]

        if self.response_choices is not None:
            return SimpleNamespace(
                choices=self.response_choices,
                usage=self.response_usage,
            )

        message_values: dict[str, object] = {
            "content": self.response_content,
            "tool_calls": self.tool_calls,
        }
        if self.response_reasoning_content is not _REASONING_UNSET:
            message_values["reasoning_content"] = self.response_reasoning_content

        return SimpleNamespace(
            usage=self.response_usage,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(**message_values),
                )
            ]
        )


def fake_tool_schema() -> dict[str, object]:
    return {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }


def fake_sdk_tool_call(
    *,
    id: str,
    name: str,
    arguments: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def fake_stream_chunk(
    *,
    content: str = "",
    reasoning_content: object = _REASONING_UNSET,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
    stop_reason: str | None = None,
) -> SimpleNamespace:
    delta_values: dict[str, object] = {
        "content": content,
        "tool_calls": tool_calls,
    }
    if reasoning_content is not _REASONING_UNSET:
        delta_values["reasoning_content"] = reasoning_content
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(**delta_values),
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
        ]
    )


def fake_empty_stream_chunk(
    *, usage: SimpleNamespace | None = None
) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=usage)


def fake_usage(*, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def fake_stream_tool_call_delta(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )
