from collections.abc import Iterator

import pytest

from mycode.context_budget import ContextBudget, ContextBudgetExceededError, TokenUsage
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.messages import Message
from mycode.session import ChatSession


def context_budget(max_input_tokens: int) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )


def test_send_user_message_returns_assistant_reply() -> None:
    session = ChatSession(llm_client=FakeLLMClient(responses=["hello"]))

    reply = session.send_user_message("hi")

    assert reply == Message(role="assistant", content="hello")


def test_send_user_message_saves_user_and_assistant_messages() -> None:
    session = ChatSession(llm_client=FakeLLMClient(responses=["assistant reply"]))

    session.send_user_message("user request")

    assert session.conversation.get_messages() == [
        Message(role="user", content="user request"),
        Message(role="assistant", content="assistant reply"),
    ]


def test_send_user_message_preserves_multi_turn_history() -> None:
    session = ChatSession(llm_client=FakeLLMClient(responses=["first reply", "second reply"]))

    session.send_user_message("first request")
    session.send_user_message("second request")

    assert session.conversation.get_messages() == [
        Message(role="user", content="first request"),
        Message(role="assistant", content="first reply"),
        Message(role="user", content="second request"),
        Message(role="assistant", content="second reply"),
    ]


def test_send_user_message_uses_model_context_without_trimming_history() -> None:
    client = RecordingChatLLMClient(reply=Message(role="assistant", content="reply"))
    session = ChatSession(
        llm_client=client,
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 20),
                Message(role="assistant", content="old reply " * 20),
            ]
        ),
        context_budget=context_budget(100),
    )

    session.send_user_message("current")

    assert client.seen_complete_messages == [[{"role": "user", "content": "current"}]]
    assert session.last_model_context is not None
    assert session.last_model_context.trimmed is True
    assert session.last_model_context.trimmed_message_count == 2
    assert session.conversation.get_messages() == [
        Message(role="user", content="old request " * 20),
        Message(role="assistant", content="old reply " * 20),
        Message(role="user", content="current"),
        Message(role="assistant", content="reply"),
    ]


def test_stream_user_message_yields_chunks_and_saves_full_reply() -> None:
    session = ChatSession(
        llm_client=FakeLLMClient(responses=["streaming reply"], stream_chunk_size=5)
    )

    chunks = list(session.stream_user_message("user request"))

    assert chunks == ["strea", "ming ", "reply"]
    assert session.conversation.get_messages() == [
        Message(role="user", content="user request"),
        Message(role="assistant", content="streaming reply"),
    ]


def test_stream_user_message_prepares_context_before_stream_iteration() -> None:
    client = RecordingChatLLMClient(stream_chunks=["reply"])
    session = ChatSession(llm_client=client)

    chunks = session.stream_user_message("user request")

    assert session.last_model_context is not None
    assert session.last_model_context.messages == (
        Message(role="user", content="user request"),
    )
    assert client.seen_stream_messages == []

    assert list(chunks) == ["reply"]
    assert client.seen_stream_messages == [
        [{"role": "user", "content": "user request"}]
    ]


def test_stream_user_message_uses_model_context_without_trimming_history() -> None:
    client = RecordingChatLLMClient(stream_chunks=["stream", " reply"])
    session = ChatSession(
        llm_client=client,
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 20),
                Message(role="assistant", content="old reply " * 20),
            ]
        ),
        context_budget=context_budget(100),
    )

    chunks = list(session.stream_user_message("current"))

    assert chunks == ["stream", " reply"]
    assert client.seen_stream_messages == [[{"role": "user", "content": "current"}]]
    assert session.last_model_context is not None
    assert session.last_model_context.trimmed is True
    assert session.conversation.get_messages()[-2:] == [
        Message(role="user", content="current"),
        Message(role="assistant", content="stream reply"),
    ]


def test_send_user_message_does_not_call_model_when_context_is_over_budget() -> None:
    client = RecordingChatLLMClient(reply=Message(role="assistant", content="unused"))
    session = ChatSession(
        llm_client=client,
        context_budget=context_budget(60),
    )

    with pytest.raises(ContextBudgetExceededError, match="exceeds"):
        session.send_user_message("中" * 100)

    assert client.seen_complete_messages == []
    assert session.last_model_context is not None
    assert session.last_model_context.estimate.over_budget is True


def test_stream_user_message_does_not_call_model_when_context_is_over_budget() -> None:
    client = RecordingChatLLMClient(stream_chunks=["unused"])
    session = ChatSession(
        llm_client=client,
        context_budget=context_budget(60),
    )

    with pytest.raises(ContextBudgetExceededError, match="exceeds"):
        list(session.stream_user_message("中" * 100))

    assert client.seen_stream_messages == []
    assert session.last_model_context is not None
    assert session.last_model_context.estimate.over_budget is True


def test_session_uses_provider_usage_to_calibrate_next_request() -> None:
    client = FakeLLMClient(
        responses=["first", "second"],
        token_usages=[
            TokenUsage(prompt_tokens=80, completion_tokens=10, total_tokens=90),
            None,
        ],
    )
    session = ChatSession(llm_client=client)

    session.send_user_message("a" * 100)
    assert session.last_token_usage == TokenUsage(
        prompt_tokens=80,
        completion_tokens=10,
        total_tokens=90,
    )

    session.send_user_message("next")

    assert session.last_model_context is not None
    assert session.last_model_context.estimate.token_estimate_source == "calibrated"


class RecordingChatLLMClient:
    def __init__(
        self,
        *,
        reply: Message | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.reply = Message(role="assistant", content="") if reply is None else reply
        self.stream_chunks = [] if stream_chunks is None else stream_chunks
        self.seen_complete_messages: list[list[dict[str, object]]] = []
        self.seen_stream_messages: list[list[dict[str, object]]] = []
        self.last_token_usage = None

    def complete(self, conversation: Conversation) -> Message:
        self.seen_complete_messages.append(conversation.to_model_messages())
        return self.reply

    def stream_complete(self, conversation: Conversation) -> Iterator[str]:
        self.seen_stream_messages.append(conversation.to_model_messages())
        yield from self.stream_chunks

    def stream_with_tools(self, conversation: Conversation, tools: list[dict[str, object]]):
        raise NotImplementedError
        yield
