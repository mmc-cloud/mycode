import pytest

from mycode.agent import AgentModelResponse, AgentToolCall
from mycode.conversation import Conversation
from mycode.messages import Message


def test_new_conversation_starts_empty() -> None:
    conversation = Conversation()

    assert conversation.get_messages() == []


def test_conversation_can_start_from_existing_messages() -> None:
    original_messages = [Message(role="system", content="You are a coding agent.")]

    conversation = Conversation.from_messages(original_messages)
    original_messages.append(Message(role="user", content="This should not leak in."))

    assert conversation.get_messages() == [
        Message(role="system", content="You are a coding agent."),
    ]


def test_conversation_observer_only_receives_new_messages() -> None:
    existing = Message(role="user", content="existing")
    added = Message(role="assistant", content="new")
    observed: list[Message] = []
    conversation = Conversation.from_messages(
        [existing],
        on_message_added=observed.append,
    )

    conversation.add_message(added)

    assert observed == [added]
    assert conversation.get_messages() == [existing, added]


def test_conversation_does_not_append_when_observer_fails() -> None:
    conversation = Conversation.from_messages(
        [],
        on_message_added=lambda message: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        conversation.add_user_message("not persisted")

    assert conversation.get_messages() == []


def test_add_message_appends_existing_message() -> None:
    conversation = Conversation()
    message = Message(role="user", content="hello")

    conversation.add_message(message)

    assert conversation.get_messages() == [message]


def test_add_role_helpers_append_messages() -> None:
    conversation = Conversation()

    conversation.add_system_message("system rules")
    conversation.add_user_message("user request")
    conversation.add_assistant_message("assistant reply")

    assert conversation.get_messages() == [
        Message(role="system", content="system rules"),
        Message(role="user", content="user request"),
        Message(role="assistant", content="assistant reply"),
    ]


def test_get_messages_returns_copy() -> None:
    conversation = Conversation.from_messages([Message(role="user", content="hello")])

    messages = conversation.get_messages()
    messages.append(Message(role="assistant", content="changed outside"))

    assert conversation.get_messages() == [Message(role="user", content="hello")]


def test_conversation_can_convert_to_model_messages() -> None:
    conversation = Conversation()
    conversation.add_system_message("system rules")
    conversation.add_user_message("hello")

    assert conversation.to_model_messages() == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "hello"},
    ]


def test_conversation_can_append_assistant_tool_calls() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    conversation = Conversation()

    conversation.add_assistant_tool_calls(content="", tool_calls=[tool_call])

    assert conversation.get_messages() == [
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
        )
    ]


def test_conversation_can_append_tool_result_message() -> None:
    conversation = Conversation()

    conversation.add_tool_result_message(
        tool_call_id="call_123",
        content="1 | hello",
    )

    assert conversation.get_messages() == [
        Message(
            role="tool",
            content="1 | hello",
            tool_call_id="call_123",
        )
    ]


def test_conversation_can_convert_tool_call_messages_to_model_messages() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    conversation = Conversation()
    conversation.add_user_message("Read README")
    conversation.add_assistant_tool_calls(
        content="",
        tool_calls=[tool_call],
        reasoning_content="private synthetic reasoning",
    )
    conversation.add_tool_result_message(
        tool_call_id="call_123",
        content="1 | hello",
    )

    assert conversation.to_model_messages() == [
        {"role": "user", "content": "Read README"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "private synthetic reasoning",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "1 | hello",
            "tool_call_id": "call_123",
        },
    ]


def test_reasoning_content_is_hidden_from_message_repr_and_requires_tool_calls() -> None:
    tool_call = AgentToolCall(
        id="call_reasoning",
        name="read_file",
        arguments={"path": "README.md"},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(tool_call,),
        reasoning_content="private synthetic reasoning",
    )

    assert "private synthetic reasoning" not in repr(message)
    with pytest.raises(ValueError, match="assistant tool-call"):
        Message(
            role="assistant",
            content="done",
            reasoning_content="must not persist",
        )


def test_present_empty_reasoning_is_replayed_as_null() -> None:
    tool_call = AgentToolCall(
        id="call_empty_reasoning",
        name="read_file",
        arguments={"path": "README.md"},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(tool_call,),
        reasoning_state="present_empty",
    )

    assert message.reasoning_content is None
    assert message.to_model_dict()["reasoning_content"] is None

    with pytest.raises(ValueError, match="assistant tool-call"):
        AgentModelResponse(
            stop_reason="tool_calls",
            reasoning_state="present_empty",
        )


def test_clear_removes_all_messages() -> None:
    conversation = Conversation.from_messages([Message(role="user", content="hello")])

    conversation.clear()

    assert conversation.get_messages() == []
