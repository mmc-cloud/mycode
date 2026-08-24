from dataclasses import FrozenInstanceError

import pytest

from mycode.agent import AgentToolCall
from mycode.messages import Message


def test_message_stores_role_and_content() -> None:
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"
    assert message.tool_calls == ()
    assert message.tool_call_id is None


def test_message_is_immutable() -> None:
    message = Message(role="assistant", content="hi")

    with pytest.raises(FrozenInstanceError):
        message.content = "changed"


def test_message_can_convert_to_model_dict() -> None:
    message = Message(role="system", content="You are a coding agent.")

    assert message.to_model_dict() == {
        "role": "system",
        "content": "You are a coding agent.",
    }


def test_assistant_tool_call_message_can_convert_to_model_dict() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=(
            AgentToolCall(
                id="call_123",
                name="read_file",
                arguments={"path": "README.md"},
            ),
        ),
    )

    assert message.to_model_dict() == {
        "role": "assistant",
        "content": "",
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
    }


def test_assistant_tool_call_message_preserves_non_ascii_arguments() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=(
            AgentToolCall(
                id="call_123",
                name="read_file",
                arguments={"path": "文档/说明.md"},
            ),
        ),
    )

    tool_call = message.to_model_dict()["tool_calls"][0]

    assert tool_call["function"]["arguments"] == '{"path": "文档/说明.md"}'


def test_tool_result_message_can_convert_to_model_dict() -> None:
    message = Message(
        role="tool",
        content="1 | hello",
        tool_call_id="call_123",
    )

    assert message.to_model_dict() == {
        "role": "tool",
        "content": "1 | hello",
        "tool_call_id": "call_123",
    }
