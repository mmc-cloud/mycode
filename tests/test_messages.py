from dataclasses import FrozenInstanceError

import pytest

from mycode.agent.events import AgentToolCall
from mycode.messages import Message
from mycode.model_projection import project_model_message


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


def test_message_has_no_provider_serialization_method() -> None:
    message = Message(role="system", content="You are a coding agent.")

    assert not hasattr(message, "to_model_dict")
    assert project_model_message(message) == {
        "role": "system",
        "content": "You are a coding agent.",
    }


def test_assistant_tool_call_has_canonical_projection() -> None:
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

    assert project_model_message(message) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "name": "read_file",
                "arguments": {"path": "README.md"},
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

    tool_call = project_model_message(message)["tool_calls"][0]

    assert tool_call["arguments"] == {"path": "文档/说明.md"}


def test_tool_result_message_has_canonical_projection() -> None:
    message = Message(
        role="tool",
        content="1 | hello",
        tool_call_id="call_123",
    )

    assert project_model_message(message) == {
        "role": "tool",
        "content": "1 | hello",
        "tool_call_id": "call_123",
    }
