from dataclasses import FrozenInstanceError

import pytest

from mycode.agent import (
    AgentEvent,
    AgentModelResponse,
    AgentProgressSnapshot,
    AgentToolCall,
)
from mycode.tools import ToolResult


def test_agent_tool_call_stores_model_requested_tool() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )

    assert tool_call.id == "call_123"
    assert tool_call.name == "read_file"
    assert tool_call.arguments == {"path": "README.md"}


def test_agent_tool_call_is_immutable() -> None:
    tool_call = AgentToolCall(id="call_123", name="glob", arguments={"pattern": "*.py"})

    with pytest.raises(FrozenInstanceError):
        tool_call.name = "grep"


def test_agent_model_response_defaults_to_final_answer_without_tool_calls() -> None:
    response = AgentModelResponse(content="final answer")

    assert response.content == "final answer"
    assert response.tool_calls == []
    assert response.stop_reason == "final_answer"
    assert response.warnings == ()


def test_agent_model_response_can_store_tool_calls() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="grep",
        arguments={"query": "AgentRunner", "path_pattern": "mycode/*.py"},
    )

    response = AgentModelResponse(
        content="",
        tool_calls=[tool_call],
        stop_reason="tool_calls",
    )

    assert response.content == ""
    assert response.tool_calls == [tool_call]
    assert response.stop_reason == "tool_calls"


def test_agent_model_response_gets_independent_default_tool_call_lists() -> None:
    first = AgentModelResponse()
    second = AgentModelResponse()

    assert first.tool_calls is not second.tool_calls


def test_agent_event_can_represent_text_delta() -> None:
    event = AgentEvent(type="text_delta", content="hello")

    assert event.type == "text_delta"
    assert event.content == "hello"
    assert event.tool_call is None
    assert event.tool_result is None
    assert event.stop_reason is None
    assert event.error is None


def test_agent_event_can_represent_context_notice() -> None:
    event = AgentEvent(type="context", content="trimmed=2")

    assert event.type == "context"
    assert event.content == "trimmed=2"


def test_agent_event_can_represent_structured_progress_without_tool_payloads() -> None:
    progress = AgentProgressSnapshot(
        task_phase="DONE",
        behaviors=("verify",),
        transition_reason="validation_succeeded",
        ready_investigation_turn_count=1,
        done_extra_tool_turn_count=0,
    )

    event = AgentEvent(type="progress", progress=progress)

    assert event.progress == progress
    assert not hasattr(progress, "command")
    assert not hasattr(progress, "content")


def test_agent_event_can_represent_model_start() -> None:
    event = AgentEvent(type="model_start")

    assert event.type == "model_start"


def test_agent_event_can_represent_tool_call() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )

    event = AgentEvent(type="tool_call", tool_call=tool_call)

    assert event.type == "tool_call"
    assert event.tool_call == tool_call


def test_agent_event_can_represent_tool_result() -> None:
    result = ToolResult.success(
        content="1 | hello",
        metadata={"path": "README.md"},
    )

    event = AgentEvent(type="tool_result", tool_result=result)

    assert event.type == "tool_result"
    assert event.tool_result == result


def test_agent_event_can_represent_stop() -> None:
    event = AgentEvent(type="stop", stop_reason="max_turns")

    assert event.type == "stop"
    assert event.stop_reason == "max_turns"


def test_agent_event_can_represent_error() -> None:
    event = AgentEvent(type="error", error="model failed")

    assert event.type == "error"
    assert event.error == "model failed"
