import io
import json
from types import SimpleNamespace

import pytest

from mycode.adapters.jsonl import (
    JsonlChannel,
    JsonlConfirmer,
    JsonlMCPTrustConfirmer,
    JsonlProtocolError,
    run_jsonl_runtime,
    serialize_runtime_event,
)
from mycode.agent.events import AgentEvent
from mycode.agent.outcome import AgentRunOutcome
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import SessionStartRequest
from mycode.mcp.config import MCPConfig
from mycode.mcp.models import MCPServerStatus
from mycode.mcp.trust import MCPTrustRequest, MCPTrustServer
from mycode.permissions import (
    ConfirmationRequest,
    PermissionDecision,
    PermissionRequest,
)


def test_runtime_event_wraps_existing_agent_event_and_preserves_identity() -> None:
    agent_event = AgentEvent(type="text_delta", content="hello")
    runtime_event = RuntimeEvent(
        type="agent",
        turn_id="turn-1",
        agent_event=agent_event,
    )

    assert runtime_event.agent_event is agent_event
    assert serialize_runtime_event(runtime_event) == {
        "type": "agent_event",
        "turn_id": "turn-1",
        "event": {"type": "text_delta", "content": "hello"},
    }


def test_runtime_event_serializes_ready_status_and_outcome() -> None:
    outcome = AgentRunOutcome.from_stop_reason("final_answer")
    events = [
        RuntimeEvent(
            type="runtime_ready",
            session_id="session-1",
            session_title="Demo",
            session_created=False,
            compact_state_recovered=True,
            instruction_sources=("project",),
            instruction_warnings=("warning",),
            skill_warnings=("skill warning",),
        ),
        RuntimeEvent(
            type="mcp_status",
            mcp_status=MCPServerStatus(
                alias="github",
                status="connected",
                tool_count=2,
            ),
        ),
        RuntimeEvent(type="turn_finished", turn_id="turn-1", outcome=outcome),
    ]

    assert serialize_runtime_event(events[0]) == {
        "type": "runtime_ready",
        "session_id": "session-1",
        "session_title": "Demo",
        "session_created": False,
        "compact_state_recovered": True,
        "instruction_sources": ["project"],
        "instruction_warnings": ["warning"],
        "skill_warnings": ["skill warning"],
    }
    assert serialize_runtime_event(events[1]) == {
        "type": "mcp_status",
        "alias": "github",
        "status": "connected",
        "tool_count": 2,
        "error_type": None,
        "error_summary": None,
    }
    assert serialize_runtime_event(events[2]) == {
        "type": "turn_finished",
        "turn_id": "turn-1",
        "status": "completed",
        "stop_reason": "final_answer",
    }


@pytest.mark.parametrize(
    "event_type",
    ["runtime_warning", "runtime_error", "runtime_closed"],
)
def test_runtime_event_rejects_jsonl_only_wire_types(event_type) -> None:
    with pytest.raises(ValueError, match="Unsupported runtime event type"):
        RuntimeEvent(type=event_type)


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("{broken\n", "invalid_json"),
        ('{"version":2,"type":"turn"}\n', "unsupported_version"),
        ('{"version":1}\n', "missing_type"),
    ],
)
def test_jsonl_channel_rejects_malformed_input(line, code) -> None:
    channel = JsonlChannel(io.StringIO(line), io.StringIO())

    with pytest.raises(JsonlProtocolError) as error:
        channel.read_message()

    assert error.value.code == code


class FakeRuntimeApplication:
    def __init__(self, seen_turns: list[tuple[str, str]]) -> None:
        self.seen_turns = seen_turns
        self.close_calls = 0
        self.interrupt_calls = 0

    def startup_events(self):
        yield RuntimeEvent(
            type="runtime_ready",
            session_id="session-1",
            session_title="Machine session",
            session_created=True,
        )

    def run_turn(self, content, *, turn_id=None, event_handler=None):
        self.seen_turns.append((content, turn_id))
        outcome = AgentRunOutcome.from_stop_reason("final_answer")
        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    type="agent",
                    turn_id=turn_id,
                    agent_event=AgentEvent(type="text_delta", content="done"),
                )
            )
            event_handler(
                RuntimeEvent(
                    type="turn_finished",
                    turn_id=turn_id,
                    outcome=outcome,
                )
            )
        return outcome

    def close(self) -> None:
        self.close_calls += 1

    def interrupt(self) -> None:
        self.interrupt_calls += 1


class FatalRuntimeApplication(FakeRuntimeApplication):
    def run_turn(self, content, *, turn_id=None, event_handler=None):
        self.seen_turns.append((content, turn_id))
        raise RuntimeError("internal turn detail")


class KeyboardInterruptRuntimeApplication(FakeRuntimeApplication):
    def run_turn(self, content, *, turn_id=None, event_handler=None):
        self.seen_turns.append((content, turn_id))
        raise KeyboardInterrupt


class RuntimeFailureApplication(FakeRuntimeApplication):
    def run_turn(self, content, *, turn_id=None, event_handler=None):
        self.seen_turns.append((content, turn_id))
        outcome = AgentRunOutcome.from_stop_reason("model_error")
        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    type="agent",
                    turn_id=turn_id,
                    agent_event=AgentEvent(
                        type="error",
                        error="model unavailable",
                    ),
                )
            )
            event_handler(
                RuntimeEvent(
                    type="turn_finished",
                    turn_id=turn_id,
                    outcome=outcome,
                )
            )
        return outcome


class CloseFailRuntimeApplication(FakeRuntimeApplication):
    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close detail")


class InterruptFailRuntimeApplication(FatalRuntimeApplication):
    def interrupt(self) -> None:
        self.interrupt_calls += 1
        raise RuntimeError("interrupt detail")


def run_fake_runtime(
    tmp_path,
    monkeypatch,
    application,
    input_text: str,
):
    monkeypatch.setattr(
        "mycode.adapters.jsonl.start_agent_application_session",
        lambda *args, **kwargs: application,
    )
    output_stream = io.StringIO()
    error_stream = io.StringIO()
    exit_code = run_jsonl_runtime(
        workspace_path=tmp_path,
        mcp_config=MCPConfig(),
        input_stream=io.StringIO(input_text),
        output_stream=output_stream,
        error_stream=error_stream,
    )
    return exit_code, output_stream, error_stream


def test_fatal_turn_exception_interrupts_and_terminates_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    application = FatalRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"turn","turn_id":"turn-1","content":"first"}\n'
        '{"version":1,"type":"turn","turn_id":"turn-2","content":"second"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert exit_code != 0
    assert application.seen_turns == [("first", "turn-1")]
    assert application.interrupt_calls == 1
    assert [payload["type"] for payload in payloads] == [
        "runtime_ready",
        "runtime_error",
    ]
    assert payloads[-1]["code"] == "turn_failed"
    assert payloads[-1]["turn_id"] == "turn-1"
    assert "turn_finished" not in output_stream.getvalue()
    assert "internal turn detail" not in output_stream.getvalue()
    assert "turn failed: RuntimeError: internal turn detail" in error_stream.getvalue()


def test_normal_runtime_failure_outcome_allows_next_turn(
    tmp_path,
    monkeypatch,
) -> None:
    application = RuntimeFailureApplication([])
    exit_code, output_stream, _ = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"turn","turn_id":"turn-1","content":"first"}\n'
        '{"version":1,"type":"turn","turn_id":"turn-2","content":"second"}\n'
        '{"version":1,"type":"close"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    finished = [
        payload for payload in payloads if payload["type"] == "turn_finished"
    ]
    assert exit_code == 0
    assert application.seen_turns == [("first", "turn-1"), ("second", "turn-2")]
    assert [payload["status"] for payload in finished] == [
        "runtime_failure",
        "runtime_failure",
    ]
    assert [payload["stop_reason"] for payload in finished] == [
        "model_error",
        "model_error",
    ]
    assert not any(payload.get("code") == "turn_failed" for payload in payloads)


def test_close_failure_emits_lifecycle_error_and_interrupts(
    tmp_path,
    monkeypatch,
) -> None:
    application = CloseFailRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"close"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    lifecycle_errors = [
        payload for payload in payloads if payload.get("code") == "lifecycle_failed"
    ]
    assert exit_code != 0
    assert application.close_calls == 1
    assert application.interrupt_calls == 1
    assert len(lifecycle_errors) == 1
    assert "runtime_closed" not in output_stream.getvalue()
    assert "close detail" in error_stream.getvalue()


def test_eof_closes_runtime_normally(tmp_path, monkeypatch) -> None:
    application = FakeRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        "",
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert exit_code == 0
    assert payloads[-1]["type"] == "runtime_closed"
    assert application.close_calls == 1
    assert application.interrupt_calls == 0
    assert error_stream.getvalue() == ""


def test_interrupt_cleanup_failure_is_reported_once_without_changing_exit_code(
    tmp_path,
    monkeypatch,
) -> None:
    application = InterruptFailRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"turn","turn_id":"turn-1","content":"first"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    lifecycle_errors = [
        payload for payload in payloads if payload.get("code") == "lifecycle_failed"
    ]
    assert exit_code != 0
    assert len(lifecycle_errors) == 1
    assert "interrupt detail" in error_stream.getvalue()
    assert all(
        isinstance(json.loads(line), dict)
        for line in output_stream.getvalue().splitlines()
    )


def test_keyboard_interrupt_keeps_exit_130_and_does_not_emit_closed(
    tmp_path,
    monkeypatch,
) -> None:
    application = KeyboardInterruptRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"turn","turn_id":"turn-1","content":"first"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert exit_code == 130
    assert application.interrupt_calls == 1
    assert [payload["type"] for payload in payloads] == ["runtime_ready"]
    assert "runtime_closed" not in output_stream.getvalue()
    assert error_stream.getvalue() == ""


def test_jsonl_machine_runtime_is_json_only_and_forwards_turn_events(
    tmp_path,
    monkeypatch,
) -> None:
    input_stream = io.StringIO(
        '{"version":1,"type":"turn","turn_id":"turn-123","content":"fix it"}\n'
        '{"version":1,"type":"close"}\n'
    )
    output_stream = io.StringIO()
    error_stream = io.StringIO()
    seen_turns: list[tuple[str, str]] = []
    application = FakeRuntimeApplication(seen_turns)
    monkeypatch.setattr(
        "mycode.adapters.jsonl.start_agent_application_session",
        lambda *args, **kwargs: application,
    )

    exit_code = run_jsonl_runtime(
        workspace_path=tmp_path,
        mcp_config=MCPConfig(),
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert exit_code == 0
    assert seen_turns == [("fix it", "turn-123")]
    assert [payload["type"] for payload in payloads] == [
        "runtime_ready",
        "agent_event",
        "turn_finished",
        "runtime_closed",
    ]
    assert all(payload["version"] == 1 for payload in payloads)
    assert "assistant>" not in output_stream.getvalue()
    assert "session>" not in output_stream.getvalue()
    assert error_stream.getvalue() == ""
    assert application.close_calls == 1
    assert application.interrupt_calls == 0


@pytest.mark.parametrize(
    ("session_request_value", "expected_mode", "expected_id"),
    [
        (None, "continue", None),
        (SessionStartRequest(mode="new"), "new", None),
        (SessionStartRequest(mode="continue"), "continue", None),
        (SessionStartRequest(mode="resume", session_id="session-2"), "resume", "session-2"),
    ],
)
def test_machine_runtime_never_opens_session_menu(
    tmp_path,
    monkeypatch,
    session_request_value,
    expected_mode,
    expected_id,
) -> None:
    calls: list[SessionStartRequest] = []
    application = FakeRuntimeApplication([])

    def fake_start(*args, **kwargs):
        calls.append(kwargs["request"])
        return application

    monkeypatch.setattr(
        "mycode.adapters.jsonl.start_agent_application_session",
        fake_start,
    )
    output_stream = io.StringIO()

    assert run_jsonl_runtime(
        workspace_path=tmp_path,
        session_request=session_request_value,
        mcp_config=MCPConfig(),
        input_stream=io.StringIO('{"version":1,"type":"close"}\n'),
        output_stream=output_stream,
        error_stream=io.StringIO(),
    ) == 0

    assert len(calls) == 1
    assert calls[0].mode == expected_mode
    assert calls[0].session_id == expected_id
    assert "session>" not in output_stream.getvalue()


@pytest.mark.parametrize("decision", ["once", "task", "session", "reject"])
def test_jsonl_permission_handshake_maps_to_existing_confirmation_result(
    monkeypatch,
    decision,
) -> None:
    request_id = "permission-1"
    monkeypatch.setattr(
        "mycode.adapters.jsonl.uuid4",
        lambda: SimpleNamespace(hex=request_id),
    )
    input_stream = io.StringIO(
        json.dumps(
            {
                "version": 1,
                "type": "permission_response",
                "request_id": request_id,
                "decision": decision,
            }
        )
        + "\n"
    )
    output_stream = io.StringIO()
    confirmation = JsonlConfirmer(JsonlChannel(input_stream, output_stream))
    permission_request = PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write file",
        target="notes.txt",
        arguments={"path": "notes.txt"},
    )
    result = confirmation.confirm(
        ConfirmationRequest(
            permission_request=permission_request,
            permission_decision=PermissionDecision.ask(),
            prompt="allow?",
        )
    )

    assert result.status == ("rejected" if decision == "reject" else "approved")
    assert result.scope == (None if decision == "reject" else decision)
    request_payload = json.loads(output_stream.getvalue().splitlines()[0])
    assert request_payload["type"] == "permission_request"
    assert request_payload["request_id"] == request_id


def test_jsonl_permission_rejects_wrong_request_id_then_accepts_correct_one(
    monkeypatch,
) -> None:
    request_id = "permission-1"
    monkeypatch.setattr(
        "mycode.adapters.jsonl.uuid4",
        lambda: SimpleNamespace(hex=request_id),
    )
    input_stream = io.StringIO(
        '{"version":1,"type":"permission_response","request_id":"wrong","decision":"once"}\n'
        '{"version":1,"type":"permission_response","request_id":"permission-1","decision":"once"}\n'
    )
    output_stream = io.StringIO()
    confirmation = JsonlConfirmer(JsonlChannel(input_stream, output_stream))
    permission_request = PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write file",
    )

    result = confirmation.confirm(
        ConfirmationRequest(
            permission_request=permission_request,
            permission_decision=PermissionDecision.ask(),
            prompt="allow?",
        )
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert result.status == "approved"
    assert payloads[1]["type"] == "runtime_error"
    assert payloads[1]["code"] == "request_id_mismatch"


def test_jsonl_permission_eof_rejects() -> None:
    confirmation = JsonlConfirmer(JsonlChannel(io.StringIO(), io.StringIO()))
    permission_request = PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write file",
    )

    result = confirmation.confirm(
        ConfirmationRequest(
            permission_request=permission_request,
            permission_decision=PermissionDecision.ask(),
            prompt="allow?",
        )
    )

    assert result.status == "rejected"
    assert result.metadata == {"input": "eof"}


def test_jsonl_mcp_trust_handshake_is_safe_and_accepts_response(monkeypatch) -> None:
    request_id = "trust-1"
    monkeypatch.setattr(
        "mycode.adapters.jsonl.uuid4",
        lambda: SimpleNamespace(hex=request_id),
    )
    input_stream = io.StringIO(
        '{"version":1,"type":"mcp_trust_response","request_id":"trust-1","approved":true}\n'
    )
    output_stream = io.StringIO()
    confirmer = JsonlMCPTrustConfirmer(JsonlChannel(input_stream, output_stream))
    request = MCPTrustRequest(
        servers=(
            MCPTrustServer(
                alias="local",
                transport="stdio",
                command="python",
                args=("server.py",),
                env_keys=("TOKEN",),
            ),
            MCPTrustServer(
                alias="remote",
                transport="streamable_http",
                url_template="https://api.example.test/mcp",
                destination="https://api.example.test",
                header_keys=("Authorization",),
            ),
        )
    )

    assert confirmer.confirm(request) is True
    payload = json.loads(output_stream.getvalue().splitlines()[0])
    rendered = output_stream.getvalue()
    assert payload["type"] == "mcp_trust_request"
    assert payload["servers"][0]["env_keys"] == ["TOKEN"]
    assert payload["servers"][1]["header_keys"] == ["Authorization"]
    assert "secret" not in rendered.lower()


def test_jsonl_protocol_errors_are_structured_and_runtime_continues(
    tmp_path,
    monkeypatch,
) -> None:
    application = FakeRuntimeApplication([])
    monkeypatch.setattr(
        "mycode.adapters.jsonl.start_agent_application_session",
        lambda *args, **kwargs: application,
    )
    input_stream = io.StringIO(
        "{broken\n"
        '{"version":2,"type":"turn"}\n'
        '{"version":1,"type":"unknown"}\n'
        '{"version":1,"type":"turn","content":"missing id"}\n'
        '{"version":1,"type":"close"}\n'
    )
    output_stream = io.StringIO()

    assert run_jsonl_runtime(
        workspace_path=tmp_path,
        mcp_config=MCPConfig(),
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=io.StringIO(),
    ) == 0

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    error_codes = [
        payload["code"]
        for payload in payloads
        if payload["type"] == "runtime_error"
    ]
    assert error_codes == [
        "invalid_json",
        "unsupported_version",
        "unexpected_message",
        "missing_turn_id",
    ]
    assert payloads[-1]["type"] == "runtime_closed"


def test_blank_turn_is_rejected_and_next_turn_can_run(
    tmp_path,
    monkeypatch,
) -> None:
    application = FakeRuntimeApplication([])
    exit_code, output_stream, error_stream = run_fake_runtime(
        tmp_path,
        monkeypatch,
        application,
        '{"version":1,"type":"turn","turn_id":"blank-1","content":""}\n'
        '{"version":1,"type":"turn","turn_id":"blank-2","content":" \\t\\n"}\n'
        '{"version":1,"type":"turn","turn_id":"turn-1","content":"valid"}\n'
        '{"version":1,"type":"close"}\n',
    )

    payloads = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    invalid_turns = [
        payload
        for payload in payloads
        if payload.get("code") == "invalid_turn_content"
    ]
    assert exit_code == 0
    assert application.seen_turns == [("valid", "turn-1")]
    assert len(invalid_turns) == 2
    assert error_stream.getvalue() == ""
