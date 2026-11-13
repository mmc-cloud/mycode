"""Structured controls against real SessionModels, storage and SDK clients.

HTTP discovery is mocked here; real inference is covered by the live smoke.
"""
import io
import json
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from mycode.adapters.jsonl import JsonlChannel, _run_management_message, run_jsonl_runtime
from mycode.application.agent_session import start_agent_application_session
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity

KEY = "synthetic-private-api-key"
HEADER = "synthetic-private-header"


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = LLMConfig(api_key=KEY, base_url="https://env.example/v1", model="main",
                       headers={"X-Private": HEADER})
    store = SessionStore(tmp_path / "projects")
    app = start_agent_application_session(store, ProjectIdentity.from_workspace(workspace),
        request=SessionStartRequest(mode="new"), llm_config=config, mcp_config=MCPConfig())
    yield app, store, config, workspace
    app.close()


def send(app, kind="model", **fields):
    output, errors = io.StringIO(), io.StringIO()
    message = {"version": 1, "type": kind, "request_id": "request-1", **fields}
    channel = JsonlChannel(io.StringIO(json.dumps(message) + "\n"), output, errors)
    _run_management_message(channel, app, channel.read_message())
    assert KEY not in output.getvalue() + errors.getvalue()
    assert HEADER not in output.getvalue() + errors.getvalue()
    result = json.loads(output.getvalue())
    assert result["type"] == kind + "_result" and result["request_id"] == "request-1"
    return result


def add(app, identifier="other"):
    result = send(app, "connect", action="add", id=identifier,
        base_url="https://other.example/v1", api_key=KEY, headers={"X-Private": HEADER})
    assert result["ok"]


def test_three_roles_typed_options_and_follow_keep_runtime_objects(application):
    app, *_ = application
    runner = app.runner
    identities = (runner, runner.conversation, runner.tool_registry, app.mcp_manager, runner.compactor)
    runner.conversation.add_user_message("Keep this conversation.")
    canonical = runner.conversation.get_messages()
    add(app)
    main = send(app, action="set", role="main", model="main-alt", thinking="on", effort="low",
                window=64000, reserve=4000, margin=2000, output=1000)
    assert main["ok"]
    assert main["roles"]["main"]["effort"] == "low"
    assert main["roles"]["main"]["window"] == 64000
    child = send(app, action="set", role="subagent", model="worker", connection="other",
                 thinking="on", effort="high", window=32000, reserve=2000, margin=1000)
    assert child["ok"]
    compact = send(app, action="set", role="compact", model=None)
    assert compact["ok"] and compact["roles"]["compact"]["thinking"] is False
    assert compact["roles"]["compact"]["model"] == "main-alt"
    assert compact["roles"]["compact"]["window"] == 64000
    send(app, action="set", role="main", model="next", thinking="auto")
    status = send(app)["roles"]
    assert status["subagent"]["model"] == "worker" and status["subagent"]["connection"] == "other"
    assert status["compact"]["model"] == "next"
    followed = send(app, action="set", role="subagent", model=None)["roles"]["subagent"]
    assert followed["model"] == "next" and followed["connection"] == "env"
    assert followed["selection"]["model"] is None
    assert (runner, runner.conversation, runner.tool_registry, app.mcp_manager, runner.compactor) == identities
    assert runner.conversation.get_messages() == canonical
    assert send(app, action="set", role="main", model="next", window=None, reserve=None,
                margin=None, thinking=None, effort=None, output="auto")["ok"]
    assert app.models.windows["main"].tokens == 128000


def test_connection_crud_defaults_discovery_and_manual_ids(application, monkeypatch):
    app, *_ = application
    add(app)
    listed = send(app, "connect")
    assert listed["connections"] == [{"id": "env", "read_only": True},
        {"id": "other", "base_url": "https://other.example/v1", "read_only": False}]
    assert send(app, "connect", action="edit", id="other", headers={})["ok"]
    assert app.models.store.read().connections["other"].headers == {}
    assert send(app, "connect", action="default", id="other")["default_connection"] == "other"
    assert app.runner.llm_client.model == "main"  # Future sessions only.
    seen = []
    def transport(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "listed", "object": "model",
            "context_window": 64000, "headers": {"private": HEADER}, "api_key": KEY}]})
    monkeypatch.setattr("mycode.providers.openai_compatible.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    result = send(app, "connect", action="models", id="other")
    assert result["ok"] and result["available"]
    assert result["models"] == [{"id": "listed", "context_window": 64000}]
    assert len(seen) == 1 and seen[0].method == "GET" and seen[0].url.path == "/v1/models"
    assert send(app, "connect", action="use", id="other", model="not-listed")["ok"]
    assert app.runner.llm_client.model == "not-listed" and app.models.windows["main"].tokens == 128000
    assert not send(app, "connect", action="delete", id="other")["ok"]
    assert send(app, action="set", role="main", model="main", connection="env")["ok"]
    assert send(app, "connect", action="default", id="env")["ok"]
    assert send(app, "connect", action="delete", id="other")["ok"]
    assert not app.models.store.read().connections


@pytest.mark.parametrize("failure", ["client", "persist", "budget", "reference"])
def test_failed_switch_preserves_runtime_and_disk_without_exception_echo(application, monkeypatch, failure):
    app, *_ = application
    assert send(app, action="set", role="main", model="main", window=64000)["ok"]
    runner = app.runner
    before = (runner.llm_client, runner.context_budget, runner.compactor.llm_client, app.models.selection)
    windows = app.models.store.path.read_bytes()
    snapshot_path = app.active_project_session.writer.layout.root / "model-selection.json"
    snapshot = snapshot_path.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError(KEY + HEADER)
    options = {"window": 32000}
    if failure == "client":
        monkeypatch.setattr("mycode.providers.factory.ModelClientFactory.create", fail)
    elif failure == "persist":
        monkeypatch.setattr(app.active_project_session.writer, "save_model_selection", fail)
    elif failure == "budget":
        options.update(reserve=32000)
    else:
        options.update(connection="missing")
    result = send(app, action="set", role="main", model="target", **options)
    assert not result["ok"] and result["error"]["code"] in {"configuration_rejected", "operation_failed"}
    assert before == (runner.llm_client, runner.context_budget, runner.compactor.llm_client, app.models.selection)
    assert app.models.store.path.read_bytes() == windows
    assert snapshot_path.read_bytes() == snapshot


@pytest.mark.parametrize("role", ["main", "subagent", "compact"])
def test_active_requests_reject_model_and_connection_changes(application, role):
    app, *_ = application
    with app.models.activity.operation(role):
        assert not send(app, action="set", role="main", model="changed")["ok"]
        assert not send(app, "connect", action="add", id="other", base_url="https://other.example/v1", api_key=KEY)["ok"]
        assert not send(app, "connect", action="models", id="env")["ok"]
    assert app.runner.llm_client.model == "main"
    assert not app.models.store.read().connections


@pytest.mark.parametrize("kind,fields", [
    ("model", {"action": "set", "role": "main"}),
    ("model", {"action": "set", "role": "main", "model": "x", "window": True}),
    ("model", {"action": "set", "role": "main", "model": "x", "window": "64000"}),
    ("model", {"action": "set", "role": "main", "model": "x", "effort": "extreme"}),
    ("model", {"action": "set", "role": "main", "model": "x", "output": 0}),
    ("model", {"action": "status", "api_key": KEY}),
    ("model", {"action": "status", "model": None}),
    ("connect", {"action": "add", "id": "other", "api_key": KEY}),
    ("connect", {"action": "add", "id": "other", "api_key": KEY, "base_url": "https://x/v1", "headers": HEADER}),
    ("connect", {"action": "delete", "id": "other", "api_key": KEY}),
    ("connect", {"action": "edit", "id": "other", "headers": None}),
    ("connect", {"action": "list", "id": "other"}),
])
def test_invalid_structured_parameters_are_rejected_without_echo(application, kind, fields):
    app, *_ = application
    result = send(app, kind, **fields)
    assert not result["ok"] and result["error"]["code"] == "invalid_parameters"
    assert app.runner.llm_client.model == "main" and not app.models.store.read().connections


def test_jsonl_loop_defaults_resume_and_existing_context_close(application):
    app, store, config, workspace = application
    app.close()
    def run(messages, request):
        output = io.StringIO()
        code = run_jsonl_runtime(workspace_path=workspace, session_request=request, session_store=store,
            llm_config=config, mcp_config=MCPConfig(), input_stream=io.StringIO("".join(
                json.dumps({"version": 1, "request_id": str(index), **message}) + "\n"
                for index, message in enumerate(messages))), output_stream=output, error_stream=io.StringIO())
        assert code == 0 and KEY not in output.getvalue() and HEADER not in output.getvalue()
        return [json.loads(line) for line in output.getvalue().splitlines()]
    payloads = run([
        {"type": "connect", "action": "add", "id": "other", "base_url": "https://other.example/v1", "api_key": KEY},
        {"type": "model", "action": "set", "role": "main", "model": "saved", "connection": "other", "window": 64000},
        {"type": "model", "action": "set", "role": "subagent", "model": "worker", "window": 32000},
        {"type": "model", "action": "default"}, {"type": "context_status"}, {"type": "compact"}, {"type": "close"}],
        SessionStartRequest(mode="new"))
    assert all(p["ok"] for p in payloads if p["type"] in {"model_result", "connect_result"})
    assert next(p for p in payloads if p["type"] == "context_status")["context_window_tokens"] == 64000
    assert any(p["type"] == "compact_result" for p in payloads)
    identifier = next(p["session_id"] for p in payloads if p["type"] == "runtime_ready")
    for request in (SessionStartRequest(mode="new"), SessionStartRequest(mode="resume", session_id=identifier)):
        restored = run([{"type": "model"}, {"type": "close"}], request)
        roles = next(p["roles"] for p in restored if p["type"] == "model_result")
        assert roles["main"]["model"] == "saved" and roles["main"]["window"] == 64000
        assert roles["subagent"]["model"] == "worker" and roles["subagent"]["window"] == 32000
        assert restored[-1]["type"] == "runtime_closed"
    cleared = run([{"type": "model", "action": "clear_default"}, {"type": "close"}], SessionStartRequest(mode="new"))
    assert next(p for p in cleared if p["type"] == "model_result")["ok"]
    assert app.models.store.read().default_models is None


@pytest.mark.parametrize("empty", [False, True])
def test_discovery_failure_or_empty_list_keeps_manual_selection_available(application, monkeypatch, empty):
    app, *_ = application
    def transport(request):
        if not empty:
            raise RuntimeError(KEY + HEADER)
        return httpx.Response(200, json={"data": []})
    monkeypatch.setattr("mycode.providers.openai_compatible.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    result = send(app, "connect", action="models", id="env")
    assert result["ok"] and result["models"] == []
    assert result["available"] is empty
    assert result["warning"] == (None if empty else "model_list_unavailable")
    assert send(app, action="set", role="main", model="manual")["ok"]


@pytest.mark.parametrize("kind", ["model", "connect"])
@pytest.mark.parametrize("handshake", ["permission", "mcp_trust"])
def test_handshake_rejects_controls_without_consuming_confirmation(monkeypatch, kind, handshake):
    from types import SimpleNamespace
    from mycode.adapters.jsonl import JsonlConfirmer, JsonlMCPTrustConfirmer
    from mycode.mcp.trust import MCPTrustRequest
    from mycode.permissions import ConfirmationRequest, PermissionRequest, PermissionDecision
    monkeypatch.setattr("mycode.adapters.jsonl.uuid4", lambda: SimpleNamespace(hex="confirmation"))
    response = {"version": 1, "type": handshake + "_response", "request_id": "confirmation"}
    response.update({"decision": "once"} if handshake == "permission" else {"approved": True})
    control = {"version": 1, "type": kind, "request_id": "control", "api_key": KEY, "headers": {"X": HEADER}}
    output = io.StringIO()
    channel = JsonlChannel(io.StringIO(json.dumps(control) + "\n" + json.dumps(response) + "\n"), output)
    if handshake == "permission":
        result = JsonlConfirmer(channel).confirm(ConfirmationRequest(
            permission_request=PermissionRequest(tool_name="write_file", capability="write", action="write file"),
            permission_decision=PermissionDecision.ask(), prompt="allow?"))
        assert result.status == "approved"
    else:
        assert JsonlMCPTrustConfirmer(channel).confirm(MCPTrustRequest(servers=())) is True
    payloads = [json.loads(line) for line in output.getvalue().splitlines()]
    assert payloads[1]["type"] == kind + "_result" and not payloads[1]["ok"]
    assert payloads[1]["error"]["code"] == "runtime_busy"
    assert KEY not in output.getvalue() and HEADER not in output.getvalue()


def test_invalid_control_is_correlated_and_loop_continues(application):
    from mycode.adapters.jsonl import _run_message_loop
    app, *_ = application
    output = io.StringIO()
    channel = JsonlChannel(io.StringIO(
        '{"version":1,"type":"model","request_id":42}\n'
        '{"version":1,"type":"model","request_id":"valid"}\n'
        '{"version":1,"type":"close"}\n'), output)
    assert _run_message_loop(channel, app) == 0
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0] == {"version": 1, "type": "model_result", "request_id": None, "ok": False,
        "error": {"code": "invalid_parameters", "message": "Invalid fields or parameter types for this action."}}
    assert rows[1]["ok"] and rows[1]["request_id"] == "valid"
    assert rows[-1]["type"] == "runtime_closed"


def test_business_error_explains_rejection_without_echoing_parameters(application):
    app, *_ = application
    result = send(app, action="set", role="main", model="unknown", connection="missing")
    assert not result["ok"] and result["error"]["code"] == "configuration_rejected"
    assert result["error"]["message"] == "Saved connection is unavailable; choose an existing connection."


def test_connect_list_default_matches_new_session_main(application):
    app, store, config, workspace = application
    add(app)
    assert send(app, action="set", role="main", model="saved", connection="other")["ok"]
    assert send(app, action="default")["ok"]

    def check(expected):
        assert send(app, "connect", action="list")["default_connection"] == expected
        fresh = start_agent_application_session(store, ProjectIdentity.from_workspace(workspace),
            request=SessionStartRequest(mode="new"), llm_config=config, mcp_config=MCPConfig())
        try:
            assert fresh.models.selection.identity("main")[0] == expected
        finally:
            fresh.close()

    check("other")  # Full role defaults take priority even without default_ref.
    add(app, "fallback")
    assert send(app, "connect", action="default", id="fallback")["ok"]
    check("other")  # A separate connection default does not override role defaults.
    assert send(app, action="clear_default")["ok"]
    check("fallback")
    assert send(app, "connect", action="default", id="env")["ok"]
    check("env")
    assert app.models.selection.identity("main")[0] == "other"  # Query is about new Sessions.
