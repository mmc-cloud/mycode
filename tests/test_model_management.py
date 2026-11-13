"""Real persistence/SDK boundaries for saved connections and in-place switching."""
import json
from pathlib import Path
import threading
import subprocess
import sys

import httpx
from openai import OpenAI
import pytest

from mycode.application.agent_session import start_agent_application_session
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig
from mycode.persistence.connection_store import ConnectionStore, ConnectionStoreError, SavedConnection
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity


@pytest.fixture
def managed(tmp_path, monkeypatch):
    # Runtime config stores and Session artifacts stay out of the user's profile.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = LLMConfig(api_key="synthetic-env-key", base_url="https://env.example/v1", model="main",
                       thinking_enabled=True, reasoning_effort="max")
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(workspace)
    session = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                               llm_config=config, mcp_config=MCPConfig())
    yield session, store, project, config
    session.close()


def add(manager, name="other", url="https://other.example/v1"):
    return manager.command("connect", ("add", name), values={
        "base_url": url, "api_key": f"synthetic-{name}-secret", "headers": '{"X-Private": "synthetic-header"}',
    })


def test_connection_store_crud_redaction_and_busy_reference(managed):
    session, *_ = managed
    manager = session.models
    add(manager)
    assert "synthetic" not in " ".join(manager.command("connect", ()))
    assert "synthetic" not in repr(manager.store.read())
    manager.command("model", ("subagent", "worker", "connection=other", "thinking=on", "effort=high"))
    for operation in (lambda: manager.command("connect", ("delete", "other")),
                      lambda: manager.command("connect", ("edit", "other"), values={"api_key": "replacement"})):
        with pytest.raises(ConnectionStoreError, match="active Session"):
            operation()
    manager.command("model", ("subagent", "follow"))
    manager.command("connect", ("edit", "other"), values={"api_key": "new-secret", "headers": "{}"})
    assert manager.store.read().connections["other"].headers == {}
    manager.command("connect", ("delete", "other"))
    assert not manager.store.read().connections


def test_default_selection_changes_future_sessions_only(managed):
    session, store, project, config = managed
    add(session.models)
    session.models.command("connect", ("default", "other"))
    assert session.runner.llm_client.config.base_url == config.base_url
    new = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                          llm_config=config, mcp_config=MCPConfig())
    try:
        assert new.runner.llm_client.config.base_url == "https://other.example/v1"
        assert new.runner.llm_client.thinking_enabled is None
    finally:
        new.close()


@pytest.mark.parametrize("thinking,effort", [("auto", None), ("on", "low")])
def test_saved_role_defaults_bootstrap_without_env_and_keep_sessions_independent(managed, monkeypatch, thinking, effort):
    from mycode.application.startup import create_application_environment, resolve_application_llm_config
    from mycode.config import MissingLLMConfigError
    session, store, project, config = managed
    add(session.models)
    options = (f"thinking={thinking}",) + ((f"effort={effort}",) if effort else ())
    session.models.command("model", ("main", "saved-main", "connection=other", "window=64000", *options))
    session.models.command("model", ("subagent", "worker", "connection=follow"))
    session.models.command("model", ("default",))
    monkeypatch.setattr("mycode.application.startup.load_llm_config",
                        lambda **kwargs: (_ for _ in ()).throw(MissingLLMConfigError("missing")))
    environment = create_application_environment(project.workspace_root, session_store=store)
    resolved = resolve_application_llm_config(environment)
    assert resolved.api_key == "synthetic-other-secret"
    assert resolved.model == "saved-main"
    assert resolved.thinking_enabled is (True if thinking == "on" else None)
    assert resolved.reasoning_effort == effort
    new = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                          llm_config=resolved, mcp_config=MCPConfig())
    try:
        assert new.models.selection == session.models.selection
        assert new.runner.context_budget.context_window_tokens == 64000
        assert new.runner is not session.runner
        assert new.runner.conversation is not session.runner.conversation
        sub = new.runner.tool_registry.require("delegate_task").runtime.llm_client_factory()
        try:
            assert sub.model == "worker"
        finally:
            sub._client.close()
    finally:
        new.close()
    session.models.command("model", ("default", "clear"))


def test_defaults_with_env_reference_do_not_reuse_other_credentials_when_env_missing(managed, monkeypatch):
    from mycode.application.startup import create_application_environment, resolve_application_llm_config
    from mycode.config import MissingLLMConfigError
    session, store, project, config = managed
    add(session.models)
    session.models.command("model", ("subagent", "worker", "connection=other"))
    session.models.command("model", ("default",))
    monkeypatch.setattr("mycode.application.startup.load_llm_config",
                        lambda **kwargs: (_ for _ in ()).throw(MissingLLMConfigError("missing")))
    with pytest.raises(MissingLLMConfigError):
        resolve_application_llm_config(create_application_environment(project.workspace_root, session_store=store))


@pytest.mark.parametrize("env_role", ["main", "subagent", "compact"])
@pytest.mark.parametrize("mode", ["resume", "continue"])
def test_envless_bootstrap_never_redirects_old_session_env_refs(managed, monkeypatch, env_role, mode):
    from mycode.application.startup import create_application_environment, prepare_application_session_factory
    from mycode.config import MissingLLMConfigError
    session, store, project, config = managed
    add(session.models)
    if env_role != "main":
        session.models.command("model", ("main", "saved-main", "connection=other"))
        session.models.command("model", (env_role, "summary", "connection=env"))
    else:
        session.models.command("model", ("main", "legacy-main"))
    identifier = session.active_project_session.record.id
    snapshot = session.active_project_session.writer.layout.root / "model-selection.json"
    before = snapshot.read_bytes()
    from mycode.model_selection import ModelSelection, RoleSelection
    defaults = ModelSelection(main=RoleSelection(connection_ref="other", model="default"))
    session.models.store.set_model_defaults(defaults)
    session.close()
    monkeypatch.setattr("mycode.application.startup.load_llm_config",
                        lambda **kwargs: (_ for _ in ()).throw(MissingLLMConfigError("missing")))
    factory = prepare_application_session_factory(
        create_application_environment(project.workspace_root, session_store=store), mcp_config=MCPConfig())
    assert factory.llm_config.env_source is False
    assert factory.llm_config.api_key == "synthetic-other-secret"
    request = SessionStartRequest(mode=mode, session_id=identifier if mode == "resume" else None)
    with pytest.raises(ValueError, match="unavailable"):
        factory.open_session(request)
    assert snapshot.read_bytes() == before
    assert store.get_session(project, identifier).status != "active"
    fresh = factory.open_session(SessionStartRequest(mode="new"))
    try:
        assert fresh.runner.llm_client.config.api_key == "synthetic-other-secret"
        with pytest.raises(ValueError, match="unavailable"):
            fresh.models.command("connect", ("models", "env"))
    finally:
        fresh.close()


def test_injected_full_configuration_is_authoritative_and_management_matches(managed):
    from mycode.application.startup import create_application_environment, prepare_application_session_factory
    from mycode.model_config import APIConnectionConfig, ModelConfiguration
    from mycode.model_selection import ModelSelection, RoleSelection
    session, store, project, legacy = managed
    add(session.models)
    session.models.store.set_model_defaults(ModelSelection(
        main=RoleSelection(connection_ref="other", model="unrelated-default")))
    models = ModelConfiguration(connections={
        "main": APIConnectionConfig("injected-main-key", "https://injected.example/v1", {"X-Main": "main"}),
        "aux": APIConnectionConfig("injected-aux-key", "https://aux.example/v1", {"X-Aux": "aux"})},
        selection=ModelSelection(main=RoleSelection(connection_ref="main", model="injected-main", thinking="on", effort="max", max_output_tokens=1000),
            subagent=RoleSelection(connection_ref="aux", model="worker", thinking="on", effort="low", max_output_tokens=512,
                context_window_tokens=16000, reserved_output_tokens=2000, context_safety_margin_tokens=1000),
            compact=RoleSelection(model="summary", thinking="on", effort="high",
                context_window_tokens=32000, reserved_output_tokens=4000, context_safety_margin_tokens=2000)))
    factory = prepare_application_session_factory(
        create_application_environment(project.workspace_root, session_store=store),
        model_configuration=models, mcp_config=MCPConfig())
    injected = factory.open_session(SessionStartRequest(mode="new"))
    identifier = injected.active_project_session.record.id
    try:
        assert injected.runner.model_configuration.resolve("main").model == models.resolve("main").model
        initial_snapshot = injected.active_project_session.writer.layout.root / "model-selection.json"
        assert "@injected:" in initial_snapshot.read_text(encoding="utf-8")
        assert "injected-main-key" not in initial_snapshot.read_text(encoding="utf-8")
        assert injected.models.selection.subagent.model == "worker"
        assert injected.models.selection.compact.thinking == "on"
        assert "unrelated-default" not in str(injected.models.status())
        with pytest.raises(ValueError, match="unavailable"):
            injected.models.command("connect", ("models", "env"))
        def runtime_configs():
            sub = injected.runner.tool_registry.require("delegate_task").runtime.llm_client_factory()
            try:
                return {"main": injected.runner.llm_client.config, "subagent": sub.config,
                        "compact": injected.runner.compactor.llm_client.config}
            finally:
                sub._client.close()
        original = runtime_configs()
        injected.models.switch(injected.models.selection)
        current = runtime_configs()
        for role in original:
            for name in ("api_key", "base_url", "headers", "model", "thinking_enabled", "reasoning_effort",
                         "context_window_tokens", "reserved_output_tokens", "context_safety_margin_tokens", "max_output_tokens"):
                assert getattr(current[role], name) == getattr(original[role], name)
        injected.models.command("model", ("main", "new-main"))
        assert runtime_configs()["subagent"].api_key == "injected-aux-key"
        snapshot = injected.active_project_session.writer.layout.root / "model-selection.json"
        assert "injected-aux-key" not in snapshot.read_text(encoding="utf-8")
        assert "@injected:" in snapshot.read_text(encoding="utf-8")
    finally:
        injected.close()
    # Transient references cannot accidentally bind to some other saved record.
    with pytest.raises(ValueError, match="unavailable"):
        start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
                                        llm_config=legacy, mcp_config=MCPConfig())
    resumed = factory.open_session(SessionStartRequest(mode="resume", session_id=identifier))
    try:
        assert resumed.runner.model_configuration.resolve("main").model == models.resolve("main").model
        assert resumed.runner.llm_client.model == "injected-main"
        assert resumed.models.selection.compact.thinking == "on"
    finally:
        resumed.close()


def test_injected_explicit_none_overrides_survive_main_only_edit(managed):
    from mycode.model_config import APIConnectionConfig, ModelConfiguration
    session, store, project, legacy = managed
    from mycode.model_selection import ModelSelection, RoleSelection
    models = ModelConfiguration(connections={"main": APIConnectionConfig("injected-key", "https://injected.example/v1")},
        selection=ModelSelection(main=RoleSelection(connection_ref="main", model="same", thinking="on", effort="max", max_output_tokens=1000),
            subagent=RoleSelection(model="same", effort="auto", max_output_tokens="auto"),
            compact=RoleSelection(thinking="on")))
    injected = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                             model_configuration=models, mcp_config=MCPConfig())
    try:
        injected.models.command("model", ("main", "same", "window=64000"))
        sub = injected.runner.tool_registry.require("delegate_task").runtime.llm_client_factory()
        try:
            assert sub.reasoning_effort is None
            assert sub.config.max_output_tokens is None
            assert injected.runner.tool_registry.require("delegate_task").runtime.context_budget.context_window_tokens == 64000
        finally:
            sub._client.close()
        assert injected.runner.compactor.llm_client.thinking_enabled is True
        assert injected.runner.compactor.llm_client.reasoning_effort == "max"
    finally:
        injected.close()

    identifier = injected.active_project_session.record.id
    with pytest.raises(ValueError, match="unavailable"):
        start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
                                        llm_config=legacy, mcp_config=MCPConfig())


def test_three_modes_follow_and_fixed_across_main_switches(managed):
    session, *_ = managed
    manager, runner = session.models, session.runner
    identities = (runner, runner.conversation, runner.tool_registry, session.mcp_manager)
    add(manager)
    manager.command("model", ("subagent", "worker", "connection=other", "thinking=on", "effort=high"))
    manager.command("model", ("compact", "summary", "connection=follow"))
    manager.command("model", ("main", "new-main", "thinking=on"))
    assert runner.llm_client.model == "new-main"
    assert runner.llm_client.reasoning_effort is None  # new On does not force high
    runtime = runner.tool_registry.require("delegate_task").runtime
    sub = runtime.llm_client_factory()
    try:
        assert sub.model == "worker"
        assert sub.config.api_key == "synthetic-other-secret"
        assert sub.config.headers == {"X-Private": "synthetic-header"}
        assert sub.reasoning_effort == "high"
    finally:
        sub._client.close()
    assert runner.compactor.llm_client.model == "summary"
    assert runner.compactor.llm_client.config.base_url == "https://env.example/v1"
    assert runner.compactor.llm_client.thinking_enabled is False
    assert identities == (runner, runner.conversation, runner.tool_registry, session.mcp_manager)
    manager.command("model", ("subagent", "follow"))
    manager.command("model", ("main", "third"))
    sub = runner.tool_registry.require("delegate_task").runtime.llm_client_factory()
    try:
        assert sub.model == "third"
        assert sub.config.base_url == runner.llm_client.config.base_url
    finally:
        sub._client.close()


def test_independent_budgets_and_calibration_reset(managed):
    session, *_ = managed
    manager, runner = session.models, session.runner
    old_main_estimator = runner.token_estimator
    old_summary_estimator = runner.compactor.summary_token_estimator
    manager.command("model", ("subagent", "worker", "window=16000", "reserve=2000", "margin=1000"))
    runtime = runner.tool_registry.require("delegate_task").runtime
    assert runtime.context_budget.max_input_tokens == 13000
    assert runner.context_budget.context_window_tokens == 128000
    assert runner.token_estimator is old_main_estimator
    manager.command("model", ("compact", "summary", "window=32000", "reserve=4000", "margin=2000"))
    assert runner.compactor.summary_budget.max_input_tokens == 26000
    assert runner.compactor.summary_token_estimator is not old_summary_estimator


def test_switch_resume_persists_references_without_credentials(managed):
    session, store, project, config = managed
    add(session.models)
    session.models.command("model", ("main", "new", "connection=other", "thinking=auto"))
    session.models.command("model", ("compact", "summary", "connection=env", "thinking=off"))
    path = session.active_project_session.writer.layout.root / "model-selection.json"
    data = path.read_text(encoding="utf-8")
    for forbidden in ("synthetic", "api_key", "base_url", "headers"):
        assert forbidden not in data
    identifier = session.active_project_session.record.id
    session.close()
    restored = start_agent_application_session(store, project,
        request=SessionStartRequest(mode="resume", session_id=identifier), llm_config=config, mcp_config=MCPConfig())
    try:
        assert restored.runner.llm_client.model == "new"
        assert restored.runner.llm_client.config.api_key == "synthetic-other-secret"
        assert restored.runner.compactor.llm_client.config.api_key == config.api_key
        assert restored.runner.compactor.llm_client.model == "summary"
    finally:
        restored.close()


@pytest.mark.parametrize("failure", ["client", "persist", "budget", "reference"])
def test_failed_switch_preserves_runner_and_saved_selection(managed, monkeypatch, failure):
    session, *_ = managed
    manager, runner = session.models, session.runner
    manager.command("model", ("main", "saved"))
    old = (runner.llm_client, runner.context_budget, runner.compactor.llm_client,
           runner.tool_registry.require("delegate_task").runtime, manager.selection)
    path = session.active_project_session.writer.layout.root / "model-selection.json"
    original = path.read_bytes()
    if failure == "client":
        monkeypatch.setattr("mycode.providers.factory.OpenAICompatibleLLMClient",
                            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("client failed")))
        args = ("main", "bad")
    elif failure == "persist":
        monkeypatch.setattr(session.active_project_session.writer, "save_model_selection",
                            lambda selection: (_ for _ in ()).throw(OSError("disk failure")))
        args = ("main", "bad")
    elif failure == "budget":
        runner.conversation.add_user_message("keep this history " * 400)
        args = ("main", "small", "window=10", "reserve=1", "margin=1")
    else:
        args = ("main", "bad", "connection=missing")
    with pytest.raises((RuntimeError, OSError, ValueError)):
        manager.command("model", args)
    assert old == (runner.llm_client, runner.context_budget, runner.compactor.llm_client,
                   runner.tool_registry.require("delegate_task").runtime, manager.selection)
    assert path.read_bytes() == original


def test_busy_switch_and_writes_are_rejected_across_threads(managed):
    session, *_ = managed
    entered, finish = threading.Event(), threading.Event()
    def request():
        with session.models.activity.operation():
            entered.set()
            finish.wait(5)
    worker = threading.Thread(target=request)
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="active"):
            session.models.command("model", ("main", "new"))
        with pytest.raises(RuntimeError, match="active"):
            add(session.models)
    finally:
        finish.set()
        worker.join(3)


def test_models_listing_fallback_and_known_effort_validation(managed, monkeypatch):
    session, *_ = managed
    def failed(self):
        raise RuntimeError("private error body")
    monkeypatch.setattr("openai.resources.models.Models.list", failed)
    lines = session.models.command("connect", ("models", "env"))
    assert "unverified" in lines[0]
    assert "private" not in str(lines)
    session.models.catalogs["env"] = {"known": {"effort": {"supported_levels": ["high"]}}}
    with pytest.raises(ConnectionStoreError, match="reasoning"):
        session.models.command("model", ("main", "known", "thinking=on", "effort=max"))


def test_model_listing_is_not_a_whitelist_for_manual_ids(managed, monkeypatch):
    session, *_ = managed
    _model_list_transport(monkeypatch, "main", context_window=64000)
    session.models.command("connect", ("models", "env"))
    session.models.command("model", ("main", "manual-unlisted", "thinking=on", "effort=low"))
    assert session.runner.llm_client.model == "manual-unlisted"
    assert session.models.windows["main"].tokens == 128000
    assert session.models.windows["main"].server_limit is None


@pytest.mark.parametrize("kind", [RuntimeError, ValueError, OSError])
def test_transaction_does_not_reclassify_operation_errors(managed, kind):
    session, *_ = managed
    error = kind("operation failed")
    with pytest.raises(kind) as caught:
        with session.models.store.transaction():
            raise error
    assert caught.value is error


def test_invalid_store_is_not_overwritten_and_errors_hide_secrets(tmp_path):
    store = ConnectionStore(tmp_path)
    store.path.write_text('{"api_key":"private"', encoding="utf-8")
    old = store.path.read_bytes()
    with pytest.raises(ConnectionStoreError) as error:
        store.read()
    assert "private" not in str(error.value)
    assert store.path.read_bytes() == old


def test_secure_snapshot_write_failure_preserves_file_and_removes_temp(tmp_path, monkeypatch):
    store = ConnectionStore(tmp_path)
    saved = SavedConnection(id="test", base_url="https://example.com", api_key="synthetic-private")
    store.save(saved)
    before = store.path.read_bytes()
    monkeypatch.setattr("mycode.persistence.connection_store.os.replace",
                        lambda *args: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(ConnectionStoreError) as error:
        store.set_default("test")
    assert "synthetic-private" not in str(error.value)
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob(".connections-*.tmp"))


def test_closed_session_model_management_is_rejected(managed):
    session, *_ = managed
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.models.command("connect", ())


@pytest.mark.parametrize("options", [("effort=high", "thinking=off"), ("thinking=off", "effort=high")])
def test_invalid_thinking_effort_is_independent_of_option_order(managed, options):
    session, *_ = managed
    with pytest.raises(ConnectionStoreError):
        session.models.command("model", ("main", "model", *options))


def test_cli_connection_and_model_commands_do_not_enter_agent(monkeypatch, tmp_path):
    from mycode.cli import run_agent_command
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    secrets = iter(["synthetic-cli-secret", "{}"])
    monkeypatch.setattr("mycode.cli.getpass.getpass", lambda prompt: next(secrets))
    entries = iter(["/connect add another", "https://another.example/v1", "/connect",
                    "/model main chosen connection=another thinking=auto", "/model", "/exit"])
    outputs = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(tmp_path / "projects")
    config = LLMConfig(api_key="test", base_url="https://env.example/v1", model="main")
    run_agent_command(workspace_path=workspace, input_func=lambda prompt: next(entries),
        output_func=outputs.append, llm_config=config, mcp_config=MCPConfig(),
        session_store=store, session_request=SessionStartRequest(mode="new"))
    text = "\n".join(outputs)
    assert "synthetic-cli-secret" not in text
    assert "Connection saved" in text and "model=chosen" in text
    project = ProjectIdentity.from_workspace(workspace)
    record = store.list_sessions(project)[0]
    assert not store.load_conversation(project, record.id).get_messages()


def test_real_sdk_uses_complete_role_connections_and_auto_on_off(managed, monkeypatch):
    session, *_ = managed
    requests = []
    def transport(request):
        payload = json.loads(request.content)
        requests.append((request.url.host, dict(request.headers), payload))
        return httpx.Response(200, json={"id": "completion", "object": "chat.completion", "created": 0,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}]})
    monkeypatch.setattr("mycode.providers.openai_compatible.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    manager = session.models
    add(manager)
    manager.command("model", ("subagent", "worker", "connection=other", "thinking=on", "effort=low"))
    from mycode.conversation import Conversation
    session.runner.llm_client.complete(Conversation())
    session.runner.compactor.llm_client.complete(Conversation())
    sub = session.runner.tool_registry.require("delegate_task").runtime.llm_client_factory()
    try:
        sub.complete(Conversation())
    finally:
        sub._client.close()
    assert [row[0] for row in requests] == ["env.example", "env.example", "other.example"]
    assert requests[0][1]["authorization"] == "Bearer synthetic-env-key"
    assert "x-private" not in requests[0][1]
    assert requests[2][1]["authorization"] == "Bearer synthetic-other-secret"
    assert requests[2][1]["x-private"] == "synthetic-header"
    assert requests[2][2]["reasoning_effort"] == "low"
    for mode in ("auto", "on", "off"):
        manager.command("model", ("main", "new", f"thinking={mode}"))
        session.runner.llm_client.complete(Conversation())
        payload = requests[-1][2]
        assert "reasoning_effort" not in payload
        if mode == "auto":
            assert "thinking" not in payload
        else:
            assert payload["thinking"]["type"] == ("enabled" if mode == "on" else "disabled")


def test_busy_actual_stream_blocks_switch_without_interrupting_request(managed, monkeypatch):
    session, *_ = managed
    entered, finish = threading.Event(), threading.Event()
    def transport(request):
        entered.set()
        assert finish.wait(5)
        chunk = {"id": "chunk", "object": "chat.completion.chunk", "created": 0, "model": "test",
                 "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
    monkeypatch.setattr("mycode.providers.openai_compatible.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    session.models.command("model", ("main", "stream-model", "thinking=auto"))
    outcomes, errors = [], []
    def request():
        try:
            outcomes.append(session.run_turn("say done"))
        except Exception as error:
            errors.append(type(error).__name__)
    worker = threading.Thread(target=request)
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="active"):
            session.models.command("model", ("main", "other"))
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive() and not errors and outcomes
    assert session.runner.llm_client.model == "stream-model"


def test_reference_lease_prevents_deletion_from_another_process(managed):
    session, *_ = managed
    add(session.models)
    session.models.command("model", ("subagent", "worker", "connection=other"))
    code = """import sys
from pathlib import Path
from mycode.persistence.connection_store import ConnectionStore, ConnectionStoreError
try:
    ConnectionStore(Path(sys.argv[1])).delete('other')
except ConnectionStoreError:
    print('blocked')
else:
    raise SystemExit(2)
"""
    result = subprocess.run([sys.executable, "-c", code, str(session.models.store.root)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "blocked"


def test_missing_connection_on_resume_fails_without_fallback(managed):
    session, store, project, config = managed
    add(session.models)
    session.models.command("model", ("subagent", "worker", "connection=other"))
    identifier = session.active_project_session.record.id
    session.close()
    ConnectionStore().delete("other")
    with pytest.raises(ValueError, match="unavailable"):
        start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
                                        llm_config=config, mcp_config=MCPConfig())


def test_windows_are_saved_per_connection_and_model_and_reused(managed):
    session, *_ = managed
    manager, runner = session.models, session.runner
    add(manager)
    manager.command("model", ("main", "main", "window=64000"))
    assert all(window.tokens == 64000 for window in manager.windows.values())
    assert "source=user (unverified)" in manager.status()[0]
    manager.command("model", ("main", "different"))
    assert runner.context_budget.context_window_tokens == 128000
    assert "source=default (unverified)" in manager.status()[0]
    manager.command("model", ("main", "main", "connection=other"))
    assert runner.context_budget.context_window_tokens == 128000
    manager.command("model", ("main", "main", "window=48000"))
    manager.command("model", ("main", "main", "connection=env"))
    assert runner.context_budget.context_window_tokens == 64000
    manager.command("model", ("main", "main", "connection=other"))
    assert runner.context_budget.context_window_tokens == 48000
    manager.command("model", ("main", "main", "window=inherit"))
    assert all(window.tokens == 128000 for window in manager.windows.values())
    assert "other" not in manager.store.read().model_windows


def test_default_connection_and_new_models_do_not_borrow_legacy_window(managed):
    from dataclasses import replace
    session, store, project, config = managed
    add(session.models)
    session.models.command("connect", ("default", "other"))
    fresh = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
        llm_config=replace(config, context_window_tokens=16000,
                           compact_model="legacy-summary", subagent_model="legacy-worker"), mcp_config=MCPConfig())
    try:
        assert fresh.runner.context_budget.context_window_tokens == 128000
        assert all(window.tokens == 128000 for window in fresh.models.windows.values())
        fresh.models.command("model", ("main", "new-model", "reserve=32000"))
        assert fresh.runner.context_budget.context_window_tokens == 128000
        assert fresh.runner.context_budget.reserved_output_tokens == 32000
    finally:
        fresh.close()


def test_window_resolution_keeps_three_role_budgets_independent(managed):
    session, *_ = managed
    manager, runner = session.models, session.runner
    add(manager)
    manager.command("model", ("main", "main", "window=64000"))
    manager.command("model", ("subagent", "main", "connection=other", "window=32000", "reserve=2000", "margin=1000"))
    manager.command("model", ("compact", "summary", "window=24000", "reserve=4000", "margin=2000"))
    assert runner.context_budget.context_window_tokens == 64000
    assert runner.tool_registry.require("delegate_task").runtime.context_budget.max_input_tokens == 29000
    assert runner.compactor.summary_budget.max_input_tokens == 18000
    assert runner.compactor.llm_client.thinking_enabled is False
    manager.command("model", ("main", "new"))
    assert runner.context_budget.context_window_tokens == 128000
    assert manager.windows["subagent"].tokens == 32000
    assert manager.windows["compact"].tokens == 24000
    # Fixed model + following connection changes identity with Main's connection.
    manager.command("model", ("main", "new", "connection=other"))
    assert manager.windows["compact"].tokens == 128000
    manager.command("model", ("main", "new", "connection=env"))
    assert manager.windows["compact"].tokens == 24000


def _model_list_transport(monkeypatch, model, **metadata):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "test", **metadata}]})
    monkeypatch.setattr("mycode.providers.openai_compatible.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    return requests


def test_api_window_is_applied_and_caps_user_preferences(managed, monkeypatch):
    session, *_ = managed
    manager = session.models
    requests = _model_list_transport(monkeypatch, "main", context_window=96000)
    lines = manager.command("connect", ("models", "env"))
    assert len(requests) == 1 and requests[0].url.path == "/v1/models"
    assert "API context window=96000" in lines[0]
    assert all(window.tokens == 96000 and window.source == "API" for window in manager.windows.values())
    manager.command("model", ("main", "main", "window=64000"))
    assert session.runner.context_budget.context_window_tokens == 64000
    old_client, before = session.runner.llm_client, manager.store.path.read_bytes()
    with pytest.raises(ConnectionStoreError, match="server limit"):
        manager.command("model", ("main", "main", "window=100000"))
    assert session.runner.llm_client is old_client and manager.store.path.read_bytes() == before
    _model_list_transport(monkeypatch, "main", context_window=48000)
    manager.command("connect", ("models", "env"))
    assert session.runner.context_budget.context_window_tokens == 48000
    assert manager.windows["main"].requested == 64000  # Shared preference, API-capped at runtime.
    assert manager.selection.main.context_window_tokens is None  # No mirror in the Session snapshot.
    assert "API limit=48000" in manager.status()[0]
    def failed(self):
        raise RuntimeError("private provider response")
    monkeypatch.setattr("openai.resources.models.Models.list", failed)
    assert "unavailable" in manager.command("connect", ("models", "env"))[0]
    manager.command("model", ("main", "main", "reserve=4000"))
    assert session.runner.context_budget.context_window_tokens == 48000
    assert manager.store.read().model_windows  # Ordinary preference writes did not invalidate API limits.


@pytest.mark.parametrize("value", [None, True, "64000", 0, -1])
def test_missing_or_invalid_api_limit_does_not_verify_default(managed, monkeypatch, value):
    session, *_ = managed
    _model_list_transport(monkeypatch, "main", context_window=value)
    lines = session.models.command("connect", ("models", "env"))
    assert "unverified" in lines[0]
    assert session.models.windows["main"].tokens == 128000
    assert session.models.windows["main"].server_limit is None
    session.models.command("model", ("main", "main", "window=64000"))
    assert "source=user (unverified)" in session.models.status()[0]


def test_api_windows_do_not_cross_connections_or_model_ids(managed, monkeypatch):
    session, *_ = managed
    manager = session.models
    add(manager)
    _model_list_transport(monkeypatch, "main", context_window_tokens=32000)
    manager.command("connect", ("models", "other"))
    assert manager.windows["main"].tokens == 128000
    manager.command("model", ("main", "main", "connection=other"))
    assert manager.windows["main"].tokens == 32000
    assert manager.windows["main"].server_limit == 32000
    manager.command("model", ("main", "main", "connection=env"))
    assert manager.windows["main"].tokens == 128000


def test_resume_uses_current_saved_window_and_does_not_resurrect_cleared_preference(managed):
    session, store, project, config = managed
    session.models.command("model", ("main", "main", "window=64000"))
    identifier = session.active_project_session.record.id
    session.models.command("model", ("default",))
    session.close()
    fresh = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                            llm_config=config, mcp_config=MCPConfig())
    try:
        assert fresh.runner.context_budget.context_window_tokens == 64000
        fresh.models.command("model", ("main", "main", "window=80000"))
    finally:
        fresh.close()
    restored = start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
                                               llm_config=config, mcp_config=MCPConfig())
    try:
        assert restored.runner.context_budget.context_window_tokens == 80000
        restored.models.command("model", ("main", "main", "window=inherit"))
    finally:
        restored.close()
    # The old default snapshot also had 64K; clearing the shared setting must
    # not re-import that stale mirror into a future Session.
    fresh = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                            llm_config=config, mcp_config=MCPConfig())
    try:
        assert fresh.runner.context_budget.context_window_tokens == 128000
    finally:
        fresh.close()


@pytest.mark.parametrize("failure", ["window_write", "session_write", "history"])
def test_failed_window_change_preserves_preferences_session_and_runtime(managed, monkeypatch, failure):
    session, *_ = managed
    manager, runner = session.models, session.runner
    manager.command("model", ("main", "main", "window=64000"))
    original = manager.store.read().model_windows
    path = session.active_project_session.writer.layout.root / "model-selection.json"
    snapshot = path.read_bytes()
    old = runner.llm_client, runner.context_budget, manager.selection
    window = 32000
    if failure == "window_write":
        def fail(doc):
            raise OSError("write failed")
        monkeypatch.setattr(manager.store, "_write", fail)
    elif failure == "session_write":
        def fail(selection):
            raise OSError("write failed")
        monkeypatch.setattr(session.active_project_session.writer, "save_model_selection", fail)
    else:
        runner.conversation.add_user_message("preserve canonical history " * 8000)
        window = 16000
    canonical = runner.conversation.get_messages()
    with pytest.raises((ValueError, OSError)):
        manager.command("model", ("main", "main", f"window={window}"))
    assert manager.store.read().model_windows == original
    assert path.read_bytes() == snapshot
    assert old == (runner.llm_client, runner.context_budget, manager.selection)
    assert runner.conversation.get_messages() == canonical


def test_env_window_preferences_are_bound_to_actual_connection(managed):
    from dataclasses import replace
    session, store, project, config = managed
    session.models.command("model", ("main", "main", "window=64000"))
    session.close()
    changed = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
        llm_config=replace(config, base_url="https://different.example/v1"), mcp_config=MCPConfig())
    try:
        assert changed.runner.context_budget.context_window_tokens == 128000
    finally:
        changed.close()


@pytest.mark.parametrize("changed_field", ["base_url", "api_key", "headers"])
def test_bootstrap_windows_are_scoped_to_complete_connection(managed, changed_field):
    from dataclasses import replace
    session, store, project, config = managed
    first_config = replace(config, env_source=False)
    changed = {"base_url": "https://bootstrap-other.example/v1", "api_key": "synthetic-other",
               "headers": {"X-Connection": "other"}}[changed_field]
    second_config = replace(first_config, **{changed_field: changed})
    def open_session(configuration):
        return start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
            llm_config=configuration, mcp_config=MCPConfig())
    first = open_session(first_config)
    try:
        first.models.command("model", ("main", "main", "window=64000"))
    finally:
        first.close()
    second = open_session(second_config)
    try:
        assert all(window.tokens == 128000 for window in second.models.windows.values())
        second.models.command("model", ("main", "main", "window=32000"))
    finally:
        second.close()
    returned = open_session(first_config)
    try:
        assert all(window.tokens == 64000 for window in returned.models.windows.values())
    finally:
        returned.close()


@pytest.mark.parametrize("case", ["independent", "api_limit", "explicit_auto", "shared_api_limit"])
def test_explicit_windows_match_all_runtime_budgets_and_resume(managed, case):
    from mycode.model_config import APIConnectionConfig, ModelConfiguration
    from mycode.model_selection import ModelSelection, RoleSelection
    from mycode.model_windows import ModelWindow
    session, store, project, config = managed
    windows = {
        "independent": {"main": ModelWindow(64000), "subagent": ModelWindow(32000), "compact": ModelWindow(24000)},
        "api_limit": {"main": ModelWindow(None, 64000)},
        "explicit_auto": {"main": ModelWindow(64000), "subagent": ModelWindow(None)},
        "shared_api_limit": {"subagent": ModelWindow(None, 64000)},
    }[case]
    models = ModelConfiguration(
        selection=ModelSelection(main=RoleSelection(connection_ref="primary", model="shared", context_window_tokens=128000),
            subagent=RoleSelection(context_window_tokens=16000)),
        connections={"primary": APIConnectionConfig("synthetic", "https://explicit.example/v1")}, windows=windows)
    expected = {role: models.resolve(role).context_window_tokens for role in ("main", "subagent", "compact")}
    if case == "shared_api_limit":
        # A trustworthy limit belongs to the connection/model, even when only
        # one role supplied it. Initialization must update the other clients.
        expected = {role: min(tokens, 64000) for role, tokens in expected.items()}
    def check(application):
        runner = application.runner
        runtime = runner.tool_registry.require("delegate_task").runtime
        budgets = {"main": runner.context_budget, "subagent": runtime.context_budget, "compact": runner.compactor.summary_budget}
        sub = runtime.llm_client_factory()
        try:
            clients = {"main": runner.llm_client, "subagent": sub, "compact": runner.compactor.llm_client}
            for role, tokens in expected.items():
                assert application.models.windows[role].tokens == budgets[role].context_window_tokens == tokens
                assert clients[role].config.context_window_tokens == runner.model_configuration.resolve(role).context_window_tokens == tokens
            assert application.get_context_status().context_window_tokens == expected["main"]
        finally:
            sub._client.close()
    injected = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
        model_configuration=models, mcp_config=MCPConfig())
    identifier = injected.active_project_session.record.id
    try:
        check(injected)
        injected.models.switch(injected.models.selection)
        check(injected)
        if case == "explicit_auto":
            assert injected.models.windows["subagent"].source == "default"
        if case == "api_limit":
            assert injected.models.windows["main"].server_limit == 64000
    finally:
        injected.close()
    resumed = start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
        model_configuration=models, mcp_config=MCPConfig())
    try:
        check(resumed)
    finally:
        resumed.close()


def test_edit_connection_clears_shared_windows_on_resume(managed):
    session, store, project, config = managed
    add(session.models)
    session.models.command("model", ("main", "main", "connection=other", "window=64000"))
    identifier = session.active_project_session.record.id
    session.close()
    connections = ConnectionStore()
    connections.save(SavedConnection(id="other", base_url="https://replaced.example/v1", api_key="synthetic"), edit=True)
    assert "other" not in connections.read().model_windows
    restored = start_agent_application_session(store, project, request=SessionStartRequest(mode="resume", session_id=identifier),
                                               llm_config=config, mcp_config=MCPConfig())
    try:
        assert restored.runner.context_budget.context_window_tokens == 128000
    finally:
        restored.close()


def test_tighter_discovered_limit_blocks_only_affected_roles_until_budget_fixed(managed, monkeypatch):
    session, *_ = managed
    manager, runner = session.models, session.runner
    add(manager)
    manager.command("model", ("main", "main", "window=64000"))
    manager.command("model", ("compact", "summary", "connection=other", "window=32000", "reserve=2000", "margin=1000"))
    runner.conversation.add_user_message("retain this history " * 500)
    canonical = runner.conversation.get_messages()
    estimate = runner.inspect_context().estimate.estimated_input_tokens
    limit = estimate + 2000
    assert limit < 64000
    requests = _model_list_transport(monkeypatch, "main", context_window=limit)
    with pytest.raises(ValueError):
        manager.command("connect", ("models", "env"))
    assert runner.context_budget.context_window_tokens == 64000
    with pytest.raises(RuntimeError, match="API context limit changed"):
        session.run_turn("must not be appended or sent")
    assert runner.conversation.get_messages() == canonical
    with manager.activity.operation("compact"):
        pass  # A different summary connection is still available for recovery.
    manager.command("model", ("main", "main", f"window={limit}", "reserve=500", "margin=500"))
    assert runner.context_budget.context_window_tokens == limit
    with manager.activity.operation("main"):
        pass
    assert len(requests) == 1  # Only GET /models; no inference or long-prompt probe.


def test_window_changes_and_api_discovery_are_rejected_while_busy(managed, monkeypatch):
    session, *_ = managed
    requests = _model_list_transport(monkeypatch, "main", context_window=32000)
    with session.models.activity.operation("subagent"):
        with pytest.raises(RuntimeError, match="active"):
            session.models.command("model", ("main", "main", "window=64000"))
        with pytest.raises(RuntimeError, match="active"):
            session.models.command("connect", ("models", "env"))
    assert not requests


def test_edit_invalidates_local_windows_in_defaults_and_sessions(managed):
    from mycode.model_selection import ModelSelection, RoleSelection
    session, store, project, config = managed
    add(session.models)
    connections = session.models.store
    # Programmatic role defaults can contain a local window preference.
    connections.set_model_defaults(ModelSelection(
        main=RoleSelection(connection_ref="other", model="main", context_window_tokens=64000)))
    fresh = start_agent_application_session(store, project, request=SessionStartRequest(mode="new"),
                                            llm_config=config, mcp_config=MCPConfig())
    identifier = fresh.active_project_session.record.id
    try:
        assert fresh.runner.context_budget.context_window_tokens == 64000
    finally:
        fresh.close()
    connections.save(SavedConnection(id="other", base_url="https://new.example/v1", api_key="synthetic"), edit=True)
    for request in (SessionStartRequest(mode="new"), SessionStartRequest(mode="resume", session_id=identifier)):
        restored = start_agent_application_session(store, project, request=request, llm_config=config, mcp_config=MCPConfig())
        try:
            assert restored.runner.context_budget.context_window_tokens == 128000
        finally:
            restored.close()


@pytest.mark.parametrize("value", [True, "64000", 0, -1])
def test_invalid_persisted_window_does_not_overwrite_store(managed, value):
    session, *_ = managed
    add(session.models)
    path = session.models.store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_windows"] = {"other": {"main": value}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ConnectionStoreError, match="invalid"):
        session.models.command("model", ("main", "main", "window=64000"))
    assert path.read_bytes() == before


def test_shared_windows_are_owned_by_connection_store(managed):
    session, *_ = managed
    session.models.command("model", ("main", "main", "window=64000"))
    path = session.active_project_session.writer.layout.root / "model-selection.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert session.models.store.read().model_windows
    assert all(role["context_window_tokens"] is None
        for role in (snapshot["main"], snapshot["subagent"], snapshot["compact"]))


def test_context_window_source_and_percent_follow_api_and_model_switch(managed, monkeypatch):
    from mycode.model_events import TokenUsage
    session, *_ = managed
    _model_list_transport(monkeypatch, "main", context_window=1000000)
    session.models.command("connect", ("models", "env"))
    status = session.get_context_status()
    assert status.context_window_source == "API"
    assert status.context_window_tokens == status.context_window_server_limit == 1000000
    assert status.context_window_percent == pytest.approx(status.estimated_input_tokens / 1000000 * 100)
    session.models.command("model", ("main", "main", "window=64000"))
    session.runner.last_token_usage = TokenUsage(999999, 0, 999999)
    status = session.get_context_status()
    assert status.context_window_percent == pytest.approx(status.estimated_input_tokens / 64000 * 100)
    assert status.input_budget_percent > status.context_window_percent
    assert status.last_provider_prompt_tokens == 999999
    assert status.estimated_input_tokens != status.last_provider_prompt_tokens
    assert status.to_dict()["context_window_percent"] == status.context_window_percent
    add(session.models)
    session.models.command("model", ("main", "other-model", "connection=other"))
    status = session.get_context_status()
    assert status.context_window_tokens == 128000
    assert status.context_window_source == "default"
    assert status.last_provider_prompt_tokens is None
    assert status.estimate_source == "default"


def test_context_uses_compacted_projection_and_recovers_on_resume(managed):
    from types import SimpleNamespace
    from mycode.messages import Message
    session, store, project, config = managed
    runner = session.runner
    session.models.command("model", ("main", "main", "window=64000"))
    for number in range(8):
        runner.conversation.add_user_message(f"Task {number}: preserve the result. " * 30)
        runner.conversation.add_assistant_message(f"Task {number}: checked the evidence. " * 30)
    before = session.get_context_status()
    canonical = runner.conversation.get_messages()
    runner.compactor.llm_client._client.close()
    runner.compactor.llm_client = SimpleNamespace(complete=lambda prompt: Message(role="assistant", content=json.dumps({
        "objective": "continue", "progress": ["checked"], "decisions": [], "constraints": [],
        "open_items": ["next"], "references": []})))
    result = session.compact_context()
    assert result.status == "compacted"
    after = session.get_context_status()
    assert after.estimated_input_tokens < before.estimated_input_tokens
    assert after.context_window_percent < before.context_window_percent
    assert after.compact_status == "active"
    assert runner.conversation.get_messages() == canonical
    identifier = session.active_project_session.record.id
    session.close()
    restored = start_agent_application_session(store, project,
        request=SessionStartRequest(mode="resume", session_id=identifier), llm_config=config, mcp_config=MCPConfig())
    try:
        status = restored.get_context_status()
        assert status.context_window_tokens == 64000
        assert status.compact_status == "active"
        assert status.estimated_input_tokens == after.estimated_input_tokens
        assert status.last_provider_prompt_tokens is None
    finally:
        restored.close()
