from __future__ import annotations

from types import SimpleNamespace

import pytest

from mycode.application.startup import (
    ApplicationStartupWarning,
    create_application_environment,
    prepare_application_session_factory,
    resolve_application_llm_config,
)
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.conversation import Conversation
from mycode.mcp.config import MCPConfig, MCPConfigError, MCPLoadedConfig
from mycode.mcp.trust import MCPTrustResolution
from mycode.permissions import ConfirmationResult, PermissionDecision
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity
from mycode.tools.registry import ToolRegistry
from test_tools_registry import (
    AuthorizedDecisionTool,
    RecordingConfirmer,
    RecordingPermissionChecker,
)


def _llm_config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )


def _loaded_config() -> MCPLoadedConfig:
    empty = MCPConfig()
    return MCPLoadedConfig(
        user=empty,
        project=empty,
        merged=empty,
        project_unresolved=empty,
    )


def test_create_environment_builds_project_identity_and_reuses_store(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "projects")

    environment = create_application_environment(tmp_path, session_store=store)

    assert environment.workspace.root == tmp_path.resolve()
    assert environment.project == ProjectIdentity.from_workspace(tmp_path)
    assert environment.session_store is store


def test_environment_creation_does_not_load_runtime_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mycode.application.startup.load_llm_config",
        lambda **kwargs: pytest.fail("environment must not load LLM config"),
    )
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: pytest.fail("environment must not load MCP config"),
    )

    environment = create_application_environment(tmp_path)

    assert environment.project.workspace_root == tmp_path.resolve()


def test_factory_loads_llm_and_resolves_mcp_once(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def load_llm(**kwargs):
        calls.append(f"llm:{kwargs['workspace_root']}")
        return _llm_config()

    def load_mcp(**kwargs):
        calls.append(f"mcp:{kwargs['workspace_root']}")
        return _loaded_config()

    def resolve(loaded, project, *, confirmer, trust_file=None):
        calls.append("trust")
        return SimpleNamespace(config=MCPConfig())

    monkeypatch.setattr("mycode.application.startup.load_llm_config", load_llm)
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers", load_mcp
    )
    monkeypatch.setattr(
        "mycode.application.startup.resolve_project_mcp_trust", resolve
    )
    environment = create_application_environment(tmp_path)

    factory = prepare_application_session_factory(
        environment,
        mcp_trust_confirmer=object(),
    )

    assert factory.llm_config.model == "test-model"
    assert calls == [f"llm:{tmp_path.resolve()}", f"mcp:{tmp_path.resolve()}", "trust"]


def test_resolve_application_llm_config_reuses_explicit_or_loads_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    loaded = _llm_config()
    calls: list[object] = []

    def load_llm(**kwargs):
        calls.append(kwargs["workspace_root"])
        return loaded

    monkeypatch.setattr("mycode.application.startup.load_llm_config", load_llm)
    environment = create_application_environment(tmp_path)

    assert resolve_application_llm_config(environment, llm_config=loaded) is loaded
    assert resolve_application_llm_config(environment) is loaded
    assert calls == [tmp_path.resolve()]


def test_explicit_configs_skip_llm_and_mcp_resolution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mycode.application.startup.load_llm_config",
        lambda **kwargs: pytest.fail("explicit LLM config must be reused"),
    )
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: pytest.fail("explicit MCP config must be reused"),
    )
    monkeypatch.setattr(
        "mycode.application.startup.resolve_project_mcp_trust",
        lambda *args, **kwargs: pytest.fail("explicit MCP config must skip trust"),
    )
    environment = create_application_environment(tmp_path)
    mcp_config = MCPConfig()

    factory = prepare_application_session_factory(
        environment,
        llm_config=_llm_config(),
        mcp_config=mcp_config,
    )

    assert factory.resolved_mcp_config is mcp_config


def test_mcp_config_error_warns_and_uses_empty_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: (_ for _ in ()).throw(MCPConfigError("broken config")),
    )
    warnings: list[ApplicationStartupWarning] = []
    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        llm_config=_llm_config(),
        mcp_trust_confirmer=object(),
        warning_handler=warnings.append,
    )

    assert factory.resolved_mcp_config == MCPConfig()
    assert warnings == [
        ApplicationStartupWarning(code="mcp_config_error", message="broken config")
    ]


def test_factory_opens_independent_candidate_sessions_without_current_state(
    tmp_path,
    monkeypatch,
) -> None:
    opened: list[object] = []

    def open_session(*args, **kwargs):
        session = object()
        opened.append((session, args, kwargs))
        return session

    monkeypatch.setattr(
        "mycode.application.startup.start_agent_application_session",
        open_session,
    )
    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        llm_config=_llm_config(),
        mcp_config=MCPConfig(),
    )

    first = factory.open_session(SessionStartRequest(mode="new"))
    second = factory.open_session(SessionStartRequest(mode="new"))
    third = factory.open_session(SessionStartRequest(mode="continue"))

    assert first is not second
    assert first is not third
    assert second is not third
    assert len(opened) == 3
    assert not hasattr(factory, "current_session")
    assert opened[0][1][0] is factory.environment.session_store
    assert opened[0][1][1] is factory.environment.project


def test_factory_reuses_llm_mcp_snapshot_across_three_sessions(tmp_path, monkeypatch):
    calls: list[str] = []
    resolved = MCPConfig()

    monkeypatch.setattr(
        "mycode.application.startup.load_llm_config",
        lambda **kwargs: calls.append("llm") or _llm_config(),
    )
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: calls.append("mcp") or _loaded_config(),
    )
    monkeypatch.setattr(
        "mycode.application.startup.resolve_project_mcp_trust",
        lambda *args, **kwargs: calls.append("trust")
        or SimpleNamespace(config=resolved),
    )
    requests: list[SessionStartRequest] = []

    def open_session(*args, **kwargs):
        requests.append(kwargs["request"])
        return object()

    monkeypatch.setattr(
        "mycode.application.startup.start_agent_application_session",
        open_session,
    )
    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        mcp_trust_confirmer=object(),
    )

    factory.open_session(SessionStartRequest(mode="new"))
    factory.open_session(SessionStartRequest(mode="new"))
    factory.open_session(SessionStartRequest(mode="resume", session_id="saved"))

    assert calls == ["llm", "mcp", "trust"]
    assert requests == [
        SessionStartRequest(mode="new"),
        SessionStartRequest(mode="new"),
        SessionStartRequest(mode="resume", session_id="saved"),
    ]
    assert factory.resolved_mcp_config is resolved


def test_factory_sessions_have_independent_runners_and_mcp_managers(
    tmp_path,
    monkeypatch,
):
    from mycode.application import agent_session as agent_session_module

    class FakeActiveSession:
        def __init__(self, session_id: str):
            self.record = SimpleNamespace(id=session_id, title=session_id)
            self.writer = SimpleNamespace(append_subagent_event=lambda *args: None)
            self.compact_state_recovered = False
            self.artifact_directory = tmp_path
            self.closed = 0
            self.interrupted = 0

        def load_history(self):
            return Conversation()

        def load_compact_state(self):
            return object()

        def persist_message(self, message):
            del message

        def persist_compact_state(self, state):
            del state

        def close(self):
            self.closed += 1

        def interrupt(self):
            self.interrupted += 1

    class FakeManager:
        def __init__(self, config, **kwargs):
            del kwargs
            self.config = config
            self.tools = ()
            self.statuses = ()
            self.close_calls = 0

        def start(self):
            pass

        def close(self):
            self.close_calls += 1

    active_sessions = iter(
        [FakeActiveSession("a"), FakeActiveSession("b")]
    )
    managers: list[FakeManager] = []
    runners: list[object] = []

    monkeypatch.setattr(
        agent_session_module,
        "start_project_session",
        lambda *args, **kwargs: next(active_sessions),
    )

    def manager_factory(config, **kwargs):
        manager = FakeManager(config, **kwargs)
        managers.append(manager)
        return manager

    monkeypatch.setattr(agent_session_module, "MCPManager", manager_factory)

    def runner_factory(**kwargs):
        del kwargs
        runner = SimpleNamespace(tool_registry=SimpleNamespace(register=lambda tool: None))
        runners.append(runner)
        return runner

    monkeypatch.setattr(agent_session_module, "build_agent_runner", runner_factory)
    snapshot = MCPConfig()
    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        llm_config=_llm_config(),
        mcp_config=snapshot,
    )

    first = factory.open_session(SessionStartRequest(mode="new"))
    second = factory.open_session(SessionStartRequest(mode="new"))

    assert first is not second
    assert first.runner is not second.runner
    assert first.mcp_manager is not second.mcp_manager
    assert first.mcp_manager.config is snapshot
    assert second.mcp_manager.config is snapshot

    first.close()
    second.interrupt()
    assert first.active_project_session.closed == 1
    assert second.active_project_session.interrupted == 1
    assert all(manager.close_calls == 1 for manager in managers)


def test_factory_reuses_confirmer_without_sharing_session_approval_state(
    tmp_path,
    monkeypatch,
):
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope="session"))

    def open_session(*args, **kwargs):
        del args, kwargs
        registry = ToolRegistry.from_tools(
            [AuthorizedDecisionTool("probe")],
            confirmer=confirmer,
            permission_checker=RecordingPermissionChecker(PermissionDecision.ask()),
        )
        return SimpleNamespace(
            runner=SimpleNamespace(tool_registry=registry),
            close=lambda: None,
            interrupt=lambda: None,
        )

    monkeypatch.setattr(
        "mycode.application.startup.start_agent_application_session",
        open_session,
    )
    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        llm_config=_llm_config(),
        mcp_config=MCPConfig(),
        confirmer=confirmer,
    )

    first = factory.open_session(SessionStartRequest(mode="new"))
    second = factory.open_session(SessionStartRequest(mode="new"))
    assert first.runner.tool_registry.run_tool("probe", {"text": "a"}).ok
    assert second.runner.tool_registry.run_tool("probe", {"text": "b"}).ok
    assert len(confirmer.requests) == 2
    first.close()
    second.interrupt()


def test_trust_rejection_uses_resolver_filtered_config(tmp_path, monkeypatch):
    filtered = MCPConfig()
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: _loaded_config(),
    )
    monkeypatch.setattr(
        "mycode.application.startup.resolve_project_mcp_trust",
        lambda *args, **kwargs: MCPTrustResolution(config=filtered, approved=False),
    )

    factory = prepare_application_session_factory(
        create_application_environment(tmp_path),
        llm_config=_llm_config(),
        mcp_trust_confirmer=object(),
    )

    assert factory.resolved_mcp_config is filtered


def test_unexpected_trust_error_propagates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mycode.application.startup.load_mcp_config_layers",
        lambda **kwargs: _loaded_config(),
    )
    monkeypatch.setattr(
        "mycode.application.startup.resolve_project_mcp_trust",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trust boom")),
    )

    with pytest.raises(RuntimeError, match="trust boom"):
        prepare_application_session_factory(
            create_application_environment(tmp_path),
            llm_config=_llm_config(),
            mcp_trust_confirmer=object(),
        )
