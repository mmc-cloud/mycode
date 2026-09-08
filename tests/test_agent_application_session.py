from types import SimpleNamespace
from typing import Iterator

import pytest

from mycode.agent.events import AgentEvent
from mycode.agent.outcome import AgentRunOutcome
from mycode.application.agent_session import (
    AgentApplicationSession,
    start_agent_application_session,
)
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig
from mycode.mcp.models import MCPServerStatus
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity
from mycode.subagents.observability import CompositeSubAgentObserver
from mycode.subagents.persistence import SessionSubAgentObserver
from mycode.tools import ToolArgs, ToolRegistry, ToolResult, PydanticTool
from mycode.messages import Message


class EchoArgs(ToolArgs):
    value: str


class EchoTool(PydanticTool[EchoArgs]):
    name = "mcp__test__echo"
    description = "Echo."
    args_model = EchoArgs
    capability = "read"
    risk = "low"

    def _run(self, args: EchoArgs) -> ToolResult:
        return ToolResult.success(args.value)


class FakeRunner:
    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()
        self.seen_content: list[str] = []

    def run(self, content: str) -> Iterator[AgentEvent]:
        self.seen_content.append(content)
        yield AgentEvent(type="text_delta", content="done")
        yield AgentEvent(type="stop", stop_reason="final_answer")


class FakeMCPManager:
    instances: list["FakeMCPManager"] = []
    fail_on_start = False

    def __init__(self, config, observability_sink=None) -> None:
        self.config = config
        self.observability_sink = observability_sink
        self.statuses = (
            MCPServerStatus(alias="local", status="connected", tool_count=1),
        )
        self.tools = (EchoTool(),)
        self.start_calls = 0
        self.close_calls = 0
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_on_start:
            raise RuntimeError("MCP startup failed")

    def close(self) -> None:
        self.close_calls += 1


def configured_llm() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )


def test_application_session_wires_runtime_and_closes_idempotently(
    tmp_path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    existing = store.create_session(project, session_id="session")
    store.append_message(project, existing.id, Message(role="user", content="history"))
    runner = FakeRunner()
    captured: dict[str, object] = {}
    external_observer = SimpleNamespace()
    supplied_sink = object()

    def fake_build_agent_runner(**kwargs):
        captured.update(kwargs)
        return runner

    FakeMCPManager.instances.clear()
    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )
    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        fake_build_agent_runner,
    )

    application_session = start_agent_application_session(
        store,
        project,
        request=SessionStartRequest(mode="resume", session_id=existing.id),
        mcp_config=MCPConfig(),
        external_observer=external_observer,
        llm_config=configured_llm(),
        observability_sink=supplied_sink,
    )

    assert isinstance(application_session, AgentApplicationSession)
    assert captured["workspace_path"] == tmp_path.resolve()
    assert captured["conversation_history"].get_messages() == [
        Message(role="user", content="history")
    ]
    assert callable(captured["on_message_added"])
    assert captured["compact_state"].boundary is None
    assert callable(captured["on_compact_state_changed"])
    assert captured["artifact_directory"].is_absolute()
    assert captured["llm_session_id"] == existing.id
    assert captured["observability_sink"] is supplied_sink
    assert isinstance(captured["subagent_observer"], CompositeSubAgentObserver)
    assert isinstance(
        captured["subagent_observer"].observers[0],
        SessionSubAgentObserver,
    )
    assert captured["subagent_observer"].observers[1] is external_observer
    assert application_session.mcp_statuses == FakeMCPManager.instances[0].statuses
    assert FakeMCPManager.instances[0].observability_sink is supplied_sink
    assert runner.tool_registry.require("mcp__test__echo").name == "mcp__test__echo"
    assert FakeMCPManager.instances[0].start_calls == 1

    startup_events = list(application_session.startup_events())
    assert startup_events[0].type == "runtime_ready"
    assert startup_events[0].session_id == existing.id
    assert startup_events[0].session_created is False
    assert startup_events[1].type == "mcp_status"
    assert startup_events[1].mcp_status == FakeMCPManager.instances[0].statuses[0]

    events: list[RuntimeEvent] = []
    outcome = application_session.run_turn("inspect", event_handler=events.append)

    assert runner.seen_content == ["inspect"]
    assert isinstance(events[0], RuntimeEvent)
    assert events[-2].type == "agent"
    assert events[-2].agent_event is not None
    assert events[-2].agent_event.type == "stop"
    assert events[-1].type == "turn_finished"
    assert events[-1].outcome == outcome
    assert outcome == AgentRunOutcome.from_stop_reason("final_answer")

    application_session.close()
    application_session.close()
    application_session.interrupt()

    assert store.get_session(project, existing.id).status == "closed"
    assert FakeMCPManager.instances[0].close_calls == 1


def test_application_session_always_persists_subagents_without_external_observer(
    tmp_path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )

    def fake_build_agent_runner(**kwargs):
        captured["kwargs"] = kwargs
        return FakeRunner()

    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        fake_build_agent_runner,
    )

    application_session = start_agent_application_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        mcp_config=MCPConfig(),
        llm_config=configured_llm(),
    )
    try:
        assert isinstance(
            captured["kwargs"]["subagent_observer"],
            SessionSubAgentObserver,
        )
    finally:
        application_session.interrupt()

    assert store.list_sessions(project)[0].status == "interrupted"


@pytest.mark.parametrize(
    ("cleanup", "expected_status"),
    [("close", "closed"), ("interrupt", "interrupted")],
)
def test_application_session_rejects_turn_after_cleanup(
    tmp_path,
    monkeypatch,
    cleanup,
    expected_status,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    runner = FakeRunner()
    FakeMCPManager.instances.clear()
    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )
    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        lambda **kwargs: runner,
    )

    application_session = start_agent_application_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        mcp_config=MCPConfig(),
        llm_config=configured_llm(),
    )

    getattr(application_session, cleanup)()

    with pytest.raises(
        RuntimeError,
        match="already closed or interrupted",
    ):
        application_session.run_turn("must not run")

    assert runner.seen_content == []
    assert store.list_sessions(project)[0].status == expected_status


def test_runner_startup_failure_closes_mcp_and_releases_session(
    tmp_path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    FakeMCPManager.instances.clear()
    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )

    def fail_build(**kwargs):
        raise RuntimeError("runner startup failed")

    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        fail_build,
    )

    with pytest.raises(RuntimeError, match="runner startup failed"):
        start_agent_application_session(
            store,
            project,
            request=SessionStartRequest(mode="new"),
            mcp_config=MCPConfig(),
            llm_config=configured_llm(),
        )

    manager = FakeMCPManager.instances[0]
    assert manager.start_calls == 1
    assert manager.close_calls == 1
    session = store.list_sessions(project)[0]
    assert session.status == "interrupted"
    with store.open_session(project, session.id):
        pass


def test_mcp_startup_failure_also_releases_session(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    FakeMCPManager.instances.clear()
    FakeMCPManager.fail_on_start = True
    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )
    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        lambda **kwargs: pytest.fail("runner must not build"),
    )

    try:
        with pytest.raises(RuntimeError, match="MCP startup failed"):
            start_agent_application_session(
                store,
                project,
                request=SessionStartRequest(mode="new"),
                mcp_config=MCPConfig(),
                llm_config=configured_llm(),
            )
    finally:
        FakeMCPManager.fail_on_start = False

    manager = FakeMCPManager.instances[0]
    assert manager.close_calls == 1
    session = store.list_sessions(project)[0]
    assert session.status == "interrupted"
    with store.open_session(project, session.id):
        pass
