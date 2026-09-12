import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread, get_ident
from types import SimpleNamespace

from mycode.subagents.contracts import SubAgentResult, SubAgentTask
from mycode.subagents.lifecycle import SubAgentStateTransition
from mycode.subagents.runtime import SubAgentExecution
from mycode.presentation.tui.interactions import (
    PermissionRequestMessage,
    SubAgentResultMessage,
    SubAgentStateMessage,
)
import pytest

from mycode.agent.events import (
    AgentEvent,
    AgentModelRetry,
    AgentProgressSnapshot,
    AgentToolCall,
)
from mycode.agent.outcome import AgentRunOutcome
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import SessionStartRequest
from mycode.cli import main
from mycode.config import LLMConfig
from mycode.conversation import Conversation
from mycode.mcp import MCPConfig
from mycode.mcp.models import MCPServerStatus
from mycode.mcp.trust import MCPTrustRequest, MCPTrustServer
from mycode.permissions import (
    ConfirmationRequest,
    PermissionDecision,
    PermissionRequest,
)
from mycode.persistence.session_store import SessionInUseError, SessionStore
from mycode.presentation.tui import app as tui_app
from mycode.presentation.tui.app import MyCodeTuiApp
from mycode.presentation.tui.screens import (
    LoadingScreen,
    MCPTrustScreen,
    MainScreen,
    PermissionScreen,
    WelcomeScreen,
)
from mycode.presentation.tui.widgets import ConversationView, HeaderBar, StatusBar
from mycode.tools import ToolResult
from mycode.messages import Message
from textual.css.query import NoMatches
from textual.widgets import OptionList


def run_async(coroutine):
    return asyncio.run(coroutine)


def _permission_modal_count(app: MyCodeTuiApp) -> int:
    return sum(isinstance(screen, PermissionScreen) for screen in app.screen_stack)


def test_tui_cli_route_starts_tui(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("mycode.cli.run_tui", lambda: calls.append(True))

    main(["tui"])

    assert calls == [True]


def test_tui_help_does_not_start_app(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "mycode.cli.run_tui",
        lambda: pytest.fail("--help must not start TUI"),
    )

    with pytest.raises(SystemExit) as error:
        main(["tui", "--help"])

    captured = capsys.readouterr()
    assert error.value.code == 0
    assert "usage: mycode tui" in captured.out
    assert "14.6.2" in captured.out
    assert captured.err == ""


def _llm_config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )


def _app(tmp_path: Path, *, store: SessionStore | None = None, **kwargs) -> MyCodeTuiApp:
    return MyCodeTuiApp(
        workspace_path=tmp_path,
        session_store=store or SessionStore(tmp_path / "projects"),
        llm_config=_llm_config(),
        mcp_config=MCPConfig(),
        **kwargs,
    )


async def _wait_for_welcome(app: MyCodeTuiApp, pilot) -> WelcomeScreen:
    for _ in range(20):
        await pilot.pause(0.05)
        if isinstance(app.screen, WelcomeScreen):
            try:
                options = app.screen.query_one(OptionList)
            except NoMatches:
                continue
            if not options.disabled:
                return app.screen
    raise AssertionError("Welcome session options did not become ready")


async def _open_main(app: MyCodeTuiApp, pilot) -> None:
    await _wait_for_welcome(app, pilot)
    app.switch_screen(MainScreen(workspace="workspace", model="model"))
    await pilot.pause()
    assert isinstance(app.screen, MainScreen)


async def _open_main_with_session(app: MyCodeTuiApp, pilot) -> None:
    await _wait_for_welcome(app, pilot)
    app.screen.query_one("#session-options").focus()
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause(0.02)
        if isinstance(app.screen, MainScreen):
            break
    assert isinstance(app.screen, MainScreen)


def test_tui_enters_welcome_and_does_not_auto_start_main(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            screen = await _wait_for_welcome(app, pilot)
            assert screen.query_one("#welcome-logo")
            assert "test-model" in str(screen.query_one("#welcome-meta").render())
            assert screen.query_one("#session-options").option_count == 2

    run_async(exercise())


def test_session_option_mapping_is_limited_to_core_modes() -> None:
    assert WelcomeScreen.request_from_option("new") == SessionStartRequest(mode="new")
    assert WelcomeScreen.request_from_option("continue") == SessionStartRequest(
        mode="continue"
    )
    assert WelcomeScreen.request_from_option("resume:abc") == SessionStartRequest(
        mode="resume",
        session_id="abc",
    )


def test_welcome_lists_real_sessions_without_fake_resume(tmp_path) -> None:
    store = SessionStore(tmp_path / "projects")
    from mycode.project import ProjectIdentity

    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="abcdef123456", title="Existing work")

    async def exercise() -> None:
        app = _app(tmp_path, store=store)
        async with app.run_test() as pilot:
            screen = await _wait_for_welcome(app, pilot)
            option_list = screen.query_one("#session-options")
            rendered = str(option_list.get_option_at_index(2).prompt)
            assert "Resume: Existing work    abcdef12" in rendered
            assert "Resume: fake" not in rendered

    run_async(exercise())


@dataclass
class _FakeApplicationSession:
    history: Conversation
    session_id: str = "session-1"
    closed: bool = False
    close_count: int = 0
    interrupt_count: int = 0

    def __post_init__(self) -> None:
        self.active_project_session = SimpleNamespace(
            load_history=lambda: self.history,
            record=SimpleNamespace(id=self.session_id),
        )

    def startup_events(self):
        yield RuntimeEvent(
            type="runtime_ready",
            session_id=self.session_id,
            session_title="Demo session",
        )

    def close(self) -> None:
        self.close_count += 1
        self.closed = True

    def interrupt(self) -> None:
        self.interrupt_count += 1
        self.closed = True


def _fake_session() -> _FakeApplicationSession:
    return _FakeApplicationSession(
        Conversation.from_messages(
            [
                Message(role="user", content="old question"),
                Message(role="assistant", content="old answer"),
                Message(role="tool", content="raw tool result", tool_call_id="call-1"),
            ]
        )
    )


def test_new_starts_application_session_in_worker_and_replays_history(monkeypatch, tmp_path) -> None:
    fake = _fake_session()
    captured: dict[str, object] = {}

    def fake_start(*args, **kwargs):
        captured["request"] = kwargs["request"]
        captured["thread"] = get_ident()
        return fake

    monkeypatch.setattr(tui_app, "start_agent_application_session", fake_start)

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("down", "enter")
            await pilot.pause(0.25)

            assert isinstance(app.screen, MainScreen)
            assert captured["request"] == SessionStartRequest(mode="new")
            assert captured["thread"] != app._thread_id
            conversation = app.screen.query_one(ConversationView)
            assert "old question" in conversation.transcript_text
            assert "old answer" in conversation.transcript_text
            assert "raw tool result" not in conversation.transcript_text
            assert "Demo session" in conversation.transcript_text

    run_async(exercise())


@pytest.mark.parametrize("option_key, expected", [("continue", "continue"), ("new", "new")])
def test_welcome_selection_builds_expected_request(monkeypatch, tmp_path, option_key, expected) -> None:
    fake = _fake_session()
    captured: list[SessionStartRequest] = []

    def fake_start(*args, **kwargs):
        captured.append(kwargs["request"])
        return fake

    monkeypatch.setattr(tui_app, "start_agent_application_session", fake_start)

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            option_list = app.screen.query_one("#session-options")
            option_list.focus()
            if option_key == "new":
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert captured == [SessionStartRequest(mode=expected)]

    run_async(exercise())


def test_session_in_use_returns_to_welcome_and_refreshes(monkeypatch, tmp_path) -> None:
    store = SessionStore(tmp_path / "projects")
    from mycode.project import ProjectIdentity

    project = ProjectIdentity.from_workspace(tmp_path)

    def fake_start(*args, **kwargs):
        store.create_session(project, session_id="refreshed-1", title="Refreshed session")
        raise SessionInUseError("Session is in use by another owner.")

    monkeypatch.setattr(tui_app, "start_agent_application_session", fake_start)

    async def exercise() -> None:
        app = _app(tmp_path, store=store)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("enter")
            refreshed = False
            for _ in range(150):
                await pilot.pause(0.02)
                if not isinstance(app.screen, WelcomeScreen):
                    continue
                notice = str(app.screen.query_one("#welcome-notice").render())
                options = app.screen.query_one("#session-options")
                if (
                    "currently in use" in notice
                    and options.option_count == 3
                    and "Resume: Refreshed session    refreshe"
                    in str(options.get_option_at_index(2).prompt)
                ):
                    refreshed = True
                    break

            assert refreshed

    run_async(exercise())


def test_startup_session_created_after_shutdown_is_closed(monkeypatch, tmp_path) -> None:
    fake = _fake_session()
    start_entered = Event()
    release_start = Event()

    def fake_start(*args, **kwargs):
        start_entered.set()
        release_start.wait()
        return fake

    monkeypatch.setattr(tui_app, "start_agent_application_session", fake_start)

    async def exercise() -> None:
        app = _app(tmp_path)
        try:
            async with app.run_test() as pilot:
                await _wait_for_welcome(app, pilot)
                app.screen.query_one("#session-options").focus()
                await pilot.press("enter")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if start_entered.is_set():
                        break
                assert start_entered.is_set()

                app.exit()
                app.on_unmount()
                assert app._shutdown_requested.is_set()

                release_start.set()
                for _ in range(40):
                    await pilot.pause(0.02)
                    if fake.closed:
                        break
                assert fake.closed is True
                with app._session_lock:
                    assert app._pending_application_sessions == []
                    assert app._application_session is None
        finally:
            release_start.set()

    run_async(exercise())


def test_startup_exception_after_session_creation_closes_and_releases_session(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()

    def fail_load_history():
        raise RuntimeError("history load failed")

    fake.active_project_session.load_history = fail_load_history
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert fake.closed is True
            with app._session_lock:
                assert app._pending_application_sessions == []
                assert app._application_session is None
            assert isinstance(app.screen, WelcomeScreen)
            assert "history load failed" in str(
                app.screen.query_one("#welcome-error").render()
            )

    run_async(exercise())


def test_startup_exception_after_shutdown_claim_does_not_double_close(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    history_entered = Event()
    release_history = Event()

    def fail_load_history_after_shutdown() -> Conversation:
        history_entered.set()
        release_history.wait()
        raise RuntimeError("history load failed after shutdown")

    fake.active_project_session.load_history = fail_load_history_after_shutdown
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        try:
            async with app.run_test() as pilot:
                await _wait_for_welcome(app, pilot)
                app.screen.query_one("#session-options").focus()
                await pilot.press("enter")
                assert history_entered.wait(1)

                app.on_unmount()
                assert fake.close_count == 1
                with app._session_lock:
                    assert app._pending_application_sessions == []
                    assert app._application_session is None

                release_history.set()
                for _ in range(40):
                    await pilot.pause(0.02)
                    if not app._startup_active:
                        break
                assert fake.close_count == 1
                with app._session_lock:
                    assert app._pending_application_sessions == []
                    assert app._application_session is None
        finally:
            release_history.set()

    run_async(exercise())


def _trust_request() -> MCPTrustRequest:
    return MCPTrustRequest(
        servers=(
            MCPTrustServer(
                alias="test",
                transport="stdio",
                command="mcp-server",
                args=(),
            ),
        )
    )


def test_mcp_trust_waiter_is_rejected_when_bridge_shuts_down() -> None:
    from mycode.presentation.tui.interactions import TuiMCPTrustConfirmer

    posted = Event()
    result: list[bool] = []

    def post_message(message: object) -> bool:
        posted.set()
        return True

    confirmer = TuiMCPTrustConfirmer(post_message)

    def wait_for_confirmation() -> None:
        result.append(confirmer.confirm(_trust_request()))

    waiter = Thread(target=wait_for_confirmation)
    waiter.start()
    assert posted.wait(1)
    confirmer.reject_all()
    waiter.join(1)

    assert not waiter.is_alive()
    assert result == [False]


def test_mcp_trust_confirm_after_bridge_shutdown_returns_false_immediately() -> None:
    from mycode.presentation.tui.interactions import TuiMCPTrustConfirmer

    posted: list[object] = []
    confirmer = TuiMCPTrustConfirmer(lambda message: posted.append(message) or True)
    confirmer.reject_all()

    assert confirmer.confirm(_trust_request()) is False
    assert posted == []


def _permission_request() -> ConfirmationRequest:
    decision = PermissionDecision.ask(
        message="Command operation requires confirmation: run_command",
        metadata={
            "command_display": "echo safe",
            "command_risk": "low",
            "secret_value": "SECRET_VALUE",
        },
    )
    return ConfirmationRequest(
        permission_request=PermissionRequest(
            tool_name="run_command",
            capability="command",
            action="run command",
            target="echo safe",
            arguments={"command": ["echo", "safe"], "token": "SECRET_VALUE"},
        ),
        permission_decision=decision,
        prompt=decision.message,
    )


def test_permission_waiter_is_rejected_when_bridge_shuts_down() -> None:
    from mycode.presentation.tui.interactions import TuiConfirmer

    posted = Event()
    result = []

    def post_message(message: object) -> bool:
        posted.set()
        return True

    confirmer = TuiConfirmer(post_message)

    def wait_for_confirmation() -> None:
        result.append(confirmer.confirm(_permission_request()))

    waiter = Thread(target=wait_for_confirmation)
    waiter.start()
    assert posted.wait(1)
    confirmer.reject_all()
    waiter.join(1)

    assert not waiter.is_alive()
    assert result[0].status == "rejected"


def test_permission_confirm_after_bridge_shutdown_returns_immediately() -> None:
    from mycode.presentation.tui.interactions import TuiConfirmer

    posted: list[object] = []
    confirmer = TuiConfirmer(lambda message: posted.append(message) or True)
    confirmer.reject_all()

    result = confirmer.confirm(_permission_request())

    assert result.status == "rejected"
    assert posted == []


@pytest.mark.parametrize(
    ("button_id", "expected_status", "expected_scope"),
    [
        ("permission-allow-once", "approved", "once"),
        ("permission-allow-task", "approved", "task"),
        ("permission-allow-session", "approved", "session"),
        ("permission-deny", "rejected", None),
    ],
)
def test_permission_modal_maps_scopes_and_hides_secret_values(
    monkeypatch,
    tmp_path,
    button_id,
    expected_status,
    expected_scope,
) -> None:
    fake = _fake_session()
    confirmer_holder: list[object] = []
    decisions = []

    def fake_start(*args, **kwargs):
        confirmer_holder.append(kwargs["confirmer"])
        return fake

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        request = _permission_request()
        decision = confirmer_holder[0].confirm(request)
        decisions.append(decision)
        assert event_handler is not None
        event_handler(
            RuntimeEvent(
                type="agent",
                turn_id=turn_id,
                agent_event=AgentEvent(
                    type="text_delta",
                    content="approved" if decision.status == "approved" else "denied",
                ),
            )
        )
        outcome = AgentRunOutcome.from_stop_reason("final_answer")
        event_handler(
            RuntimeEvent(type="turn_finished", turn_id=turn_id, outcome=outcome)
        )
        return outcome

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(tui_app, "start_agent_application_session", fake_start)

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main_with_session(app, pilot)
            app.submit_user_message("needs permission")
            for _ in range(40):
                await pilot.pause(0.02)
                if isinstance(app.screen, PermissionScreen):
                    try:
                        app.screen.query_one(f"#{button_id}")
                    except NoMatches:
                        continue
                    break
            assert isinstance(app.screen, PermissionScreen)
            details = str(app.screen.query_one("#permission-details").render())
            assert "command_display: echo safe" in details
            assert "SECRET_VALUE" not in details

            app.screen.query_one(f"#{button_id}").press()
            await pilot.pause()
            for _ in range(40):
                await pilot.pause(0.02)
                if app._active_turn_id is None:
                    break
            assert app._active_turn_id is None
            assert decisions[0].status == expected_status
            assert decisions[0].scope == expected_scope

    run_async(exercise())


def test_stale_session_refresh_cannot_replace_loading_screen(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    refresh_started = Event()
    release_refresh = Event()
    startup_started = Event()
    release_startup = Event()
    list_calls = 0
    start_calls = 0

    def controlled_list(*args, **kwargs):
        nonlocal list_calls
        list_calls += 1
        if list_calls == 2:
            refresh_started.set()
            release_refresh.wait()
        return ()

    def controlled_start(*args, **kwargs):
        nonlocal start_calls
        start_calls += 1
        if start_calls == 1:
            raise SessionInUseError("Session is in use by another owner.")
        startup_started.set()
        release_startup.wait()
        return fake

    monkeypatch.setattr(tui_app, "list_project_sessions", controlled_list)
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        controlled_start,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        try:
            async with app.run_test() as pilot:
                await _wait_for_welcome(app, pilot)
                app.screen.query_one("#session-options").focus()
                await pilot.press("enter")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if refresh_started.is_set():
                        break
                assert refresh_started.is_set()

                app.screen.query_one("#session-options").focus()
                await pilot.press("enter")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if startup_started.is_set():
                        break
                assert startup_started.is_set()
                assert isinstance(app.screen, LoadingScreen)

                release_refresh.set()
                await pilot.pause(0.15)
                assert isinstance(app.screen, LoadingScreen)

                release_startup.set()
                for _ in range(40):
                    await pilot.pause(0.02)
                    if isinstance(app.screen, MainScreen):
                        break
                assert isinstance(app.screen, MainScreen)
        finally:
            release_refresh.set()
            release_startup.set()

    run_async(exercise())


@pytest.mark.parametrize("approved", [True, False])
def test_mcp_trust_modal_bridges_worker_and_hides_secret_values(
    monkeypatch,
    tmp_path,
    approved,
) -> None:
    fake = _fake_session()
    decisions: list[bool] = []

    monkeypatch.setattr(
        tui_app,
        "load_mcp_config_layers",
        lambda **kwargs: object(),
    )

    def fake_resolve(loaded, project, *, confirmer, trust_file):
        decision = confirmer.confirm(
            MCPTrustRequest(
                servers=(
                    MCPTrustServer(
                        alias="private",
                        transport="stdio",
                        command="mcp-server",
                        args=("--safe",),
                        env_keys=("API_TOKEN",),
                    ),
                )
            )
        )
        decisions.append(decision)
        return SimpleNamespace(config=MCPConfig())

    monkeypatch.setattr(tui_app, "resolve_project_mcp_trust", fake_resolve)
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = MyCodeTuiApp(
            workspace_path=tmp_path,
            session_store=SessionStore(tmp_path / "projects"),
            llm_config=_llm_config(),
        )
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("enter")
            await pilot.pause(0.15)
            assert isinstance(app.screen, MCPTrustScreen)
            details = str(app.screen.query_one("#mcp-trust-details").render())
            assert "API_TOKEN" in details
            assert "SECRET_VALUE" not in details
            await pilot.click(
                "#mcp-trust-approve" if approved else "#mcp-trust-reject"
            )
            await pilot.pause(0.2)
            assert decisions == [approved]
            assert isinstance(app.screen, MainScreen)

    run_async(exercise())


def test_startup_failure_returns_to_welcome_without_traceback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("startup boom")),
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, WelcomeScreen)
            assert "RuntimeError" in str(app.screen.query_one("#welcome-error").render())
            assert "startup boom" in str(app.screen.query_one("#welcome-error").render())

    run_async(exercise())


def test_main_input_only_adds_user_message_and_clears_input(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            await pilot.click("#prompt")
            await pilot.press("h", "i", "enter")
            conversation = app.screen.query_one(ConversationView)
            assert "hi" in conversation.transcript_text
            assert app.screen.query_one("#prompt").value == ""

    run_async(exercise())


def test_input_runs_application_session_in_worker_and_enforces_one_active_turn(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    turn_started = Event()
    release_turn = Event()
    calls: list[tuple[str, int]] = []

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        calls.append((content, get_ident()))
        turn_started.set()
        assert event_handler is not None
        event_handler(
            RuntimeEvent(
                type="agent",
                turn_id=turn_id,
                agent_event=AgentEvent(type="model_start"),
            )
        )
        release_turn.wait()
        event_handler(
            RuntimeEvent(
                type="agent",
                turn_id=turn_id,
                agent_event=AgentEvent(type="text_delta", content="answer"),
            )
        )
        outcome = AgentRunOutcome.from_stop_reason("final_answer")
        event_handler(
            RuntimeEvent(type="turn_finished", turn_id=turn_id, outcome=outcome)
        )
        return outcome

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        try:
            async with app.run_test() as pilot:
                await _wait_for_welcome(app, pilot)
                app.screen.query_one("#session-options").focus()
                await pilot.press("enter")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if isinstance(app.screen, MainScreen):
                        break
                assert isinstance(app.screen, MainScreen)

                presenter_threads: list[int] = []
                original_present = app.presenter.present

                def present_on_ui_thread(event) -> None:
                    presenter_threads.append(get_ident())
                    original_present(event)

                app.presenter.present = present_on_ui_thread

                await pilot.click("#prompt")
                await pilot.press("h", "i", "enter")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if turn_started.is_set():
                        break
                assert turn_started.is_set()
                assert calls[0][0] == "hi"
                assert calls[0][1] != app._thread_id
                assert app.screen.query_one("#prompt").disabled is True

                app.submit_user_message("second")
                assert [content for content, _ in calls] == ["hi"]
                app.action_quit()
                assert not app._shutdown_requested.is_set()
                assert fake.closed is False

                release_turn.set()
                for _ in range(40):
                    await pilot.pause(0.02)
                    if app._active_turn_id is None:
                        break
                assert app._active_turn_id is None
                assert app.screen.query_one("#prompt").disabled is False
                assert str(app.screen.query_one(StatusBar).render()) == "Ready"
                assert "answer" in app.screen.query_one(ConversationView).transcript_text
                assert presenter_threads
                assert all(thread_id == app._thread_id for thread_id in presenter_threads)
        finally:
            release_turn.set()

    run_async(exercise())


def test_runtime_failure_finishes_turn_and_allows_next_turn(monkeypatch, tmp_path) -> None:
    fake = _fake_session()
    calls: list[str] = []

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        calls.append(content)
        outcome = AgentRunOutcome.from_stop_reason(
            None if len(calls) == 1 else "final_answer"
        )
        assert event_handler is not None
        event_handler(
            RuntimeEvent(type="turn_finished", turn_id=turn_id, outcome=outcome)
        )
        return outcome

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main_with_session(app, pilot)
            app.submit_user_message("first")
            for _ in range(40):
                await pilot.pause(0.02)
                if app._active_turn_id is None:
                    break
            assert calls == ["first"]
            assert app._session_unusable is False
            assert app.screen.query_one("#prompt").disabled is False
            assert str(app.screen.query_one(StatusBar).render()) == "Ready"

            app.submit_user_message("second")
            for _ in range(40):
                await pilot.pause(0.02)
                if app._active_turn_id is None:
                    break
            assert calls == ["first", "second"]
            assert app._session_unusable is False

    run_async(exercise())


def test_turn_exception_interrupts_session_and_blocks_future_submit(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    interrupted = Event()
    calls: list[str] = []

    def interrupt() -> None:
        interrupted.set()

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        calls.append(content)
        raise RuntimeError("turn exploded")

    fake.interrupt = interrupt
    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main_with_session(app, pilot)
            app.submit_user_message("explode")
            for _ in range(40):
                await pilot.pause(0.02)
                if app._session_unusable:
                    break
            assert interrupted.is_set()
            assert app._session_unusable is True
            assert calls == ["explode"]
            assert app.screen.query_one("#prompt").disabled is True
            assert str(app.screen.query_one(StatusBar).render()) == "Fatal error"
            assert "fatal agent error" in app.screen.query_one(ConversationView).transcript_text

            app.submit_user_message("must not run")
            await pilot.pause()
            assert calls == ["explode"]

    run_async(exercise())


def test_active_turn_shutdown_defers_session_cleanup_to_turn_worker(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    turn_started = Event()
    release_turn = Event()

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        del content, turn_id, event_handler
        turn_started.set()
        release_turn.wait()
        return AgentRunOutcome.from_stop_reason("final_answer")

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        try:
            async with app.run_test() as pilot:
                await _open_main_with_session(app, pilot)
                app.submit_user_message("long turn")
                for _ in range(40):
                    await pilot.pause(0.02)
                    if turn_started.is_set():
                        break
                assert turn_started.is_set()
                with app._session_lock:
                    assert app._turn_session_owned_by_worker is True

                app.on_unmount()
                assert fake.close_count == 0
                assert fake.interrupt_count == 0

                release_turn.set()
                for _ in range(40):
                    await pilot.pause(0.02)
                    if fake.close_count == 1:
                        break
                assert fake.close_count == 1
                assert fake.interrupt_count == 0
                with app._session_lock:
                    assert app._application_session is None
                    assert app._pending_application_sessions == []
        finally:
            release_turn.set()

    run_async(exercise())


def test_shutdown_before_turn_worker_claim_closes_session_once(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    claim_entered = Event()
    release_claim = Event()
    claim_finished = Event()
    run_called = Event()

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        del content, turn_id, event_handler
        run_called.set()
        return AgentRunOutcome.from_stop_reason("final_answer")

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        original_claim = app._claim_turn_session_for_worker

        def blocked_claim():
            claim_entered.set()
            try:
                release_claim.wait()
                return original_claim()
            finally:
                claim_finished.set()

        app._claim_turn_session_for_worker = blocked_claim
        try:
            async with app.run_test() as pilot:
                await _open_main_with_session(app, pilot)
                app.submit_user_message("claim boundary")
                for _ in range(300):
                    await pilot.pause(0.02)
                    if claim_entered.is_set():
                        break
                assert claim_entered.is_set()

                app.on_unmount()
                assert fake.close_count == 1
                with app._session_lock:
                    assert app._application_session is None
                    assert app._turn_session_owned_by_worker is False

                release_claim.set()
                for _ in range(300):
                    await pilot.pause(0.02)
                    if claim_finished.is_set():
                        break
                assert claim_finished.is_set()
                await pilot.pause()
                assert run_called.is_set() is False
                assert fake.close_count == 1
        finally:
            release_claim.set()

    run_async(exercise())


def test_shutdown_after_worker_release_closes_session_once_before_completion(
    monkeypatch, tmp_path
) -> None:
    fake = _fake_session()
    run_returned = Event()
    completion_posted = Event()
    release_completion = Event()
    worker_finished = Event()

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        del content, turn_id, event_handler
        run_returned.set()
        return AgentRunOutcome.from_stop_reason("final_answer")

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        original_post_message = app.post_message

        def blocked_post_message(message):
            if isinstance(message, tui_app.TurnCompletedMessage):
                completion_posted.set()
                release_completion.wait()
                worker_finished.set()
                return True
            return original_post_message(message)

        app.post_message = blocked_post_message
        try:
            async with app.run_test() as pilot:
                await _open_main_with_session(app, pilot)
                app.submit_user_message("release boundary")
                for _ in range(300):
                    await pilot.pause(0.02)
                    if completion_posted.is_set():
                        break
                assert run_returned.is_set()
                assert completion_posted.is_set()
                with app._session_lock:
                    assert app._application_session is fake
                    assert app._turn_session_owned_by_worker is False

                app.on_unmount()
                assert fake.close_count == 1
                with app._session_lock:
                    assert app._application_session is None

                release_completion.set()
                for _ in range(300):
                    await pilot.pause(0.02)
                    if worker_finished.is_set():
                        break
                assert worker_finished.is_set()
                await pilot.pause()
                assert fake.close_count == 1
        finally:
            release_completion.set()

    run_async(exercise())


def test_conversation_auto_scrolls_after_incremental_entry_updates(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            conversation = app.screen.query_one(ConversationView)
            for index in range(40):
                conversation.add_notice(f"notice {index}")
            await pilot.pause()
            assert conversation.max_scroll_y > 0
            assert conversation.scroll_y == conversation.max_scroll_y

    run_async(exercise())


def test_tui_presenter_maps_runtime_and_agent_events(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            conversation = app.screen.query_one(ConversationView)
            status = app.screen.query_one(StatusBar)
            header = app.screen.query_one(HeaderBar)

            app.present_event(
                RuntimeEvent(
                    type="runtime_ready",
                    session_id="session-1",
                    session_title="Demo",
                    instruction_warnings=("instruction warning",),
                    skill_warnings=("skill warning",),
                )
            )
            app.present_event(
                RuntimeEvent(
                    type="mcp_status",
                    mcp_status=MCPServerStatus(alias="github", status="connected", tool_count=2),
                )
            )
            app.present_event(
                RuntimeEvent(
                    type="agent",
                    agent_event=AgentEvent(type="turn", turn_number=3, max_turns=20),
                )
            )
            app.present_event(
                RuntimeEvent(
                    type="agent",
                    agent_event=AgentEvent(type="context", content="trimmed=2"),
                )
            )
            app.present_event(
                RuntimeEvent(
                    type="agent",
                    agent_event=AgentEvent(
                        type="progress",
                        progress=AgentProgressSnapshot(
                            reason="repetition_observed",
                            stagnation_turns=2,
                            same_tool_repeat=1,
                        ),
                    ),
                )
            )
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="model_start")))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="text_delta", content="hello")))
            await pilot.pause()
            assistant_widget = conversation.children[-1]
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="text_delta", content=" world")))
            await pilot.pause()

            text = conversation.transcript_text
            assert "session=Demo" in text
            assert "MCP github connected (2 tools)" in text
            assert "· turn 3/20" in text
            assert "· context trimmed=2" in text
            assert "reason=repetition_observed" in text
            assert "hello world" in text
            assert conversation.children[-1] is assistant_widget
            assert "Demo" in str(header.render())
            assert str(status.render()) == "Responding…"

    run_async(exercise())


def test_tool_activity_updates_pending_rows_and_reasoning_is_hidden(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            conversation = app.screen.query_one(ConversationView)
            status = app.screen.query_one(StatusBar)
            before_reasoning = conversation.transcript_text
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="reasoning_state", reasoning_state="present_nonempty")))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="reasoning_delta", reasoning_content="private reasoning")))
            assert conversation.transcript_text == before_reasoning

            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="tool_call", tool_call=AgentToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}))))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="tool_call", tool_call=AgentToolCall(id="call-2", name="grep", arguments={"query": "main", "path_pattern": "mycode/*.py"}))))
            await pilot.pause()
            read_file_widget, grep_widget = conversation.children

            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="tool_result", tool_result=ToolResult.success(content="file content"))))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="tool_result", tool_result=ToolResult.failure(error="file not found"))))
            await pilot.pause()
            assert "✓ read_file" in conversation.transcript_text
            assert "✗ grep file not found" in conversation.transcript_text
            assert conversation.children[0] is read_file_widget
            assert conversation.children[1] is grep_widget

            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="model_retry", model_retry=AgentModelRetry(attempt=1, max_retries=2, delay_seconds=1.5, error_type="TimeoutError", error_code="timeout", retryable=True, call_kind="agent_tools", stream_started=False, partial_output_chars=0))))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="artifact_warning", content="artifact not persisted")))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="error", error="model failed")))
            app.present_event(RuntimeEvent(type="agent", agent_event=AgentEvent(type="stop", stop_reason="model_error", content="turn stopped")))
            text = conversation.transcript_text
            assert "model retry 1/2" in text
            assert "⚠ artifact not persisted" in text
            assert "✗ model failed" in text
            assert "· stop reason=model_error turn stopped" in text
            assert str(status.render()) == "Stopped"

            app.present_event(RuntimeEvent(type="turn_finished", outcome=AgentRunOutcome.from_stop_reason("model_error")))
            assert str(status.render()) == "Ready"

    run_async(exercise())


def test_tool_pending_rows_reset_at_turn_boundary(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            conversation = app.screen.query_one(ConversationView)

            app.present_event(
                RuntimeEvent(
                    type="agent",
                    turn_id="turn-a",
                    agent_event=AgentEvent(
                        type="tool_call",
                        tool_call=AgentToolCall(
                            id="call-a",
                            name="read_file",
                            arguments={"path": "README.md"},
                        ),
                    ),
                )
            )
            read_file_widget = conversation.children[0]
            app.present_event(
                RuntimeEvent(
                    type="agent",
                    turn_id="turn-a",
                    agent_event=AgentEvent(
                        type="stop",
                        stop_reason="repeated_tool_call",
                    ),
                )
            )
            app.present_event(
                RuntimeEvent(
                    type="turn_finished",
                    turn_id="turn-a",
                    outcome=AgentRunOutcome.from_stop_reason("repeated_tool_call"),
                )
            )

            app.present_event(
                RuntimeEvent(
                    type="agent",
                    turn_id="turn-b",
                    agent_event=AgentEvent(
                        type="tool_call",
                        tool_call=AgentToolCall(
                            id="call-b",
                            name="grep",
                            arguments={"query": "main"},
                        ),
                    ),
                )
            )
            grep_widget = conversation.children[2]
            app.present_event(
                RuntimeEvent(
                    type="agent",
                    turn_id="turn-b",
                    agent_event=AgentEvent(
                        type="tool_result",
                        tool_result=ToolResult.success(
                            "grep result",
                            metadata={"tool_name": "grep"},
                        ),
                    ),
                )
            )
            await pilot.pause()

            assert conversation.children[0] is read_file_widget
            assert conversation.children[2] is grep_widget
            assert "› read_file" in conversation.transcript_text
            assert "✓ grep" in conversation.transcript_text
            assert "✓ read_file" not in conversation.transcript_text

    run_async(exercise())


def test_completed_turn_restores_prompt_focus(monkeypatch, tmp_path) -> None:
    fake = _fake_session()

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        del content, turn_id, event_handler
        return AgentRunOutcome.from_stop_reason("final_answer")

    fake.run_turn = fake_run_turn
    monkeypatch.setattr(
        tui_app,
        "start_agent_application_session",
        lambda *args, **kwargs: fake,
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main_with_session(app, pilot)
            prompt = app.screen.query_one("#prompt")
            app.submit_user_message("focus after turn")
            for _ in range(40):
                await pilot.pause(0.02)
                if app._active_turn_id is None:
                    break
            assert prompt.disabled is False
            assert app.focused is prompt

    run_async(exercise())


def test_successful_session_is_closed_on_normal_app_exit(monkeypatch, tmp_path) -> None:
    fake = _fake_session()
    monkeypatch.setattr(tui_app, "start_agent_application_session", lambda *args, **kwargs: fake)

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.screen.query_one("#session-options").focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, MainScreen)
            app.exit()
            await pilot.pause()
        assert fake.closed is True

    run_async(exercise())


# ---------------------------------------------------------------------------
# Stage 14.6.4 SubAgent TUI + Permission interaction queue
# ---------------------------------------------------------------------------


def _subagent_task() -> SubAgentTask:
    return SubAgentTask(role="explorer", objective="find call sites")


def _subagent_transition(state: str, reason: str) -> SubAgentStateTransition:
    return SubAgentStateTransition(
        run_id="run-1",
        role="explorer",
        state=state,
        occurred_at=datetime.now(UTC),
        reason=reason,
    )


def _subagent_result(status: str, stop_reason: str) -> SubAgentResult:
    return SubAgentResult(
        run_id="run-1",
        role="explorer",
        status=status,
        stop_reason=stop_reason,
        summary="found three call sites",
        payload=(
            {
                "status": "completed",
                "summary": "done",
                "searched_scope": ["src"],
                "findings": [
                    {
                        "path": "src/a.py",
                        "claim": "call site",
                        "evidence": "line 3",
                    }
                ],
            }
            if status == "completed"
            else None
        ),
    )


def _subagent_execution(result: SubAgentResult) -> SubAgentExecution:
    return SubAgentExecution(
        result=result,
        transitions=(),
        snapshot=None,
        context=None,
        token_usage=None,
        conversation_message_count=0,
        tool_call_count=0,
        validation_execution_count=0,
    )


def test_subagent_observer_posts_from_worker_thread_only() -> None:
    from mycode.presentation.tui.interactions import TuiSubAgentObserver

    posts: list[tuple[object, int]] = []
    observer = TuiSubAgentObserver(
        lambda message: posts.append((message, get_ident())) or True
    )

    def worker() -> None:
        observer.on_state(
            _subagent_task(), _subagent_transition("running", "run_started")
        )
        observer.on_snapshot(_subagent_task(), "run-1", None, datetime.now(UTC))
        observer.on_tool_audit(_subagent_task(), "run-1", None, datetime.now(UTC))
        observer.on_result(
            _subagent_task(),
            _subagent_execution(_subagent_result("completed", "submitted")),
            datetime.now(UTC),
        )

    thread = Thread(target=worker)
    thread.start()
    thread.join(1)

    assert thread.ident is not None
    worker_idents = {ident for _, ident in posts}
    assert worker_idents == {thread.ident}
    assert len(posts) >= 2


def test_subagent_lifecycle_entries_reach_conversation(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            app.post_message(
                SubAgentStateMessage(_subagent_transition("running", "run_started"))
            )
            app.post_message(
                SubAgentStateMessage(
                    _subagent_transition(
                        "awaiting_confirmation",
                        "permission_confirmation",
                    )
                )
            )
            app.post_message(
                SubAgentStateMessage(
                    _subagent_transition("interrupted", "agent_interrupt")
                )
            )
            app.post_message(
                SubAgentResultMessage(
                    _subagent_task(),
                    _subagent_execution(_subagent_result("completed", "submitted")),
                )
            )
            app.post_message(
                SubAgentResultMessage(
                    _subagent_task(),
                    _subagent_execution(_subagent_result("failed", "model_error")),
                )
            )
            await pilot.pause()
            await pilot.pause()

            transcript = app.screen.query_one(ConversationView).transcript_text
            assert "› Explorer started" in transcript
            assert "· Explorer waiting for permission" in transcript
            assert "⚠ Explorer interrupted" in transcript
            assert "✓ Explorer completed" in transcript
            assert "✗ Explorer failed" in transcript

    run_async(exercise())


def test_concurrent_permission_requests_are_serialized(tmp_path) -> None:
    from mycode.presentation.tui.interactions import PermissionResponseHandle

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            first = PermissionResponseHandle(_permission_request())
            second = PermissionResponseHandle(_permission_request())
            app.post_message(PermissionRequestMessage(first))
            app.post_message(PermissionRequestMessage(second))
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, PermissionScreen)
            assert _permission_modal_count(app) == 1
            status = app._status_bar
            assert status is not None
            for _ in range(20):
                await pilot.pause(0.02)
                if "Waiting for permission" in str(status.render()):
                    break
            assert "Waiting for permission" in str(status.render())

            app.screen.query_one("#permission-deny").press()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, PermissionScreen)
            assert _permission_modal_count(app) == 1

            app.screen.query_one("#permission-allow-once").press()
            for _ in range(40):
                await pilot.pause(0.02)
                if first._event.is_set() and second._event.is_set():
                    break
            for _ in range(40):
                await pilot.pause(0.02)
                if _permission_modal_count(app) == 0:
                    break
            assert _permission_modal_count(app) == 0
            assert first._result.status == "rejected"
            assert second._result.status == "approved"
            assert second._result.scope == "once"
            for _ in range(20):
                await pilot.pause(0.02)
                if str(status.render()) not in ("Ready", "Runtime Ready"):
                    break
            assert str(status.render()) in ("Ready", "Runtime Ready")

    run_async(exercise())


def test_main_and_subagent_permissions_share_one_queue(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            captured: dict[str, ConfirmationResult] = {}

            def subagent_worker() -> None:
                captured["subagent"] = app._permission_confirmer.confirm(
                    _permission_request()
                )

            main_result: list[ConfirmationResult] = []

            def main_worker() -> None:
                main_result.append(app._permission_confirmer.confirm(
                    _permission_request()
                ))

            main_thread = Thread(target=main_worker)
            main_thread.start()
            for _ in range(40):
                await pilot.pause(0.02)
                if isinstance(app.screen, PermissionScreen):
                    break
            assert isinstance(app.screen, PermissionScreen)

            sub_thread = Thread(target=subagent_worker)
            sub_thread.start()
            await pilot.pause()
            await pilot.pause()
            assert _permission_modal_count(app) == 1

            # Deny the main agent request; the queued SubAgent request shows next.
            modal = app.screen
            assert isinstance(modal, PermissionScreen)
            modal.query_one("#permission-deny").press()
            for _ in range(40):
                await pilot.pause(0.02)
                if (
                    isinstance(app.screen, PermissionScreen)
                    and app.screen is not modal
                    and not main_thread.is_alive()
                ):
                    break
            assert isinstance(app.screen, PermissionScreen)
            assert _permission_modal_count(app) == 1

            next_modal = app.screen
            next_modal.query_one("#permission-allow-once").press()
            for _ in range(40):
                await pilot.pause(0.02)
                if not main_thread.is_alive() and not sub_thread.is_alive():
                    break
            for _ in range(40):
                await pilot.pause(0.02)
                if _permission_modal_count(app) == 0:
                    break
            main_thread.join(1)
            sub_thread.join(1)
            assert main_result[0].status == "rejected"
            assert captured["subagent"].status == "approved"
            assert captured["subagent"].scope == "once"

    run_async(exercise())


def test_permission_queue_shutdown_releases_all_waiters(tmp_path) -> None:
    from mycode.presentation.tui.interactions import PermissionResponseHandle

    handles: list[PermissionResponseHandle] = []

    async def exercise() -> None:
        nonlocal handles
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            handles = [
                PermissionResponseHandle(_permission_request()) for _ in range(2)
            ]
            for handle in handles:
                app.post_message(PermissionRequestMessage(handle))
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, PermissionScreen)
            assert app._permission_queue
            app.exit()
        # App unmount runs during run_test teardown; waiters must be released.

    run_async(exercise())

    for handle in handles:
        assert handle._event.is_set()
        assert handle._result.status == "rejected"


def test_late_permission_request_after_shutdown_is_rejected_immediately(
    tmp_path,
) -> None:
    from mycode.presentation.tui.interactions import PermissionResponseHandle

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            app._shutdown_requested.set()

            handle = PermissionResponseHandle(_permission_request())
            app.post_message(PermissionRequestMessage(handle))
            await pilot.pause()
            await pilot.pause()

            assert handle._event.is_set()
            assert handle._result.status == "rejected"
            assert not isinstance(app.screen, PermissionScreen)
            assert app._permission_queue == []

    run_async(exercise())


def test_permission_escape_denies(tmp_path) -> None:
    from mycode.presentation.tui.interactions import PermissionResponseHandle

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)
            handle = PermissionResponseHandle(_permission_request())
            app.post_message(PermissionRequestMessage(handle))
            for _ in range(40):
                await pilot.pause(0.02)
                if isinstance(app.screen, PermissionScreen):
                    break
            assert isinstance(app.screen, PermissionScreen)

            await pilot.press("escape")
            for _ in range(40):
                await pilot.pause(0.02)
                if _permission_modal_count(app) == 0:
                    break
            assert _permission_modal_count(app) == 0
            assert handle._result.status == "rejected"
            assert handle._result.message == "Permission denied by user."

    run_async(exercise())


def test_mcp_trust_escape_rejects(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _wait_for_welcome(app, pilot)
            app.switch_screen(
                MCPTrustScreen(
                    MCPTrustRequest(
                        servers=(
                            MCPTrustServer(
                                alias="a",
                                transport="stdio",
                                command="mcp-server",
                                args=(),
                                env_keys=(),
                            ),
                        )
                    )
                )
            )
            await pilot.pause()
            assert isinstance(app.screen, MCPTrustScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, MCPTrustScreen)

    run_async(exercise())


def test_subagent_late_event_after_shutdown_is_safe(tmp_path) -> None:
    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main(app, pilot)

            # Late callbacks arriving after shutdown must not raise or post.
            app._subagent_observer.close()
            app._subagent_observer.on_state(
                _subagent_task(),
                _subagent_transition("running", "run_started"),
            )
            app._subagent_observer.on_result(
                _subagent_task(),
                _subagent_execution(_subagent_result("completed", "submitted")),
                datetime.now(UTC),
            )

            # Message handlers ignore SubAgent traffic once shutdown is requested.
            app._shutdown_requested.set()
            app.post_message(
                SubAgentStateMessage(_subagent_transition("running", "run_started"))
            )
            app.post_message(
                SubAgentResultMessage(
                    _subagent_task(),
                    _subagent_execution(_subagent_result("completed", "submitted")),
                )
            )
            await pilot.pause()
            await pilot.pause()

    run_async(exercise())


def test_session_title_refresh_reaches_header(monkeypatch, tmp_path) -> None:
    fake = _fake_session()
    record = SimpleNamespace(id="session-1")

    def fake_run_turn(content, *, turn_id=None, event_handler=None):
        record.title = "Auto generated title"
        outcome = AgentRunOutcome.from_stop_reason("final_answer")
        event_handler(
            RuntimeEvent(type="turn_finished", turn_id=turn_id, outcome=outcome)
        )
        return outcome

    fake.run_turn = fake_run_turn
    fake.active_project_session = SimpleNamespace(
        load_history=lambda: fake.history,
        record=record,
    )
    monkeypatch.setattr(
        tui_app, "start_agent_application_session", lambda *args, **kwargs: fake
    )

    async def exercise() -> None:
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _open_main_with_session(app, pilot)
            header = app.screen.query_one(HeaderBar)
            assert "Auto generated title" not in str(header.render())

            app.submit_user_message("first message")
            for _ in range(60):
                await pilot.pause(0.02)
                if "Auto generated title" in str(header.render()):
                    break
            assert "Auto generated title" in str(header.render())

    run_async(exercise())
