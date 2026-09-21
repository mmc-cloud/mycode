"""Textual application orchestration for MyCode's presentation layer."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

from textual import on
from textual.app import App
from textual.message import Message
from textual.widgets import OptionList

from mycode import startup_profile
from mycode.agent.outcome import AgentRunOutcome
from mycode.application.agent_session import (
    AgentApplicationSession,
    CompactResult,
    ContextStatus,
)
from mycode.application.startup import (
    ApplicationEnvironment,
    ApplicationSessionFactory,
    ApplicationStartupWarning,
    create_application_environment,
    prepare_application_session_factory,
    resolve_application_llm_config,
)
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import (
    SessionStartRequest,
    list_project_sessions,
)
from mycode.config import LLMConfig
from mycode.error_handling import error_summary
from mycode.mcp.config import MCPConfig
from mycode.permissions import ConfirmationResult
from mycode.persistence.session_store import (
    SessionInUseError,
    SessionRecord,
    SessionStore,
)
from mycode.presentation.commands import (
    CommandParseError,
    ParsedCommand,
    parse_slash_command,
)
from mycode.presentation.command_format import (
    format_command_help,
    format_compact_result,
    format_context_status,
    format_session_list,
)
from mycode.presentation.tui.interactions import (
    MCPTrustRequestMessage,
    MCPTrustWarningMessage,
    PermissionRequestMessage,
    PermissionResponseHandle,
    SubAgentResultMessage,
    SubAgentStateMessage,
    TuiConfirmer,
    TuiMCPTrustConfirmer,
    TuiSubAgentObserver,
)
from mycode.presentation.tui.presenter import TuiPresenter
from mycode.presentation.tui.screens import (
    LoadingScreen,
    MCPTrustScreen,
    MainScreen,
    PermissionScreen,
    WelcomeScreen,
)
from mycode.presentation.tui.widgets import (
    SPLASH_LOGO,
    CommandPicker,
    ConversationView,
    HeaderBar,
    PromptTextArea,
    StatusBar,
)


HistoryItem = tuple[str, str]


def _try_close_application_session(
    application_session: AgentApplicationSession,
) -> str:
    """Close a TUI-owned session without breaking presentation cleanup."""
    try:
        application_session.close()
    except Exception as caught:  # noqa: BLE001 - presentation cleanup boundary
        return error_summary(caught)
    return ""


class WelcomeMetadataMessage(Message):
    def __init__(
        self,
        *,
        generation: int,
        llm_config: LLMConfig | None,
        sessions: tuple[SessionRecord, ...],
        error: str = "",
    ) -> None:
        self.generation = generation
        self.llm_config = llm_config
        self.sessions = sessions
        self.error = error
        super().__init__()


class StartupProgressMessage(Message):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__()


class StartupEventMessage(Message):
    def __init__(self, event: RuntimeEvent) -> None:
        self.event = event
        super().__init__()


class StartupSucceededMessage(Message):
    def __init__(
        self,
        application_session: AgentApplicationSession,
        history: tuple[HistoryItem, ...],
        *,
        replace_current: bool = False,
    ) -> None:
        self.application_session = application_session
        self.history = history
        self.replace_current = replace_current
        super().__init__()


class StartupFailedMessage(Message):
    def __init__(
        self,
        error: BaseException,
        *,
        session_in_use: bool = False,
        replace_current: bool = False,
        cleanup_warning: str = "",
    ) -> None:
        self.error = error
        self.session_in_use = session_in_use
        self.replace_current = replace_current
        self.cleanup_warning = cleanup_warning
        super().__init__()


class SessionListMessage(Message):
    def __init__(
        self,
        sessions: tuple[SessionRecord, ...],
        *,
        error: str = "",
    ) -> None:
        self.sessions = sessions
        self.error = error
        super().__init__()


class ContextStatusMessage(Message):
    def __init__(
        self,
        status: ContextStatus | None = None,
        *,
        error: str = "",
    ) -> None:
        self.status = status
        self.error = error
        super().__init__()


class CompactResultMessage(Message):
    def __init__(
        self,
        result: CompactResult | None = None,
        *,
        error: str = "",
    ) -> None:
        self.result = result
        self.error = error
        super().__init__()


class TurnRuntimeEventMessage(Message):
    def __init__(self, event: RuntimeEvent) -> None:
        self.event = event
        super().__init__()


class TurnCompletedMessage(Message):
    def __init__(self, turn_id: str, outcome: AgentRunOutcome) -> None:
        self.turn_id = turn_id
        self.outcome = outcome
        super().__init__()


class TurnFailedMessage(Message):
    def __init__(
        self,
        turn_id: str,
        error: BaseException,
        interrupt_error: BaseException | None = None,
    ) -> None:
        self.turn_id = turn_id
        self.error = error
        self.interrupt_error = interrupt_error
        super().__init__()


class MyCodeTuiApp(App[None]):
    """Welcome, session startup, and the Textual Agent presentation shell."""

    TITLE = "MyCode"
    CSS = """
    Screen {
        background: $surface;
    }

    #welcome-content, #loading-content {
        align: center middle;
        height: 100%;
        width: 100%;
    }

    #welcome-logo, #loading-logo {
        color: $text;
        text-align: center;
        width: auto;
    }

    #welcome-meta, #loading-meta, #loading-session {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
        width: auto;
    }

    #welcome-error {
        color: $error;
        margin-top: 1;
        width: 60%;
    }

    #welcome-notice, #welcome-help, #loading-status {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
        width: auto;
    }

    #session-options {
        height: auto;
        max-height: 12;
        margin-top: 1;
        min-width: 60;
        width: 60%;
    }

    #loading-indicator {
        margin-top: 1;
        width: 5;
    }

    #mcp-trust-dialog {
        align: center middle;
        background: $panel;
        border: round $primary;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        width: 70%;
    }

    #mcp-trust-details {
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
    }

    #mcp-trust-actions {
        align: center middle;
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }

    #mcp-trust-actions Button {
        margin: 0 1;
    }

    #permission-dialog {
        align: center middle;
        background: $panel;
        border: round $primary;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        width: 70%;
    }

    #permission-details {
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
    }

    #permission-actions {
        align: center middle;
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }

    #permission-actions Button {
        margin: 0 1;
    }

    #header {
        background: $panel;
        color: $text;
        height: 3;
        padding: 1 2;
    }

    #conversation {
        border: round $primary-darken-2;
        height: 1fr;
        margin: 1 2;
        padding: 1 2;
    }

    #status {
        background: $panel;
        color: $text-muted;
        height: 1;
        padding: 0 2;
    }

    #command-picker {
        background: $panel;
        border: round $primary-darken-2;
        height: auto;
        margin: 0 2;
        padding: 0 1;
    }

    #prompt {
        height: 7;
        margin: 0 2 1 2;
    }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        *,
        workspace_path: Path | None = None,
        session_store: SessionStore | None = None,
        llm_config: LLMConfig | None = None,
        mcp_config: MCPConfig | None = None,
        trust_file: str | Path | None = None,
        # Kept as a compatibility keyword for callers of the 14.6.1 shell.
        splash_duration: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        del splash_duration
        self._environment: ApplicationEnvironment = create_application_environment(
            workspace_path,
            session_store=session_store,
        )
        self.workspace = self._environment.workspace
        self.project = self._environment.project
        self.session_store = self._environment.session_store
        self._llm_config = llm_config
        self._mcp_config_override = mcp_config
        self._trust_file = trust_file
        self._trust_confirmer = TuiMCPTrustConfirmer(self.post_message)
        self._permission_confirmer = TuiConfirmer(self.post_message)
        self._subagent_observer = TuiSubAgentObserver(self.post_message)

        self.presenter: TuiPresenter | None = None
        self._queued_events: list[RuntimeEvent] = []
        self._session_records: tuple[SessionRecord, ...] = ()
        self._metadata_ready = False
        self._metadata_generation = 0
        self._startup_active = False
        self._shutdown_requested = Event()
        self._session_lock = Lock()
        self._session_factory: ApplicationSessionFactory | None = None
        self._application_session: AgentApplicationSession | None = None
        self._pending_application_sessions: list[AgentApplicationSession] = []
        self._turn_session_owned_by_worker = False
        self._active_turn_id: str | None = None
        self._command_worker_active = False
        self._session_unusable = False
        self._session_switch_active = False
        self._session_switch_events: list[RuntimeEvent] = []
        self._command_session_owned_by_worker: AgentApplicationSession | None = None
        self._status_bar: StatusBar | None = None
        self._permission_queue: list[PermissionResponseHandle] = []
        self._active_permission_handle: PermissionResponseHandle | None = None
        self._status_before_permission: str | None = None

    @property
    def workspace_label(self) -> str:
        return self.workspace.root.name or str(self.workspace.root)

    @property
    def model_label(self) -> str:
        return "—" if self._llm_config is None else self._llm_config.model

    def on_mount(self) -> None:
        self.push_screen(
            WelcomeScreen(
                workspace=self.workspace_label,
                model=self.model_label,
                enabled=False,
            )
        )
        self._start_welcome_metadata_load(name="welcome-metadata")

    def _start_welcome_metadata_load(self, *, name: str) -> None:
        self._metadata_generation += 1
        generation = self._metadata_generation
        self.run_worker(
            lambda: self._load_welcome_metadata(generation),
            name=name,
            group="startup-refresh",
            thread=True,
            exit_on_error=False,
        )

    def _load_welcome_metadata(self, generation: int) -> None:
        config = self._llm_config
        error = ""
        try:
            config = resolve_application_llm_config(
                self._environment,
                llm_config=config,
            )
        except Exception as caught:  # noqa: BLE001 - UI boundary reports a summary
            error = error_summary(caught)

        try:
            sessions = tuple(list_project_sessions(self.session_store, self.project))
        except Exception as caught:  # noqa: BLE001 - UI boundary reports a summary
            sessions = ()
            error = error or error_summary(caught)

        self.post_message(
            WelcomeMetadataMessage(
                generation=generation,
                llm_config=config,
                sessions=sessions,
                error=error,
            )
        )

    @on(WelcomeMetadataMessage)
    def _on_welcome_metadata(self, message: WelcomeMetadataMessage) -> None:
        if message.generation != self._metadata_generation:
            return
        if message.llm_config is not None:
            self._llm_config = message.llm_config
        self._session_records = message.sessions
        self._metadata_ready = True
        if self._startup_active or not isinstance(self.screen, WelcomeScreen):
            return
        notice = ""
        error = message.error
        notice = self.screen.notice
        error = self.screen.error or error
        self._show_welcome(notice=notice, error=error)

    @on(OptionList.OptionSelected)
    def _on_session_selected(self, event: OptionList.OptionSelected) -> None:
        if self._startup_active:
            return
        if not isinstance(self.screen, WelcomeScreen):
            return
        request = WelcomeScreen.request_from_option(event.option.id)
        if request is None or self._llm_config is None:
            self._show_welcome(error="LLM configuration is not available yet.")
            return
        self._begin_startup(request)

    def _begin_startup(self, request: SessionStartRequest) -> None:
        # Invalidate any refresh result that was started before this startup.
        self._metadata_generation += 1
        self._startup_active = True
        self._session_switch_active = False
        self._session_switch_events = []
        self.switch_screen(
            LoadingScreen(
                workspace=self.workspace_label,
                model=self.model_label,
                request_label=_request_label(request),
            )
        )
        self.run_worker(
            lambda: self._startup_worker(request),
            name="session-startup",
            group="startup",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _begin_session_switch(self, request: SessionStartRequest) -> None:
        if self._startup_active or self._active_turn_id is not None:
            return
        if self._llm_config is None:
            self._show_command_notice(
                "Session commands are unavailable until runtime startup completes.",
                level="warning",
            )
            return
        self._metadata_generation += 1
        self._startup_active = True
        self._session_switch_active = True
        self._session_switch_events = []
        self._set_prompt_enabled(False)
        self._set_status("Opening session…")
        self.run_worker(
            lambda: self._startup_worker(request, replace_current=True),
            name="session-switch",
            group="startup",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _startup_worker(
        self,
        request: SessionStartRequest,
        *,
        replace_current: bool = False,
    ) -> None:
        application_session: AgentApplicationSession | None = None
        try:
            if self._shutdown_requested.is_set():
                return
            if self._session_factory is None:
                self._post_progress("Loading project configuration...")
                config = resolve_application_llm_config(
                    self._environment,
                    llm_config=self._llm_config,
                )
                self._llm_config = config
                self._post_progress("Checking MCP trust...")
                self._session_factory = prepare_application_session_factory(
                    self._environment,
                    llm_config=config,
                    mcp_config=self._mcp_config_override,
                    mcp_trust_confirmer=self._trust_confirmer,
                    trust_file=self._trust_file,
                    confirmer=self._permission_confirmer,
                    external_observer=self._subagent_observer,
                    warning_handler=self._handle_startup_warning,
                )

            self._post_progress("Connecting MCP...")
            self._post_progress("Starting runtime...")
            if self._shutdown_requested.is_set():
                return
            application_session = self._session_factory.open_session(request)
            startup_profile.total("runtime.ready")
            if not self._register_pending_application_session(application_session):
                return
            history = _visible_history(
                application_session.active_project_session.load_history()
            )

            for event in application_session.startup_events():
                self.post_message(StartupEventMessage(event))
            self.post_message(
                StartupSucceededMessage(
                    application_session,
                    history,
                    replace_current=replace_current,
                )
            )
        except SessionInUseError as caught:
            cleanup_warning = ""
            if application_session is not None:
                cleanup_warning = self._cleanup_startup_session(application_session)
            self.post_message(
                StartupFailedMessage(
                    caught,
                    session_in_use=True,
                    replace_current=replace_current,
                    cleanup_warning=cleanup_warning,
                )
            )
        except Exception as caught:  # noqa: BLE001 - worker boundary returns to Welcome
            cleanup_warning = ""
            if application_session is not None:
                cleanup_warning = self._cleanup_startup_session(application_session)
            self.post_message(
                StartupFailedMessage(
                    caught,
                    replace_current=replace_current,
                    cleanup_warning=cleanup_warning,
                )
            )

    def _handle_startup_warning(self, warning: ApplicationStartupWarning) -> None:
        if warning.code == "mcp_config_error":
            self._post_progress(f"MCP config unavailable: {warning.message}")

    def _register_pending_application_session(
        self,
        application_session: AgentApplicationSession,
    ) -> bool:
        with self._session_lock:
            if self._shutdown_requested.is_set():
                should_close = True
            else:
                self._pending_application_sessions.append(application_session)
                should_close = False
        if should_close:
            _try_close_application_session(application_session)
            return False
        return True

    def _cleanup_startup_session(
        self,
        application_session: AgentApplicationSession,
    ) -> str:
        with self._session_lock:
            try:
                self._pending_application_sessions.remove(application_session)
            except ValueError:
                claimed = False
            else:
                claimed = True
        if not claimed:
            return ""
        return _try_close_application_session(application_session)

    def _post_progress(self, value: str) -> None:
        self.post_message(StartupProgressMessage(value))

    @on(StartupProgressMessage)
    def _on_startup_progress(self, message: StartupProgressMessage) -> None:
        if isinstance(self.screen, LoadingScreen):
            self.screen.set_progress(message.value)
        elif self._session_switch_active and isinstance(self.screen, MainScreen):
            self._set_status(message.value)

    @on(MCPTrustRequestMessage)
    def _on_mcp_trust_request(self, message: MCPTrustRequestMessage) -> None:
        if self._shutdown_requested.is_set():
            message.handle.resolve(False)
            return
        self.push_screen(
            MCPTrustScreen(message.handle.request),
            lambda approved: message.handle.resolve(bool(approved)),
        )

    @on(MCPTrustWarningMessage)
    def _on_mcp_trust_warning(self, message: MCPTrustWarningMessage) -> None:
        self._post_progress(f"Warning: {message.warning.message}")

    @on(PermissionRequestMessage)
    def _on_permission_request(self, message: PermissionRequestMessage) -> None:
        if self._shutdown_requested.is_set() or self._session_unusable:
            message.handle.resolve(
                ConfirmationResult.rejected(
                    message="Permission confirmation unavailable."
                )
            )
            return
        self._permission_queue.append(message.handle)
        self._maybe_show_next_permission()

    def _maybe_show_next_permission(self) -> None:
        if self._active_permission_handle is not None:
            return
        if self._shutdown_requested.is_set() or self._session_unusable:
            self._drain_permission_queue_rejected()
            return
        if not self._permission_queue:
            self._restore_status_after_permission()
            return
        handle = self._permission_queue.pop(0)
        self._active_permission_handle = handle
        if self._status_before_permission is None:
            self._status_before_permission = self._current_status_text()
        self._set_status("Waiting for permission…")
        self.push_screen(
            PermissionScreen(handle.request),
            self._resolve_active_permission,
        )

    def _resolve_active_permission(self, result: ConfirmationResult) -> None:
        handle = self._active_permission_handle
        self._active_permission_handle = None
        if handle is not None:
            handle.resolve(result)
        self._maybe_show_next_permission()

    def _drain_permission_queue_rejected(self) -> None:
        queue = self._permission_queue
        self._permission_queue = []
        active = self._active_permission_handle
        self._active_permission_handle = None
        for handle in (*queue, active):
            if handle is None:
                continue
            handle.resolve(
                ConfirmationResult.rejected(
                    message="Permission confirmation unavailable."
                )
            )
        self._restore_status_after_permission()

    def _restore_status_after_permission(self) -> None:
        previous = self._status_before_permission
        self._status_before_permission = None
        if previous is not None:
            self._set_status(previous)

    def _current_status_text(self) -> str:
        try:
            if isinstance(self.screen, MainScreen):
                return str(self.screen.query_one(StatusBar).render())
            if self._status_bar is not None:
                return str(self._status_bar.render())
        except Exception:  # noqa: BLE001 - snapshot is best-effort
            pass
        return "Ready"

    def _set_status(self, value: str) -> None:
        try:
            if isinstance(self.screen, MainScreen):
                self.screen.query_one(StatusBar).set_status(value)
                return
            if self._status_bar is not None:
                self._status_bar.set_status(value)
        except Exception:  # noqa: BLE001 - status update is best-effort
            pass

    @on(StartupEventMessage)
    def _on_startup_event(self, message: StartupEventMessage) -> None:
        if self._session_switch_active:
            self._session_switch_events.append(message.event)
            return
        self.present_event(message.event)

    @on(SessionListMessage)
    def _on_session_list(self, message: SessionListMessage) -> None:
        if self._shutdown_requested.is_set() or not isinstance(self.screen, MainScreen):
            return
        self._command_worker_active = False
        if message.error:
            self._show_command_notice(
                f"/sessions failed: {message.error}",
                level="error",
            )
            self._set_prompt_enabled(True)
            self._set_status("Ready")
            self._focus_prompt()
            return

        self._session_records = message.sessions
        self._show_sessions(message.sessions)
        self._set_prompt_enabled(True)
        self._set_status("Ready")
        self._focus_prompt()

    @on(ContextStatusMessage)
    def _on_context_status(self, message: ContextStatusMessage) -> None:
        if self._shutdown_requested.is_set() or not isinstance(self.screen, MainScreen):
            return
        self._command_worker_active = False
        if message.error or message.status is None:
            self._show_command_notice(
                f"/context failed: {message.error or 'unavailable'}",
                level="error",
            )
        else:
            self._show_context_status(message.status)
        self._set_prompt_enabled(True)
        self._set_status("Ready")
        self._focus_prompt()

    @on(CompactResultMessage)
    def _on_compact_result(self, message: CompactResultMessage) -> None:
        if self._shutdown_requested.is_set() or not isinstance(self.screen, MainScreen):
            return
        self._command_worker_active = False
        if message.error or message.result is None:
            self._show_command_notice(
                f"/compact failed: {message.error or 'unavailable'}",
                level="error",
            )
        else:
            result = message.result
            level = "info"
            if result.status == "skipped":
                level = "warning"
            elif result.status == "failed":
                level = "error"
            self._show_command_notice(
                format_compact_result(result),
                level=level,
            )
        self._set_prompt_enabled(True)
        self._set_status("Ready")
        self._focus_prompt()

    @on(StartupSucceededMessage)
    def _on_startup_succeeded(self, message: StartupSucceededMessage) -> None:
        old_session: AgentApplicationSession | None = None
        switch_events: tuple[RuntimeEvent, ...] = ()
        with self._session_lock:
            try:
                self._pending_application_sessions.remove(message.application_session)
            except ValueError:
                claimed = False
            else:
                claimed = True
            if not claimed:
                should_close = False
            elif self._shutdown_requested.is_set():
                should_close = True
            elif message.replace_current:
                old_session = self._application_session
                self._application_session = message.application_session
                self._turn_session_owned_by_worker = False
                self._session_unusable = False
                self._session_switch_active = False
                self._startup_active = False
                switch_events = tuple(self._session_switch_events)
                self._session_switch_events = []
                should_close = False
            else:
                self._application_session = message.application_session
                self._turn_session_owned_by_worker = False
                should_close = False
        if not claimed:
            return
        if should_close:
            _try_close_application_session(message.application_session)
            return

        if message.replace_current:
            close_warning = (
                _try_close_application_session(old_session)
                if old_session is not None
                else ""
            )
            self._queued_events.extend(switch_events)
            self.switch_screen(
                MainScreen(
                    workspace=self.workspace_label,
                    model=self.model_label,
                    history=message.history,
                )
            )
            if close_warning:
                self.call_after_refresh(
                    lambda: self._show_command_notice(
                        f"Old session cleanup warning: {close_warning}",
                        level="warning",
                    )
                )
            return

        self._startup_active = False
        self.switch_screen(
            MainScreen(
                workspace=self.workspace_label,
                model=self.model_label,
                history=message.history,
            )
        )

    @on(StartupFailedMessage)
    def _on_startup_failed(self, message: StartupFailedMessage) -> None:
        if message.replace_current:
            self._startup_active = False
            self._session_switch_active = False
            self._session_switch_events = []
            if message.session_in_use:
                detail = "selected session is currently in use"
            else:
                detail = error_summary(message.error)
            self._show_command_notice(
                f"Session switch failed: {detail}",
                level="warning" if message.session_in_use else "error",
            )
            if message.cleanup_warning:
                self._show_command_notice(
                    f"Session cleanup warning: {message.cleanup_warning}",
                    level="warning",
                )
            self._set_prompt_enabled(True)
            self._set_status("Ready")
            self._focus_prompt()
            self._start_welcome_metadata_load(name="refresh-session-list")
            return

        self._startup_active = False
        if message.session_in_use:
            notice = "Selected session is currently in use. Session list refreshed."
            error = ""
        else:
            notice = "Startup failed. Choose a session to try again."
            error = f"{type(message.error).__name__}: {error_summary(message.error)}"
        if message.cleanup_warning:
            cleanup_notice = f"Session cleanup warning: {message.cleanup_warning}"
            if error:
                error += f"\n⚠ {cleanup_notice}"
            else:
                notice += f" {cleanup_notice}"
        self._show_welcome(notice=notice, error=error)
        self._start_welcome_metadata_load(name="refresh-session-list")

    @on(TurnRuntimeEventMessage)
    def _on_turn_runtime_event(self, message: TurnRuntimeEventMessage) -> None:
        if message.event.turn_id != self._active_turn_id:
            return
        self.present_event(message.event)

    @on(TurnCompletedMessage)
    def _on_turn_completed(self, message: TurnCompletedMessage) -> None:
        if message.turn_id != self._active_turn_id:
            return
        self._active_turn_id = None
        self._set_prompt_enabled(True)
        self._set_status("Ready")
        self._refresh_session_title()
        self._focus_prompt()

    @on(SubAgentStateMessage)
    def _on_subagent_state(self, message: SubAgentStateMessage) -> None:
        if self._shutdown_requested.is_set() or self.presenter is None:
            return
        self.presenter.present_subagent_transition(message.transition)

    @on(SubAgentResultMessage)
    def _on_subagent_result(self, message: SubAgentResultMessage) -> None:
        if self._shutdown_requested.is_set() or self.presenter is None:
            return
        self.presenter.present_subagent_result(message.execution)

    def _refresh_session_title(self) -> None:
        session = self._application_session
        if session is None or self.presenter is None:
            return
        record = getattr(session.active_project_session, "record", None)
        title = getattr(record, "title", None)
        if not title:
            return
        self.presenter.header.set_session(str(title))

    @on(TurnFailedMessage)
    def _on_turn_failed(self, message: TurnFailedMessage) -> None:
        if message.turn_id != self._active_turn_id:
            return
        self._active_turn_id = None
        self._session_unusable = True
        self._set_prompt_enabled(False)
        if self.presenter is not None:
            self.presenter.conversation.add_notice(
                f"✗ fatal agent error: {error_summary(message.error)}",
                level="error",
            )
            if message.interrupt_error is not None:
                self.presenter.conversation.add_notice(
                    "✗ session interrupt failed: "
                    f"{error_summary(message.interrupt_error)}",
                    level="error",
                )
        self._set_status("Fatal error")

    def _show_welcome(self, *, notice: str = "", error: str = "") -> None:
        self.switch_screen(
            WelcomeScreen(
                workspace=self.workspace_label,
                model=self.model_label,
                sessions=self._session_records,
                enabled=self._metadata_ready,
                notice=notice,
                error=error,
            )
        )

    def _handle_command(self, command: ParsedCommand) -> None:
        if command.name == "help":
            self._show_command_help()
            return
        if command.name == "sessions":
            self._begin_session_list()
            return
        if command.name == "new":
            self._begin_session_switch(SessionStartRequest(mode="new"))
            return
        if command.name == "resume":
            target_session_id = command.args[0]
            if target_session_id == self._current_session_id():
                self._show_command_notice(
                    "already using current session",
                    level="info",
                )
                return
            self._begin_session_switch(
                SessionStartRequest(mode="resume", session_id=target_session_id)
            )
            return
        if command.name == "exit":
            self.action_quit()
            return
        if command.name == "context":
            self._begin_context_status()
            return
        if command.name == "compact":
            self._begin_compact()
            return
        self._show_command_notice(
            f"/{command.name} is not available yet.",
            level="info",
        )

    def _show_command_help(self) -> None:
        for line in format_command_help():
            self._show_command_notice(line)

    def _begin_session_list(self) -> None:
        if self._startup_active or self._command_worker_active:
            return
        self._command_worker_active = True
        self._set_prompt_enabled(False)
        self._set_status("Loading sessions…")
        self.run_worker(
            self._session_list_worker,
            name="session-list",
            group="session-command",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _begin_context_status(self) -> None:
        if self._startup_active or self._command_worker_active:
            return
        self._command_worker_active = True
        self._set_prompt_enabled(False)
        self._set_status("Loading context…")
        self.run_worker(
            self._context_status_worker,
            name="context-status",
            group="session-command",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _context_status_worker(self) -> None:
        application_session = self._claim_command_session_for_worker()
        if application_session is None:
            self.post_message(ContextStatusMessage(error="runtime unavailable"))
            return
        try:
            message = ContextStatusMessage(application_session.get_context_status())
        except Exception as caught:  # noqa: BLE001 - UI boundary reports a summary
            message = ContextStatusMessage(error=error_summary(caught))
        if self._release_command_session_after_worker(application_session):
            _try_close_application_session(application_session)
            return
        self.post_message(message)

    def _begin_compact(self) -> None:
        if self._startup_active or self._command_worker_active:
            return
        self._command_worker_active = True
        self._set_prompt_enabled(False)
        self._set_status("Compacting…")
        self.run_worker(
            self._compact_worker,
            name="compact-context",
            group="session-command",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _compact_worker(self) -> None:
        application_session = self._claim_command_session_for_worker()
        if application_session is None:
            self.post_message(CompactResultMessage(error="runtime unavailable"))
            return
        try:
            message = CompactResultMessage(application_session.compact_context())
        except Exception as caught:  # noqa: BLE001 - UI boundary reports a summary
            message = CompactResultMessage(error=error_summary(caught))
        if self._release_command_session_after_worker(application_session):
            _try_close_application_session(application_session)
            return
        self.post_message(message)

    def _session_list_worker(self) -> None:
        try:
            sessions = tuple(list_project_sessions(self.session_store, self.project))
            message = SessionListMessage(sessions)
        except Exception as caught:  # noqa: BLE001 - UI boundary reports a summary
            message = SessionListMessage((), error=error_summary(caught))
        self.post_message(message)

    def _show_sessions(self, sessions: tuple[SessionRecord, ...]) -> None:
        current_session_id = self._current_session_id()
        for line in format_session_list(
            sessions,
            current_session_id=current_session_id or "",
        ):
            self._show_command_notice(line)

    def _show_context_status(self, status: ContextStatus) -> None:
        for line in format_context_status(status):
            self._show_command_notice(line)

    def _show_command_notice(self, content: str, *, level: str = "info") -> None:
        if self.presenter is not None:
            self.presenter.conversation.add_notice(
                f"command> {content}",
                level=level,
            )

    def _current_session_id(self) -> str | None:
        with self._session_lock:
            session = self._application_session
        if session is None:
            return None
        record = getattr(session.active_project_session, "record", None)
        session_id = getattr(record, "id", None)
        return session_id if isinstance(session_id, str) else None

    def _activate_main_screen(self, screen: MainScreen) -> None:
        conversation = screen.query_one(ConversationView)
        self._status_bar = screen.query_one(StatusBar)
        self.presenter = TuiPresenter(
            conversation=conversation,
            header=screen.query_one(HeaderBar),
            status=screen.query_one(StatusBar),
        )
        for role, content in screen.history:
            conversation.add_history_message(role, content)
        queued_events = self._queued_events
        self._queued_events = []
        for runtime_event in queued_events:
            self.presenter.present(runtime_event)

    def present_event(self, event: RuntimeEvent) -> None:
        if self.presenter is None:
            self._queued_events.append(event)
            return
        self.presenter.present(event)

    def submit_user_message(self, content: str) -> None:
        if not content.strip() or self.presenter is None or self._session_unusable:
            return
        try:
            command = parse_slash_command(content)
        except CommandParseError as error:
            self._show_command_notice(str(error), level="error")
            return
        if command is not None:
            if self._active_turn_id is not None:
                if command.name == "exit":
                    self.action_quit()
                else:
                    self._show_command_notice(
                        "Agent turn is running; wait for completion before using commands.",
                        level="warning",
                    )
                return
            if command.name != "exit" and self._startup_active:
                self._show_command_notice(
                    "Session startup is running; wait for it to finish.",
                    level="warning",
                )
                return
            if command.name != "exit" and self._command_worker_active:
                self._show_command_notice(
                    "Command is running; wait for it to finish.",
                    level="warning",
                )
                return
            self._handle_command(command)
            return
        if (
            self._active_turn_id is not None
            or self._startup_active
            or self._command_worker_active
        ):
            return
        self.presenter.show_user_message(content)
        if self._application_session is None:
            return
        turn_id = uuid4().hex
        self._active_turn_id = turn_id
        self._set_prompt_enabled(False)
        self._set_status("Thinking…")
        self.run_worker(
            lambda: self._turn_worker(content, turn_id),
            name=f"agent-turn-{turn_id[:8]}",
            group="agent-turn",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _turn_worker(self, content: str, turn_id: str) -> None:
        application_session = self._claim_turn_session_for_worker()
        if application_session is None:
            return

        def handle_event(event: RuntimeEvent) -> None:
            self.post_message(TurnRuntimeEventMessage(event))

        try:
            outcome = application_session.run_turn(
                content,
                turn_id=turn_id,
                event_handler=handle_event,
            )
        except Exception as caught:  # noqa: BLE001 - worker boundary reports fatal turns
            interrupt_error: BaseException | None = None
            try:
                application_session.interrupt()
            except BaseException as interrupt_caught:  # noqa: BLE001 - preserve UI recovery
                interrupt_error = interrupt_caught
            if self._release_turn_session_after_turn(application_session):
                _try_close_application_session(application_session)
                return
            self.post_message(TurnFailedMessage(turn_id, caught, interrupt_error))
            return
        if self._release_turn_session_after_turn(application_session):
            _try_close_application_session(application_session)
            return
        self.post_message(TurnCompletedMessage(turn_id, outcome))

    def _claim_command_session_for_worker(
        self,
    ) -> AgentApplicationSession | None:
        with self._session_lock:
            if (
                self._shutdown_requested.is_set()
                or self._application_session is None
                or self._turn_session_owned_by_worker
                or self._command_session_owned_by_worker is not None
            ):
                return None
            self._command_session_owned_by_worker = self._application_session
            return self._application_session

    def _release_command_session_after_worker(
        self,
        application_session: AgentApplicationSession,
    ) -> bool:
        """Release command ownership, returning whether the worker must close."""
        with self._session_lock:
            if self._command_session_owned_by_worker is not application_session:
                return False
            self._command_session_owned_by_worker = None
            if self._shutdown_requested.is_set():
                if self._application_session is application_session:
                    self._application_session = None
                return True
            return False

    def _claim_turn_session_for_worker(
        self,
    ) -> AgentApplicationSession | None:
        with self._session_lock:
            if (
                self._shutdown_requested.is_set()
                or self._application_session is None
                or self._turn_session_owned_by_worker
            ):
                return None
            self._turn_session_owned_by_worker = True
            return self._application_session

    def _release_turn_session_after_turn(
        self,
        application_session: AgentApplicationSession,
    ) -> bool:
        """Release a worker-owned session, or keep cleanup in the worker.

        Returns whether the worker must close the session. The lock makes the
        release-to-UI and shutdown decision one atomic ownership transition.
        """
        with self._session_lock:
            if (
                self._application_session is not application_session
                or not self._turn_session_owned_by_worker
            ):
                return False
            if self._shutdown_requested.is_set():
                self._application_session = None
                self._turn_session_owned_by_worker = False
                return True
            self._turn_session_owned_by_worker = False
            return False

    def _set_prompt_enabled(self, enabled: bool) -> None:
        if not isinstance(self.screen, MainScreen):
            return
        self.screen.query_one(PromptTextArea).disabled = not enabled

    def _focus_prompt(self) -> None:
        if self._session_unusable or not isinstance(self.screen, MainScreen):
            return
        self.set_focus(self.screen.query_one(PromptTextArea))

    def action_quit(self) -> None:
        if self._active_turn_id is not None:
            if self.presenter is not None:
                self.presenter.conversation.add_notice(
                    "⚠ Agent turn is running; wait for completion before quitting.",
                    level="warning",
                )
            self._set_status("Running…")
            return
        if self._command_worker_active:
            if self.presenter is not None:
                self.presenter.conversation.add_notice(
                    "⚠ Command is running; wait for it to finish before quitting.",
                    level="warning",
                )
            self._set_status("Running…")
            return
        self.exit()

    def on_unmount(self) -> None:
        self._shutdown_requested.set()
        self._trust_confirmer.reject_all()
        self._permission_confirmer.reject_all()
        self._subagent_observer.close()
        self._drain_permission_queue_rejected()
        with self._session_lock:
            sessions = list(self._pending_application_sessions)
            self._pending_application_sessions.clear()
            if (
                self._application_session is not None
                and not self._turn_session_owned_by_worker
                and self._command_session_owned_by_worker is None
            ):
                sessions.append(self._application_session)
                self._application_session = None
        for application_session in sessions:
            _try_close_application_session(application_session)


def _request_label(request: SessionStartRequest) -> str:
    if request.mode == "resume":
        identifier = request.session_id or ""
        return f"resume {identifier[:8]}"
    return request.mode


def _visible_history(conversation) -> tuple[HistoryItem, ...]:
    return tuple(
        (message.role, message.content)
        for message in conversation.get_messages()
        if message.role in {"user", "assistant"}
    )


def run_tui() -> None:
    MyCodeTuiApp().run()


__all__ = [
    "CommandPicker",
    "ConversationView",
    "HeaderBar",
    "MainScreen",
    "MyCodeTuiApp",
    "PromptTextArea",
    "SPLASH_LOGO",
    "StatusBar",
    "run_tui",
]
