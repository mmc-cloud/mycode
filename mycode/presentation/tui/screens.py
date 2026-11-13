"""Textual screens for the MyCode welcome and startup flow."""

from __future__ import annotations

from collections.abc import Sequence

from textual.css.query import NoMatches
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Paste
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, LoadingIndicator, OptionList, Static, Input, Select
from textual.widgets.option_list import Option

from mycode.application.sessions import SessionStartRequest
from mycode.application.agent_session import ContextStatus
from mycode.mcp.trust import MCPTrustRequest
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.persistence.session_store import SessionRecord
from mycode.presentation.tui.widgets import (
    SPLASH_LOGO,
    CommandPicker,
    ConversationView,
    HeaderBar,
    PromptTextArea,
    StatusBar,
)


def _session_options(sessions: Sequence[SessionRecord]) -> list[Option]:
    options = [
        Option("Continue latest", id="continue"),
        Option("New session", id="new"),
    ]
    options.extend(
        Option(
            f"Resume: {session.title}    {session.id[:8]}",
            id=f"resume:{session.id}",
        )
        for session in sessions
    )
    return options


class WelcomeScreen(Screen[None]):
    def __init__(
        self,
        *,
        workspace: str,
        model: str,
        sessions: Sequence[SessionRecord] = (),
        enabled: bool = False,
        notice: str = "",
        error: str = "",
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.model = model
        self.sessions = tuple(sessions)
        self.enabled = enabled
        self.notice = notice
        self.error = error

    def compose(self) -> ComposeResult:
        with Vertical(id="welcome-content"):
            yield Static(SPLASH_LOGO, id="welcome-logo", markup=False)
            yield Static(
                f"Workspace  {self.workspace}\nModel      {self.model}",
                id="welcome-meta",
                markup=False,
            )
            if self.error:
                yield Static(f"⚠ {self.error}", id="welcome-error")
            elif self.notice:
                yield Static(self.notice, id="welcome-notice")
            yield OptionList(
                *_session_options(self.sessions),
                id="session-options",
                disabled=not self.enabled,
            )
            yield Static(
                "↑ ↓ select        Enter confirm",
                id="welcome-help",
            )

    def on_mount(self) -> None:
        if self.enabled:
            self.call_after_refresh(self._focus_options)

    def _focus_options(self) -> None:
        if not self.is_mounted or self.app.screen is not self:
            return
        try:
            self.query_one(OptionList).focus()
        except NoMatches:
            return

    @staticmethod
    def request_from_option(option_id: str | None) -> SessionStartRequest | None:
        if option_id == "new":
            return SessionStartRequest(mode="new")
        if option_id == "continue":
            return SessionStartRequest(mode="continue")
        if option_id is not None and option_id.startswith("resume:"):
            return SessionStartRequest(
                mode="resume",
                session_id=option_id.removeprefix("resume:"),
            )
        return None


class LoadingScreen(Screen[None]):
    def __init__(self, *, workspace: str, model: str, request_label: str) -> None:
        super().__init__()
        self.workspace = workspace
        self.model = model
        self.request_label = request_label
        self.progress = "Opening session..."

    def compose(self) -> ComposeResult:
        with Vertical(id="loading-content"):
            yield Static(SPLASH_LOGO, id="loading-logo", markup=False)
            yield Static(
                f"Workspace  {self.workspace}\nModel      {self.model}",
                id="loading-meta",
                markup=False,
            )
            yield Static(f"Session   {self.request_label}", id="loading-session")
            yield LoadingIndicator(id="loading-indicator")
            yield Static(self.progress, id="loading-status")

    def set_progress(self, value: str) -> None:
        self.progress = value
        if self.is_mounted:
            self.query_one("#loading-status", Static).update(value)


class MainScreen(Screen[None]):
    def __init__(
        self,
        *,
        workspace: str,
        model: str,
        history: tuple[tuple[str, str], ...] = (),
        context_status: ContextStatus | None = None,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.model = model
        self.history = history
        self.context_status = context_status

    def compose(self) -> ComposeResult:
        yield HeaderBar(self.workspace, self.model, id="header")
        yield ConversationView(id="conversation")
        yield StatusBar("Runtime Ready", id="status")
        yield CommandPicker(id="command-picker")
        yield PromptTextArea(
            placeholder="Ask MyCode...  (Ctrl+J / Ctrl+Enter to send, / for commands)",
            id="prompt",
        )

    def on_mount(self) -> None:
        self.app._activate_main_screen(self)
        self.query_one(PromptTextArea).focus()

    def on_prompt_text_area_submitted(
        self, event: PromptTextArea.Submitted
    ) -> None:
        """Route a submitted prompt to the existing Agent/command entry point.

        The composer clears itself before posting, and the submitted text is
        parsed again by ``parse_slash_command()`` inside the app, so the picker
        never becomes a second command-validation path.
        """
        if event.text.strip():
            self.app.submit_user_message(event.text)

    def on_prompt_text_area_picker_changed(
        self, event: PromptTextArea.PickerChanged
    ) -> None:
        self.query_one(CommandPicker).render_state(event.state)

    def _on_paste(self, event: Paste) -> None:
        """Route a paste that reached the screen back into the prompt composer.

        Textual forwards ``Paste`` to the focused widget and falls back to the
        current screen when nothing is focused. Windows Terminal blurs the app
        while its large-paste confirmation dialog is up, so a large paste is
        delivered during that window and would otherwise land here and be
        dropped. The composer stops the events it handles itself, so this only
        sees pastes that would otherwise be lost.
        """
        prompt = self.query_one(PromptTextArea)
        if prompt.disabled:
            return
        event.stop()
        prompt.post_message(Paste(event.text))


class ModelSettingsScreen(ModalScreen[dict | None]):
    """Secrets enter only password controls and never the conversation view."""
    BINDINGS = [("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    ModelSettingsScreen { align: center middle; }
    ModelSettingsScreen > Vertical { width: 78; height: auto; max-height: 95%; overflow-y: auto; background: $surface; padding: 1 2; border: round $accent; }
    ModelSettingsScreen Input, ModelSettingsScreen Select { margin-bottom: 1; }
    """

    def __init__(self, *, connection=False, selection=None, windows=None):
        super().__init__()
        self.connection = connection
        self._initial = {}
        if selection is not None:
            names = {"window": "context_window_tokens", "reserve": "reserved_output_tokens",
                     "margin": "context_safety_margin_tokens", "output": "max_output_tokens"}
            for role in ("main", "subagent", "compact"):
                selected = getattr(selection, role)
                values = {"role": role, "model": selected.model or "follow",
                          "connection": selected.connection_ref or "follow",
                          "thinking": selected.thinking or "inherit",
                          "effort": selected.effort or ("auto" if role == "main" else "inherit")}
                for key, name in names.items():
                    value = getattr(selected, name)
                    if key == "window" and windows is not None:
                        value = windows[role].requested
                    values[key] = str(value) if value is not None else (
                        "auto" if key == "output" and role == "main" else "inherit")
                self._initial[role] = values
        self._drafts = {role: dict(values) for role, values in self._initial.items()}
        self._selected_role = "main"

    def compose(self):
        with Vertical():
            yield Static("API connection (blank fields keep existing values)" if self.connection else "Edit the selected role; existing values and per-role drafts are preserved")
            if self.connection:
                yield Input(placeholder="Base URL", id="setting-base_url")
                yield Input(placeholder="API Key (hidden)", password=True, id="setting-api_key")
                yield Input(placeholder="Headers JSON (hidden; {} clears)", password=True, id="setting-headers")
            else:
                yield Select([(role, role) for role in ("main", "subagent", "compact")], value="main", allow_blank=False, id="setting-role")
                yield Input(placeholder="Model ID, or follow for SubAgent/Compact", id="setting-model")
                yield Input(placeholder="Connection ID (blank retains; follow inherits Main; env uses .env)", id="setting-connection")
                yield Select([("Follow Main/default", "inherit"), ("Auto", "auto"), ("On", "on"), ("Off", "off")], value="auto", allow_blank=False, id="setting-thinking")
                for key, label in (("effort", "Effort: low / high / max (blank = API default)"),
                                   ("window", "Context window"), ("reserve", "Output reserve"),
                                   ("margin", "Safety margin"), ("output", "Request output limit")):
                    yield Input(placeholder=label, id=f"setting-{key}")
            yield Button("Save" if self.connection else "Save selected role", id="settings-save", variant="primary")
            yield Button("Cancel", id="settings-cancel")

    def on_button_pressed(self, event):
        if event.button.id == "settings-cancel":
            self.action_cancel()
            return
        if event.button.id == "settings-save":
            values = self._values()
            original = self._initial.get(self._selected_role)
            if not self.connection and original is not None:
                if values == original:
                    self.dismiss(None)
                    return
                values = {key: value for key, value in values.items()
                          if key in {"role", "model", "thinking", "effort", "output"} or value != original.get(key)}
            self.dismiss(values)

    def _values(self):
        values = {widget.id.removeprefix("setting-"): widget.value for widget in self.query(Input)}
        for widget in self.query(Select):
            values[widget.id.removeprefix("setting-")] = str(widget.value)
        return values

    def on_mount(self):
        if not self.connection:
            self._load_role("main")

    def _load_role(self, role):
        values = self._drafts.get(role, {"model": "", "connection": "",
            "thinking": "off" if role == "compact" else "auto"})
        for widget in self.query(Input):
            widget.value = values.get(widget.id.removeprefix("setting-"), "")
        self.query_one("#setting-thinking", Select).value = values["thinking"]

    def on_select_changed(self, event):
        if event.select.id == "setting-role":
            role = str(event.value)
            if role == self._selected_role:
                return
            try:
                self._drafts[self._selected_role] = self._values()
                self._selected_role = role
                self._load_role(role)
            except NoMatches:
                pass

    def action_cancel(self):
        self.dismiss(None)


class MCPTrustScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "reject", "Reject")]

    def __init__(self, request: MCPTrustRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        lines = [
            "Project MCP trust request",
            "Review the safe configuration summary before enabling MCP.",
            "",
        ]
        for server in self.request.servers:
            lines.extend((f"server: {server.alias}", f"transport: {server.transport}"))
            if server.transport == "stdio":
                lines.append(f"command: {server.command!r}")
                lines.append(f"args: {list(server.args)!r}")
                if server.env_keys:
                    lines.append(f"env keys: {list(server.env_keys)!r}")
            else:
                lines.append(f"url template: {server.url_template!r}")
                if server.destination is not None:
                    lines.append(f"destination: {server.destination!r}")
                if server.header_keys:
                    lines.append(f"header keys: {list(server.header_keys)!r}")
            lines.append("")

        with Vertical(id="mcp-trust-dialog"):
            yield Static("\n".join(lines), id="mcp-trust-details")
            with Vertical(id="mcp-trust-actions"):
                yield Button("Approve", id="mcp-trust-approve", variant="success")
                yield Button("Reject", id="mcp-trust-reject", variant="error")

    def on_mount(self) -> None:
        self.query_one("#mcp-trust-approve", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-trust-approve":
            self.dismiss(True)
        elif event.button.id == "mcp-trust-reject":
            self.dismiss(False)

    def action_reject(self) -> None:
        self.dismiss(False)


_PERMISSION_METADATA_KEYS = (
    "resolved_path",
    "workspace_root",
    "path_scope",
    "pattern_scope",
    "command_display",
    "resolved_cwd",
    "cwd_scope",
    "command_risk_category",
    "command_risk",
    "command_risk_reason",
    "memory_scope",
    "memory_kind",
    "memory_key",
    "memory_path",
    "skill_name",
    "skill_source",
    "script",
)


def _permission_details(request: ConfirmationRequest) -> str:
    permission_request = request.permission_request
    lines = [
        f"tool: {permission_request.tool_name}",
        f"capability: {permission_request.capability}",
        f"action: {permission_request.action}",
    ]
    if permission_request.target is not None:
        lines.append(f"target: {permission_request.target}")
    lines.append(f"reason: {request.permission_decision.reason}")
    if request.prompt:
        lines.append(f"prompt: {request.prompt}")

    metadata = {**request.permission_decision.metadata, **request.metadata}
    for key in _PERMISSION_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        rendered = str(value)
        if len(rendered) > 240:
            rendered = rendered[:237] + "..."
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


class PermissionScreen(ModalScreen[ConfirmationResult]):
    BINDINGS = [("escape", "deny", "Deny")]

    def __init__(self, request: ConfirmationRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Static(
                "Permission required\n\n" + _permission_details(self.request),
                id="permission-details",
                markup=False,
            )
            with Vertical(id="permission-actions"):
                yield Button("Allow once", id="permission-allow-once", variant="success")
                yield Button("Allow task", id="permission-allow-task", variant="success")
                yield Button("Allow session", id="permission-allow-session", variant="success")
                yield Button("Deny", id="permission-deny", variant="error")

    def on_mount(self) -> None:
        self.query_one("#permission-allow-once", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        scopes = {
            "permission-allow-once": "once",
            "permission-allow-task": "task",
            "permission-allow-session": "session",
        }
        if event.button.id in scopes:
            self.dismiss(ConfirmationResult.approved(scope=scopes[event.button.id]))
        elif event.button.id == "permission-deny":
            self.dismiss(
                ConfirmationResult.rejected(message="Permission denied by user.")
            )

    def action_deny(self) -> None:
        self.dismiss(
            ConfirmationResult.rejected(message="Permission denied by user.")
        )


__all__ = [
    "LoadingScreen",
    "MCPTrustScreen",
    "MainScreen",
    "PermissionScreen",
    "WelcomeScreen",
]
