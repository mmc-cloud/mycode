"""Reusable widgets for the MyCode Textual presentation."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.containers import VerticalScroll
from textual.events import Key, Paste
from textual.message import Message
from textual.widgets import Static, TextArea

from mycode.presentation.commands import COMMAND_SPECS, CommandSpec


SPLASH_LOGO = r"""        ╭──────────╮
        │    >_    │
        │  MYCODE  │
        ╰──────────╯

 __  __       ____          _
|  \/  |_   _/ ___|___   __| | ___
| |\/| | | | | |   / _ \ / _` |/ _ \
| |  | | |_| | |__| (_) | (_| |  __/
|_|  |_|\__, |\____\___/ \__,_|\___|
        |___/

        Extensible Terminal Coding Agent"""


@dataclass
class _ConversationEntry:
    kind: str
    content: str
    name: str = ""
    arguments: str = ""
    completed: bool = False
    ok: bool = True
    summary: str = ""


class _ConversationEntryWidget(Static):
    """Render one conversation entry that can be updated in place."""

    def __init__(self, entry: _ConversationEntry) -> None:
        self.entry = entry
        super().__init__(self._render_entry(entry), markup=False)

    def update_entry(self, entry: _ConversationEntry) -> None:
        self.entry = entry
        self.update(self._render_entry(entry))

    @staticmethod
    def _render_entry(entry: _ConversationEntry) -> Text:
        if entry.kind == "user":
            return Text.assemble(("You", "bold cyan"), "\n", entry.content)
        if entry.kind == "assistant":
            return Text.assemble(("MyCode", "bold green"), "\n", entry.content)
        if entry.kind == "tool":
            if not entry.completed:
                line = f"› {entry.name}"
                if entry.arguments:
                    line += f" {entry.arguments}"
                return Text(line, style="yellow")
            marker = "✓" if entry.ok else "✗"
            line = f"{marker} {entry.name}"
            if not entry.ok and entry.summary:
                line += f" {entry.summary}"
            return Text(line, style="green" if entry.ok else "red")

        style = {
            "warning": "yellow",
            "error": "red",
            "info": "dim",
        }.get(entry.kind, "dim")
        return Text(entry.content, style=style)


class ConversationView(VerticalScroll):
    """Incremental conversation projection with a model-backed transcript."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._entries: list[_ConversationEntry] = []
        self._entry_widgets: dict[int, _ConversationEntryWidget] = {}

    def add_user_message(self, content: str) -> None:
        self._entries.append(_ConversationEntry("user", content))
        self._mount_entry(len(self._entries) - 1)

    def add_assistant_message(self, content: str) -> None:
        self._entries.append(_ConversationEntry("assistant", content))
        self._mount_entry(len(self._entries) - 1)

    def add_history_message(self, role: str, content: str) -> None:
        if role == "user":
            self.add_user_message(content)
        elif role == "assistant":
            self.add_assistant_message(content)

    def append_assistant_delta(self, content: str) -> None:
        if self._entries and self._entries[-1].kind == "assistant":
            entry_index = len(self._entries) - 1
            self._entries[entry_index].content += content
            self._refresh_entry(entry_index)
        else:
            self.add_assistant_message(content)

    def add_notice(self, content: str, *, level: str = "info") -> None:
        self._entries.append(_ConversationEntry(level, content))
        self._mount_entry(len(self._entries) - 1)

    def add_tool_activity(self, name: str, arguments: str) -> int:
        token = len(self._entries)
        self._entries.append(
            _ConversationEntry("tool", "", name=name, arguments=arguments)
        )
        self._mount_entry(token)
        return token

    def complete_tool_activity(
        self,
        token: int,
        name: str,
        *,
        ok: bool,
        summary: str,
    ) -> None:
        if token >= len(self._entries) or self._entries[token].kind != "tool":
            self.add_notice(
                f"⚠ tool result without pending call: {name}",
                level="warning",
            )
            return
        entry = self._entries[token]
        entry.completed = True
        entry.ok = ok
        entry.summary = summary
        self._refresh_entry(token)

    @property
    def transcript_text(self) -> str:
        lines: list[str] = []
        for entry in self._entries:
            if entry.kind in {"user", "assistant", "info", "warning", "error"}:
                lines.append(entry.content)
            elif entry.kind == "tool":
                if entry.completed:
                    marker = "✓" if entry.ok else "✗"
                    line = f"{marker} {entry.name}"
                    if not entry.ok and entry.summary:
                        line += f" {entry.summary}"
                    lines.append(line)
                else:
                    line = f"› {entry.name}"
                    if entry.arguments:
                        line += f" {entry.arguments}"
                    lines.append(line)
        return "\n".join(lines)

    def _mount_entry(self, index: int) -> None:
        widget = _ConversationEntryWidget(self._entries[index])
        self._entry_widgets[index] = widget
        self.mount(widget)
        self.call_after_refresh(self._scroll_to_end)

    def _refresh_entry(self, index: int) -> None:
        widget = self._entry_widgets.get(index)
        if widget is None:
            self._mount_entry(index)
            return
        widget.update_entry(self._entries[index])
        self.call_after_refresh(self._scroll_to_end)

    def _scroll_to_end(self) -> None:
        if self.is_mounted:
            self.scroll_end(animate=False, immediate=True, force=True)


class HeaderBar(Static):
    def __init__(
        self,
        workspace: str = "—",
        model: str = "—",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._workspace = workspace or "—"
        self._model = model or "—"
        self._session = "—"

    def on_mount(self) -> None:
        self._refresh()

    def set_session(self, value: str) -> None:
        self._session = value or "—"
        self._refresh()

    def _refresh(self) -> None:
        self.update(
            Text(
                f">_ MyCode   {self._workspace}   {self._model}   "
                f"session: {self._session}",
                style="bold",
            )
        )


class StatusBar(Static):
    def __init__(self, value: str = "UI Ready", *args, **kwargs) -> None:
        super().__init__(value, *args, **kwargs)

    def set_status(self, value: str) -> None:
        self.update(Text(value))


@dataclass(frozen=True)
class CommandPickerState:
    """The matches and highlight the slash-command picker should display."""

    query: str
    specs: tuple[CommandSpec, ...]
    highlighted: int = 0


def filter_command_specs(query: str) -> tuple[CommandSpec, ...]:
    """Match registered commands by name or alias prefix.

    The picker reads the shared ``COMMAND_SPECS`` registry instead of keeping a
    second command list, so the picker and ``parse_slash_command()`` can never
    disagree about which commands exist. An empty query lists every command.
    """

    normalized = query.strip().lower()
    if not normalized:
        return COMMAND_SPECS
    return tuple(
        spec
        for spec in COMMAND_SPECS
        if any(
            candidate.startswith(normalized)
            for candidate in (spec.name, *spec.aliases)
        )
    )


class CommandPicker(Static):
    """Read-only dropdown that renders the composer's current matches.

    ``PromptTextArea`` owns the query and the highlighted index; this widget
    only draws that state, so keyboard focus never leaves the composer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("", *args, markup=False, **kwargs)
        self.state: CommandPickerState | None = None
        self.display = False

    @property
    def is_open(self) -> bool:
        """Whether the picker is currently shown.

        Named ``is_open`` rather than ``visible`` because Textual already
        defines ``Widget.visible`` for the ``visibility`` style, which is a
        different thing from this widget's ``display`` toggle.
        """
        return bool(self.display)

    def render_state(self, state: CommandPickerState | None) -> None:
        """Show ``state``, or hide the picker when it is ``None``."""
        self.state = state
        if state is None:
            self.update("")
            self.display = False
            return
        self.update(self._render_options(state))
        self.display = True

    @staticmethod
    def _render_options(state: CommandPickerState) -> Text:
        rendered = Text()
        for index, spec in enumerate(state.specs):
            if index:
                rendered.append("\n")
            style = "reverse" if index == state.highlighted else "dim"
            rendered.append(f"{spec.usage}  {spec.description}", style=style)
        return rendered


class PromptTextArea(TextArea):
    """Multi-line prompt composer with an inline slash-command picker.

    Key contract:

    - ``Enter`` inserts a newline, so long prompts keep their line breaks.
    - ``Ctrl+Enter`` submits the composed text and clears the composer.
    - ``Ctrl+J`` submits as well. On Windows consoles Textual's driver reads
      only the key's character, so plain ``Enter`` arrives as CR while both
      ``Ctrl+Enter`` and ``Ctrl+J`` arrive as LF; terminals that report the
      kitty keyboard protocol send ``Ctrl+Enter`` separately. Accepting both
      names covers those terminals without assuming any of them is universal.
    - While the picker is open, ``Up``/``Down`` move the highlight, ``Enter``
      accepts the highlighted command and ``Esc`` closes the picker.

    Choosing a command from the picker has two outcomes, decided by
    ``CommandSpec.requires_arguments``: a command that needs no argument runs
    immediately, and a command that needs one is only written as ``/name ``
    so the user can type the argument. Either way the text is submitted through
    the normal path, so ``parse_slash_command()`` stays the only validator.

    A paste arrives as a single ``Paste`` event, so a pasted block is inserted
    atomically and never runs the normal key contract. A paste also never opens
    the picker on its own: ``/xxx`` inside pasted text stays ordinary prompt
    text.
    """

    SUBMIT_KEYS = frozenset({"ctrl+enter", "ctrl+j"})
    """Keys that submit. ``ctrl+j`` is the LF byte. Textual's Windows console
    driver only reads the key's character, so on those consoles plain ``Enter``
    (CR) and ``Ctrl+Enter`` (LF) do arrive differently, but ``Ctrl+Enter`` is
    not distinguishable from ``Ctrl+J``. Terminals that report the kitty
    keyboard protocol give ``Ctrl+Enter`` its own name."""

    class Submitted(Message):
        """Posted after the composer submits and clears its text."""

        def __init__(self, prompt: "PromptTextArea", text: str) -> None:
            self.prompt = prompt
            self.text = text
            super().__init__()

    class PickerChanged(Message):
        """Posted when the picker should render new matches, or close."""

        def __init__(self, state: CommandPickerState | None) -> None:
            self.state = state
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._picker_state: CommandPickerState | None = None
        self._pasted_edits = 0

    @property
    def picker_state(self) -> CommandPickerState | None:
        """The picker state, or ``None`` while the picker is closed."""
        return self._picker_state

    async def _on_key(self, event: Key) -> None:
        if self._picker_state is not None and self._consume_picker_key(event):
            return
        if event.key in self.SUBMIT_KEYS:
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        # Every other key is left to Textual's own dispatch, which keeps walking
        # the MRO and runs ``TextArea._on_key`` for text insertion. Calling the
        # base handler here as well does not duplicate that insertion today, but
        # only because ``TextArea._on_key`` calls ``prevent_default()`` for the
        # keys it handles; depending on a base class mutating the event to avoid
        # double handling is what this leaves out.

    async def _on_paste(self, event: Paste) -> None:
        """Insert a whole paste as one edit, without opening the picker.

        Textual also calls base-class handlers unless the default action is
        prevented, and the base handler would insert the same text twice.
        """
        event.stop()
        event.prevent_default()
        before = self.text
        await super()._on_paste(event)
        if self.text != before:
            self._pasted_edits += 1

    def _on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._pasted_edits:
            self._pasted_edits -= 1
            self._refresh_picker(allow_open=False)
            return
        self._refresh_picker(allow_open=True)

    def action_submit(self) -> None:
        """Submit the composed text and clear the composer."""
        self._submit(self.text)

    def _submit(self, content: str) -> None:
        """Post ``content`` for submission and reset the composer state."""
        if self.disabled:
            return
        self._set_picker_state(None)
        self.clear()
        self.post_message(self.Submitted(self, content))

    def _consume_picker_key(self, event: Key) -> bool:
        """Handle one picker key, returning whether the picker consumed it."""
        if event.key == "up":
            self._move_picker(-1)
        elif event.key == "down":
            self._move_picker(1)
        elif event.key == "enter":
            self._accept_highlighted_command()
        elif event.key == "escape":
            self._set_picker_state(None)
        else:
            return False
        event.stop()
        event.prevent_default()
        return True

    def _move_picker(self, delta: int) -> None:
        state = self._picker_state
        if state is None:
            return
        highlighted = min(max(state.highlighted + delta, 0), len(state.specs) - 1)
        if highlighted != state.highlighted:
            self._set_picker_state(
                CommandPickerState(state.query, state.specs, highlighted)
            )

    def _accept_highlighted_command(self) -> None:
        """Run the highlighted command, or fill it in when it needs arguments."""
        state = self._picker_state
        if state is None or not state.specs:
            return
        spec = state.specs[state.highlighted]
        if not spec.requires_arguments:
            self._submit(f"/{spec.name}")
            return
        self.replace(f"/{spec.name} ", (0, 0), self.document.end)
        self.move_cursor(self.document.end)
        self._set_picker_state(None)

    def _command_query(self) -> str | None:
        """Return the pending command query, or ``None`` for normal text.

        Only a single unfinished ``/token`` line can be a command query.
        Multi-line text, arguments and ordinary prompts never qualify, which is
        what keeps a pasted ``/xxx`` block out of the picker.
        """
        text = self.text
        if not text.startswith("/"):
            return None
        token = text[1:]
        if any(character.isspace() for character in token):
            return None
        return token

    def _refresh_picker(self, *, allow_open: bool) -> None:
        query = self._command_query()
        if query is None:
            self._set_picker_state(None)
            return
        matches = filter_command_specs(query)
        if not matches:
            self._set_picker_state(None)
            return
        if not allow_open and self._picker_state is None:
            return
        highlighted = 0
        if self._picker_state is not None and self._picker_state.query == query:
            highlighted = min(self._picker_state.highlighted, len(matches) - 1)
        self._set_picker_state(
            CommandPickerState(query=query, specs=matches, highlighted=highlighted)
        )

    def _set_picker_state(self, state: CommandPickerState | None) -> None:
        if state == self._picker_state:
            return
        self._picker_state = state
        self.post_message(self.PickerChanged(state))


__all__ = [
    "CommandPicker",
    "CommandPickerState",
    "ConversationView",
    "HeaderBar",
    "PromptTextArea",
    "SPLASH_LOGO",
    "StatusBar",
    "filter_command_specs",
]
