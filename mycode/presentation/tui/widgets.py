"""Reusable widgets for the MyCode Textual presentation."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static


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

        Lightweight Coding Agent"""


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


__all__ = ["ConversationView", "HeaderBar", "SPLASH_LOGO", "StatusBar"]
