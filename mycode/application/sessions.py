from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from mycode.context.compact import CompactState
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.persistence.session_store import (
    DEFAULT_SESSION_TITLE,
    SessionNotFoundError,
    SessionRecord,
    SessionStore,
    WritableSession,
)
from mycode.project import ProjectIdentity


SessionStartMode = Literal["new", "continue", "resume"]
DEFAULT_SESSION_LIST_LIMIT = 10
AUTO_SESSION_TITLE_CHARS = 80


@dataclass(frozen=True)
class SessionStartRequest:
    mode: SessionStartMode
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"new", "continue", "resume"}:
            raise ValueError(f"Unsupported session start mode: {self.mode}")
        if self.mode == "resume":
            if self.session_id is None or self.session_id.strip() == "":
                raise ValueError("resume mode requires session_id.")
            return
        if self.session_id is not None:
            raise ValueError(f"{self.mode} mode must not include session_id.")


@dataclass
class ActiveProjectSession:
    store: SessionStore
    project: ProjectIdentity
    writer: WritableSession
    _ownership: AbstractContextManager[WritableSession] = field(repr=False)
    _created: bool = field(default=False, repr=False)
    _finished: bool = field(default=False, init=False)
    _compact_state_recovered: bool = field(default=False, init=False)

    @property
    def record(self) -> SessionRecord:
        return replace(
            self.writer.record,
            status=self.writer.record.status if self._finished else "active",
        )

    @property
    def created(self) -> bool:
        return self._created

    def load_history(self) -> Conversation:
        return self.writer.load_history()

    def load_compact_state(self) -> CompactState:
        loaded = self.writer.load_or_reset_compact_state()
        self._compact_state_recovered = loaded.recovered_invalid_state
        return loaded.state

    @property
    def compact_state_recovered(self) -> bool:
        return self._compact_state_recovered

    @property
    def artifact_directory(self) -> Path:
        return self.writer.layout.artifacts_directory

    def persist_message(self, message: Message) -> None:
        self.writer.append_messages([message])
        if message.role == "user" and self.record.title == DEFAULT_SESSION_TITLE:
            title = _session_title_from_message(message.content)
            if title is not None:
                self.writer.rename(title)

    def persist_compact_state(self, state: CompactState) -> None:
        self.writer.save_compact_state(state)

    def close(self) -> None:
        self._finish("closed")

    def interrupt(self) -> None:
        self._finish("interrupted")

    def _finish(self, status: Literal["closed", "interrupted"]) -> None:
        if self._finished:
            return
        try:
            self.writer.finish(status)
        finally:
            self._finished = True
            self._ownership.__exit__(None, None, None)


def list_project_sessions(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    limit: int = DEFAULT_SESSION_LIST_LIMIT,
) -> list[SessionRecord]:
    return store.list_sessions(project, limit=limit)


def delete_project_session(
    store: SessionStore,
    project: ProjectIdentity,
    session_id: str,
) -> bool:
    return store.delete_session(project, session_id)


def start_project_session(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    request: SessionStartRequest,
) -> ActiveProjectSession:
    if request.mode == "new":
        return _open_active(store, project, created=True)

    if request.mode == "continue":
        sessions = list_project_sessions(store, project, limit=1)
        if not sessions:
            return _open_active(store, project, created=True)
        return _open_active(store, project, record=sessions[0])

    identifier = request.session_id or ""
    session = store.get_session(project, identifier)
    if session is None:
        raise SessionNotFoundError(
            f"Session not found in current project: {identifier}"
        )
    return _open_active(store, project, record=session)


def _open_active(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    record: SessionRecord | None = None,
    created: bool = False,
) -> ActiveProjectSession:
    ownership = store.open_session(
        project,
        None if record is None else record.id,
        create=record is None,
    )
    writer = ownership.__enter__()
    active = ActiveProjectSession(
        store,
        project,
        writer,
        ownership,
        _created=created,
    )
    try:
        writer.finish("interrupted")
        return active
    except BaseException:
        active.interrupt()
        raise


def _session_title_from_message(content: str) -> str | None:
    normalized = " ".join(content.split())
    if normalized == "":
        return None
    if len(normalized) <= AUTO_SESSION_TITLE_CHARS:
        return normalized
    return normalized[: AUTO_SESSION_TITLE_CHARS - 3].rstrip() + "..."
