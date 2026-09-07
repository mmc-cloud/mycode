"""Project-scoped persistence; writable handles own a lifecycle OS lock."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mycode.agent import AgentToolCall
from mycode.context_compact import CompactState, DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES
from mycode.conversation import Conversation
from mycode.filesystem import (
    FilesystemStorageError, JsonSnapshotError, append_jsonl_record, prepare_jsonl_for_append,
    read_json_snapshot, read_jsonl_records, write_json_snapshot,
)
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.project_storage import ProjectStorage, ProjectStorageError, validate_storage_component
from mycode.session_lock import (
    SessionLifecycleLock, SessionLockError, SessionLockTimeoutError,
)

DEFAULT_SESSION_TITLE = "New session"
MAX_SESSION_TITLE_CHARS = 200
MAX_COMPACT_STATE_JSON_CHARS = 20000


class SessionStoreError(RuntimeError):
    pass


class SessionNotFoundError(SessionStoreError):
    pass


class SessionInUseError(SessionStoreError):
    pass


class SessionDataError(SessionStoreError):
    pass


class CompactStateDataError(SessionDataError):
    """Only invalid Compact snapshot data; never session or boundary failures."""


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    session_id: str
    title: str = Field(min_length=1, max_length=MAX_SESSION_TITLE_CHARS)
    created_at: datetime
    updated_at: datetime
    last_terminal_state: Literal["closed", "interrupted"] = "interrupted"


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


class _TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[_ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    reasoning_state: Literal["absent", "present_empty", "present_nonempty"] = "absent"

    @model_validator(mode="after")
    def validate_message(self) -> "_TranscriptRecord":
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages can contain tool calls.")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("Only tool results require tool_call_id.")
        if self.reasoning_state != "absent" and (
            self.role != "assistant" or not self.tool_calls
        ):
            raise ValueError("Reasoning requires an assistant tool call.")
        if self.reasoning_state == "present_nonempty":
            if not self.reasoning_content:
                raise ValueError("Nonempty reasoning required.")
        elif self.reasoning_content is not None:
            raise ValueError("Inconsistent reasoning content.")
        return self

    def message(self) -> Message:
        return Message(
            role=self.role, content=self.content,
            tool_calls=tuple(AgentToolCall(
                id=c.id, name=c.name, arguments=c.arguments,
            ) for c in self.tool_calls),
            tool_call_id=self.tool_call_id,
            reasoning_content=self.reasoning_content,
            reasoning_state=self.reasoning_state,
        )


@dataclass(frozen=True)
class SessionRecord:
    id: str
    project_key: str
    workspace_root: Path
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CompactStateLoadResult:
    state: CompactState
    recovered_invalid_state: bool = False


@contextmanager
def _storage_errors() -> Iterator[None]:
    try:
        yield
    except SessionLockTimeoutError as error:
        raise SessionInUseError("Session is in use by another owner.") from error
    except (FilesystemStorageError, ProjectStorageError, ValidationError) as error:
        raise SessionDataError("Invalid session filesystem data.") from error
    except (OSError, SessionLockError) as error:
        raise SessionStoreError("Session filesystem operation failed.") from error


class SessionStore:
    def __init__(
        self, projects_root: str | Path | None = None, *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.projects_root = projects_root
        self._now = now or (lambda: datetime.now(timezone.utc))

    def project_storage(self, project: ProjectIdentity) -> ProjectStorage:
        with _storage_errors():
            return ProjectStorage.open(project, projects_root=self.projects_root)

    def _metadata(self, storage: ProjectStorage, session_id: str) -> _Metadata:
        layout = storage.session(session_id)
        if not layout.root.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")
        # lstat distinguishes an absent commit marker from a dangling link.
        try:
            layout.meta_path.lstat()
        except FileNotFoundError:
            raise SessionNotFoundError(f"Session not committed: {session_id}") from None
        data = read_json_snapshot(storage.project_directory, layout.meta_path)
        meta = _Metadata.model_validate_json(json.dumps(data))
        if meta.session_id != session_id:
            raise SessionDataError("Session metadata identity mismatch.")
        if any(t.tzinfo is None for t in (meta.created_at, meta.updated_at)):
            raise SessionDataError("Session timestamps require timezone.")
        return meta

    def _record(self, project: ProjectIdentity, meta: _Metadata) -> SessionRecord:
        return SessionRecord(
            id=meta.session_id, project_key=project.key,
            workspace_root=project.workspace_root, title=meta.title,
            status=meta.last_terminal_state,
            created_at=meta.created_at, updated_at=meta.updated_at,
        )

    def get_session(self, project: ProjectIdentity, session_id: str) -> SessionRecord | None:
        with _storage_errors():
            storage = self.project_storage(project)
            try:
                meta = self._metadata(storage, session_id)
            except SessionNotFoundError:
                return None
            record = self._record(project, meta)
            try:
                with SessionLifecycleLock(
                    storage.session(session_id).lock_path, timeout_seconds=0,
                ).acquire():
                    pass
            except SessionLockTimeoutError:
                record = replace(record, status="active")
            return record

    def list_sessions(self, project: ProjectIdentity, *, limit: int = 10) -> list[SessionRecord]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        with _storage_errors():
            storage = self.project_storage(project)
            records = []
            for path in storage.sessions_directory.iterdir():
                try:
                    validate_storage_component(path.name, field_name="session_id")
                except ProjectStorageError:
                    continue
                if not path.is_dir() and not path.is_symlink():
                    continue
                record = self.get_session(project, path.name)
                if record is not None:
                    records.append(record)
            return sorted(records, key=lambda r: (r.updated_at, r.id), reverse=True)[:limit]

    @contextmanager
    def open_session(
        self, project: ProjectIdentity, session_id: str | None = None, *,
        create: bool = False, title: str = DEFAULT_SESSION_TITLE,
    ) -> Iterator["WritableSession"]:
        if session_id is None and not create:
            raise SessionNotFoundError("Session id is required for resume.")
        identifier = str(uuid4()) if session_id is None else session_id
        with ExitStack() as ownership:
            with _storage_errors():
                storage = self.project_storage(project)
                layout = storage.session(identifier)
                ownership.enter_context(SessionLifecycleLock(
                    layout.lock_path, timeout_seconds=0,
                ).acquire())
                if create:
                    if layout.root.exists():
                        try:
                            self._metadata(storage, identifier)
                        except SessionNotFoundError:
                            _validate_deletion_tree(layout.root)
                            shutil.rmtree(layout.root)
                        else:
                            raise SessionDataError("Session already exists.")
                    meta = _Metadata(
                        session_id=identifier, title=_title(title),
                        created_at=self._now(), updated_at=self._now(),
                    )
                    try:
                        layout = storage.session(identifier, create=True)
                        layout.transcript_path.touch(exist_ok=False)
                        # Publish last: all prior data is an invisible orphan until this succeeds.
                        write_json_snapshot(
                            storage.project_directory, layout.meta_path, meta.model_dump(mode="json"),
                        )
                    except BaseException:
                        try:
                            if not layout.meta_path.exists() and layout.root.exists():
                                _validate_deletion_tree(layout.root)
                                shutil.rmtree(layout.root)
                        except (OSError, SessionDataError):
                            pass  # An uncommitted orphan remains invisible and can be retried.
                        raise
                else:
                    meta = self._metadata(storage, identifier)
                records = prepare_jsonl_for_append(
                    storage.project_directory, layout.transcript_path,
                )
                messages = _messages(records)
                writer = WritableSession(self, storage, meta, messages)
                del records, messages
            try:
                yield writer
            finally:
                with writer._mutex:
                    writer._closed = True

    def create_session(
        self, project: ProjectIdentity, *, session_id: str | None = None,
        title: str = DEFAULT_SESSION_TITLE,
    ) -> SessionRecord:
        with self.open_session(project, session_id, create=True, title=title) as session:
            return session.record

    def load_conversation(self, project: ProjectIdentity, session_id: str) -> Conversation:
        with _storage_errors():
            storage = self.project_storage(project)
            self._metadata(storage, session_id)
            records = read_jsonl_records(
                storage.project_directory, storage.session(session_id).transcript_path,
                trailing_record="ignore",
            )
            return Conversation.from_messages(_messages(records))

    def append_message(self, project: ProjectIdentity, session_id: str, message: Message) -> None:
        self.append_messages(project, session_id, [message])

    def append_messages(
        self, project: ProjectIdentity, session_id: str, messages: Iterable[Message],
    ) -> None:
        with self.open_session(project, session_id) as session:
            session.append_messages(messages)

    def rename_session(self, project: ProjectIdentity, session_id: str, title: str) -> SessionRecord:
        with self.open_session(project, session_id) as session:
            return session.rename(title)

    def load_compact_state(self, project: ProjectIdentity, session_id: str) -> CompactState:
        with _storage_errors():
            storage = self.project_storage(project)
            self._metadata(storage, session_id)
            message_count = len(self.load_conversation(project, session_id).get_messages())
            return _load_compact_snapshot(storage, session_id, message_count)

    def save_compact_state(self, project: ProjectIdentity, session_id: str, state: CompactState) -> None:
        with self.open_session(project, session_id) as session:
            session.save_compact_state(state)

    def delete_session(self, project: ProjectIdentity, session_id: str) -> bool:
        with _storage_errors():
            storage = self.project_storage(project)
            layout = storage.session(session_id)
            with SessionLifecycleLock(layout.lock_path, timeout_seconds=0).acquire():
                if not layout.root.exists():
                    return False
                _validate_deletion_tree(layout.root)
                # Revoke logical existence before any partial physical cleanup.
                layout.meta_path.unlink(missing_ok=True)
                shutil.rmtree(layout.root)
                return True


class WritableSession:
    """Scoped write capability; all methods reject use after the context exits."""

    def __init__(
        self, store: SessionStore, storage: ProjectStorage,
        metadata: _Metadata, messages: list[Message],
    ) -> None:
        self.store = store
        self.storage = storage
        self.layout = storage.session(metadata.session_id)
        self._meta = metadata
        self._messages = list(messages)
        self._closed = False
        self._mutex = RLock()

    @property
    def record(self) -> SessionRecord:
        return self.store._record(self.storage.identity, self._meta)

    def _check_open(self) -> None:
        if self._closed:
            raise SessionStoreError("Session writer is closed.")
        self.storage.session(self.record.id)

    @property
    def _message_count(self) -> int:
        return len(self._messages)

    def load_history(self) -> Conversation:
        with self._mutex, _storage_errors():
            self._check_open()
            return Conversation.from_messages(self._messages)

    def _save_metadata(self) -> None:
        self._meta.updated_at = self.store._now()
        write_json_snapshot(
            self.storage.project_directory, self.layout.meta_path, self._meta.model_dump(mode="json"),
        )

    def append_messages(self, messages: Iterable[Message]) -> None:
        with self._mutex, _storage_errors():
            self._check_open()
            prepared = [
                _transcript(message, self._message_count + i + 1)
                for i, message in enumerate(messages)
            ]
            for record in prepared:
                message = _TranscriptRecord.model_validate(record).message()
                append_jsonl_record(self.storage.project_directory, self.layout.transcript_path, record)
                self._messages.append(message)
            if prepared:
                self._save_metadata()

    def rename(self, title: str) -> SessionRecord:
        with self._mutex, _storage_errors():
            self._check_open()
            self._meta.title = _title(title)
            self._save_metadata()
            return self.record

    def finish(self, status: Literal["closed", "interrupted"]) -> SessionRecord:
        with self._mutex, _storage_errors():
            self._check_open()
            self._meta.last_terminal_state = status
            self._save_metadata()
            return self.record

    def save_compact_state(self, state: CompactState) -> None:
        with self._mutex, _storage_errors():
            self._check_open()
            _validate_compact(state, self._message_count)
            write_json_snapshot(
                self.storage.project_directory, self.layout.compact_path, state.model_dump(mode="json"),
            )

    def load_or_reset_compact_state(self) -> CompactStateLoadResult:
        with self._mutex, _storage_errors():
            self._check_open()
            storage = self.store.project_storage(self.storage.identity)
            self.store._metadata(storage, self.record.id)
            try:
                state = _load_compact_snapshot(storage, self.record.id, self._message_count)
                return CompactStateLoadResult(state)
            except CompactStateDataError:
                state = CompactState(
                    consecutive_failure_count=1,
                    retry_after_message_count=self._message_count + DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES,
                    last_failure_reason="stored_compact_state_invalid",
                )
                self.save_compact_state(state)
                return CompactStateLoadResult(state, True)

    def append_subagent_event(self, run_id: str, event: dict[str, object]) -> None:
        with self._mutex, _storage_errors():
            self._check_open()
            if event.get("type") not in {"state", "snapshot", "tool_audit", "result"}:
                raise SessionDataError("Unsupported SubAgent event.")
            append_jsonl_record(
                self.storage.project_directory, self.layout.subagent_log_path(run_id),
                {**event, "version": 1},
            )


def _title(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise SessionDataError("Invalid session title.")
    return value.strip()


def _transcript(message: Message, sequence: int) -> dict[str, object]:
    record = _TranscriptRecord(
        sequence=sequence, role=message.role, content=message.content,
        tool_calls=[_ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in message.tool_calls],
        tool_call_id=message.tool_call_id, reasoning_content=message.reasoning_content,
        reasoning_state=message.reasoning_state,
    )
    payload = record.model_dump(mode="json")
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as error:
        raise SessionDataError("Tool arguments must be finite JSON.") from error
    return payload


def _messages(records: list[dict[str, object]]) -> list[Message]:
    messages = []
    for sequence, raw in enumerate(records, 1):
        try:
            json.dumps(raw, allow_nan=False)
        except ValueError as error:
            raise SessionDataError("Transcript contains nonfinite JSON.") from error
        record = _TranscriptRecord.model_validate(raw)
        if record.sequence != sequence:
            raise SessionDataError("Transcript sequence is not contiguous.")
        messages.append(record.message())
    return messages


def _load_compact_snapshot(
    storage: ProjectStorage, session_id: str, message_count: int,
) -> CompactState:
    path = storage.session(session_id).compact_path
    try:
        path.lstat()
    except FileNotFoundError:
        return CompactState()
    try:
        payload = read_json_snapshot(storage.project_directory, path)
    except JsonSnapshotError as error:
        if isinstance(error.__cause__, OSError):
            raise  # I/O failure is not evidence of corrupt Compact data.
        raise CompactStateDataError("Invalid Compact JSON snapshot.") from error
    try:
        state = CompactState.model_validate(payload)
    except ValidationError as error:
        raise CompactStateDataError("Invalid Compact schema.") from error
    _validate_compact(state, message_count)
    return state


def _validate_compact(state: CompactState, message_count: int) -> None:
    if len(state.model_dump_json()) > MAX_COMPACT_STATE_JSON_CHARS:
        raise CompactStateDataError("Session Compact state exceeds size limit.")
    if state.boundary and state.boundary.covered_message_count > message_count:
        raise CompactStateDataError("Compact boundary exceeds persisted history.")


def _validate_deletion_tree(path: Path) -> None:
    import stat

    info = path.lstat()
    if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0,
    ):
        raise SessionDataError("Session deletion refuses links/reparse points.")
    if path.is_dir():
        for child in path.iterdir():
            _validate_deletion_tree(child)
