from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Literal, cast
from uuid import uuid4

from pydantic import ValidationError

from mycode.agent import AgentToolCall
from mycode.context_compact import (
    DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES,
    CompactState,
)
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.reasoning import ReasoningState
from mycode.session_lock import (
    SessionOperationLock,
    database_maintenance_lock_path,
    session_operation_lock_path,
)


SESSION_SCHEMA_VERSION = 8
DEFAULT_SESSION_TITLE = "New session"
MAX_SESSION_TITLE_CHARS = 200
DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS = 30.0
DEFAULT_SESSION_LEASE_SECONDS = 30.0
SQLITE_JOURNAL_MODE = "wal"
DEFAULT_SUBAGENT_RUN_RETENTION = 100
DEFAULT_SUBAGENT_EVENT_LIMIT = 100
DEFAULT_SUBAGENT_TOOL_AUDIT_LIMIT = 200
MAX_SUBAGENT_EVENT_DATA_CHARS = 4000
MAX_SUBAGENT_SNAPSHOT_JSON_CHARS = 4000
MAX_SUBAGENT_CONTEXT_JSON_CHARS = 4000
MAX_SUBAGENT_TOKEN_USAGE_JSON_CHARS = 1000
MAX_SUBAGENT_RESULT_JSON_CHARS = 16000
MAX_SUBAGENT_ARGUMENT_SUMMARY_CHARS = 2000
MAX_COMPACT_STATE_JSON_CHARS = 20000
MAX_SESSION_DELETION_ERROR_CODE_CHARS = 100

SessionStatus = Literal["active", "closed", "interrupted"]
SESSION_STATUSES = frozenset({"active", "closed", "interrupted"})
SessionDeletionStage = Literal[
    "pending",
    "database_deleted",
    "initial_checkpoint_complete",
    "vacuum_complete",
    "final_checkpoint_complete",
]
SESSION_DELETION_STAGES = frozenset(
    {
        "pending",
        "database_deleted",
        "initial_checkpoint_complete",
        "vacuum_complete",
        "final_checkpoint_complete",
    }
)
PERSISTED_MESSAGE_ROLES = frozenset({"user", "assistant", "tool"})
SubAgentRunStatus = Literal[
    "running",
    "awaiting_confirmation",
    "completed",
    "failed",
    "interrupted",
]
SUBAGENT_RUN_STATUSES = frozenset(
    {"running", "awaiting_confirmation", "completed", "failed", "interrupted"}
)
SUBAGENT_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
SUBAGENT_ROLES = frozenset({"explorer", "tester", "reviewer"})


class SessionStoreError(RuntimeError):
    pass


class SessionNotFoundError(SessionStoreError):
    pass


class SessionInUseError(SessionStoreError):
    pass


class SessionLeaseLostError(SessionStoreError):
    pass


class SessionDeletingError(SessionStoreError):
    pass


class SessionDataError(SessionStoreError):
    pass


class SessionDatabaseCorruptionError(SessionStoreError):
    pass


class SessionMaintenanceError(SessionStoreError):
    pass


class UnsupportedSessionSchemaError(SessionStoreError):
    pass


@dataclass(frozen=True)
class SessionRecord:
    id: str
    project_key: str
    workspace_root: Path
    title: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SessionDeletionRecord:
    id: str
    session_id: str
    project_key: str
    stage: SessionDeletionStage
    artifact_present: bool | None
    retry_count: int
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DatabaseMaintenanceState:
    post_delete_scrub_required: bool
    retry_count: int
    last_error_code: str | None
    updated_at: datetime


@dataclass(frozen=True)
class CompactStateLoadResult:
    state: CompactState
    recovered_invalid_state: bool = False


@dataclass(frozen=True)
class SubAgentRunRecord:
    id: str
    parent_session_id: str
    project_key: str
    role: str
    status: SubAgentRunStatus
    stop_reason: str | None
    task_sha256: str
    objective_chars: int
    context_chars: int
    scope_path_count: int
    snapshot: dict[str, object] | None
    context: dict[str, object] | None
    token_usage: dict[str, object] | None
    tool_call_count: int
    validation_execution_count: int
    omitted_event_count: int
    omitted_tool_audit_count: int
    result: dict[str, object] | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class SubAgentEventRecord:
    run_id: str
    sequence: int
    event_type: str
    state: str | None
    reason: str | None
    data: dict[str, object]
    occurred_at: datetime


@dataclass(frozen=True)
class SubAgentToolAuditRecord:
    run_id: str
    sequence: int
    tool_name: str
    arguments_sha256: str
    argument_summary: dict[str, object]
    ok: bool
    exit_code: int | None
    duration_ms: int | None
    output_chars: int
    truncated: bool
    reason: str | None
    occurred_at: datetime


def default_session_database_path() -> Path:
    return Path.home() / ".mycode" / "state.sqlite3"


class SessionStore:
    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        busy_timeout_seconds: float = DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS,
        session_lock_timeout_seconds: float | None = None,
        subagent_run_retention: int = DEFAULT_SUBAGENT_RUN_RETENTION,
        subagent_event_limit: int = DEFAULT_SUBAGENT_EVENT_LIMIT,
        subagent_tool_audit_limit: int = DEFAULT_SUBAGENT_TOOL_AUDIT_LIMIT,
    ) -> None:
        if not math.isfinite(busy_timeout_seconds) or busy_timeout_seconds < 0:
            raise ValueError(
                "busy_timeout_seconds must be a finite number at least 0."
            )
        effective_lock_timeout = (
            busy_timeout_seconds
            if session_lock_timeout_seconds is None
            else session_lock_timeout_seconds
        )
        if (
            not math.isfinite(effective_lock_timeout)
            or effective_lock_timeout < 0
        ):
            raise ValueError(
                "session_lock_timeout_seconds must be a finite number at least 0."
            )
        self.database_path = Path(
            default_session_database_path()
            if database_path is None
            else database_path
        ).resolve(strict=False)
        self._now = _utc_now if now is None else now
        self.busy_timeout_seconds = busy_timeout_seconds
        self.session_lock_timeout_seconds = effective_lock_timeout
        self.subagent_run_retention = _validate_positive_limit(
            subagent_run_retention,
            field_name="subagent_run_retention",
        )
        self.subagent_event_limit = _validate_positive_limit(
            subagent_event_limit,
            field_name="subagent_event_limit",
        )
        self.subagent_tool_audit_limit = _validate_positive_limit(
            subagent_tool_audit_limit,
            field_name="subagent_tool_audit_limit",
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def session_operation_lock(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> Iterator[None]:
        with self.session_operation_lock_for_key(project.key, session_id):
            yield

    @contextmanager
    def session_operation_lock_for_key(
        self,
        project_key: str,
        session_id: str,
    ) -> Iterator[None]:
        normalized_project_key = _validate_sha256(
            project_key,
            field_name="project_key",
        )
        identifier = _validate_session_id(session_id)
        lock = SessionOperationLock(
            session_operation_lock_path(
                self.database_path.parent,
                project_key=normalized_project_key,
                session_id=identifier,
            ),
            timeout_seconds=self.session_lock_timeout_seconds,
        )
        with lock.acquire():
            yield

    @contextmanager
    def database_maintenance_lock(self) -> Iterator[None]:
        lock = SessionOperationLock(
            database_maintenance_lock_path(self.database_path.parent),
            timeout_seconds=self.session_lock_timeout_seconds,
        )
        with lock.acquire():
            yield

    def _begin_immediate_at(
        self,
        connection: sqlite3.Connection,
    ) -> datetime:
        """Acquire SQLite's write lock before sampling lease-sensitive time."""
        connection.execute("BEGIN IMMEDIATE")
        return _normalize_datetime(self._now())

    @contextmanager
    def artifact_write_guard(
        self,
        project: ProjectIdentity,
        session_id: str,
        lease_owner_id: str,
    ) -> Iterator[None]:
        identifier = _validate_session_id(session_id)
        owner_id = _validate_lease_owner_id(lease_owner_id)
        with self.session_operation_lock(project, identifier):
            with self._connect() as connection:
                now = self._begin_immediate_at(connection)
                _require_session(connection, project, identifier)
                _require_current_session_lease(
                    connection,
                    identifier,
                    owner_id,
                    now,
                )
            yield

    def create_session(
        self,
        project: ProjectIdentity,
        *,
        title: str = DEFAULT_SESSION_TITLE,
        session_id: str | None = None,
        lease_owner_id: str | None = None,
        lease_duration_seconds: float = DEFAULT_SESSION_LEASE_SECONDS,
    ) -> SessionRecord:
        normalized_title = _validate_title(title)
        identifier = str(uuid4()) if session_id is None else _validate_session_id(session_id)
        owner_id = (
            None if lease_owner_id is None else _validate_lease_owner_id(lease_owner_id)
        )
        duration = _validate_lease_duration(lease_duration_seconds)
        initial_status: SessionStatus = "closed" if owner_id is None else "active"
        try:
            maintenance_guard = (
                self.database_maintenance_lock()
                if owner_id is not None
                else nullcontext()
            )
            with maintenance_guard:
                with self.session_operation_lock(project, identifier):
                    with self._connect() as connection:
                        now = self._begin_immediate_at(connection)
                        _require_no_session_deletion(
                            connection,
                            project,
                            identifier,
                        )
                        connection.execute(
                            """
                            INSERT INTO sessions (
                                id,
                                project_key,
                                workspace_root,
                                title,
                                status,
                                created_at,
                                updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                identifier,
                                project.key,
                                str(project.workspace_root),
                                normalized_title,
                                initial_status,
                                _datetime_to_text(now),
                                _datetime_to_text(now),
                            ),
                        )
                        if owner_id is not None:
                            _insert_session_lease(
                                connection,
                                identifier,
                                owner_id,
                                now,
                                duration,
                            )
        except sqlite3.IntegrityError as error:
            raise SessionStoreError(f"Session already exists: {identifier}") from error

        return SessionRecord(
            id=identifier,
            project_key=project.key,
            workspace_root=project.workspace_root,
            title=normalized_title,
            status=initial_status,
            created_at=now,
            updated_at=now,
        )

    def list_sessions(
        self,
        project: ProjectIdentity,
        *,
        limit: int = 20,
    ) -> list[SessionRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, project_key, workspace_root, title, status,
                       created_at, updated_at
                FROM sessions AS candidate
                WHERE project_key = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM session_deletion_tasks AS deletion
                      WHERE deletion.session_id = candidate.id
                        AND deletion.project_key = candidate.project_key
                  )
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (project.key, limit),
            ).fetchall()

        return [_session_from_row(row) for row in rows]

    def get_session(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> SessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT candidate.id, candidate.project_key,
                       candidate.workspace_root, candidate.title,
                       candidate.status, candidate.created_at,
                       candidate.updated_at
                FROM sessions AS candidate
                WHERE candidate.id = ? AND candidate.project_key = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM session_deletion_tasks AS deletion
                      WHERE deletion.session_id = candidate.id
                        AND deletion.project_key = candidate.project_key
                  )
                """,
                (_validate_session_id(session_id), project.key),
            ).fetchone()

        return None if row is None else _session_from_row(row)

    def append_message(
        self,
        project: ProjectIdentity,
        session_id: str,
        message: Message,
        *,
        lease_owner_id: str | None = None,
    ) -> int:
        sequences = self.append_messages(
            project,
            session_id,
            [message],
            lease_owner_id=lease_owner_id,
        )
        return sequences[0]

    def append_messages(
        self,
        project: ProjectIdentity,
        session_id: str,
        messages: Iterable[Message],
        *,
        lease_owner_id: str | None = None,
    ) -> list[int]:
        identifier = _validate_session_id(session_id)
        owner_id = (
            None if lease_owner_id is None else _validate_lease_owner_id(lease_owner_id)
        )
        prepared = [_prepare_message(message) for message in messages]
        if not prepared:
            return []
        try:
            with self._connect() as connection:
                now = self._begin_immediate_at(connection)
                _require_session(connection, project, identifier)
                if owner_id is not None:
                    _require_current_session_lease(
                        connection,
                        identifier,
                        owner_id,
                        now,
                    )
                else:
                    _require_session_not_in_use(
                        connection,
                        identifier,
                        now,
                        subagent_event_limit=self.subagent_event_limit,
                        subagent_run_retention=self.subagent_run_retention,
                    )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), -1) + 1 AS next_sequence
                    FROM session_messages
                    WHERE session_id = ?
                    """,
                    (identifier,),
                ).fetchone()
                next_sequence = int(row["next_sequence"])
                sequences: list[int] = []
                for offset, item in enumerate(prepared):
                    sequence = next_sequence + offset
                    connection.execute(
                        """
                        INSERT INTO session_messages (
                            session_id,
                            sequence,
                            role,
                            content,
                            tool_calls_json,
                            tool_call_id,
                            reasoning_content,
                            reasoning_state,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identifier,
                            sequence,
                            item.role,
                            item.content,
                            item.tool_calls_json,
                            item.tool_call_id,
                            item.reasoning_content,
                            item.reasoning_state,
                            _datetime_to_text(now),
                        ),
                    )
                    sequences.append(sequence)

                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (_datetime_to_text(now), identifier),
                )
        except sqlite3.IntegrityError as error:
            raise SessionStoreError(
                "Failed to append session messages because their sequence "
                "conflicted with another writer."
            ) from error

        return sequences

    def load_conversation(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> Conversation:
        identifier = _validate_session_id(session_id)
        with self._connect() as connection:
            _require_session(connection, project, identifier)
            rows = connection.execute(
                """
                SELECT sequence, role, content, tool_calls_json, tool_call_id,
                       reasoning_content, reasoning_state
                FROM session_messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (identifier,),
            ).fetchall()

        messages = [_message_from_row(row) for row in rows]
        return Conversation.from_messages(messages)

    def load_compact_state(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> CompactState:
        identifier = _validate_session_id(session_id)
        with self._connect() as connection:
            _require_session(connection, project, identifier)
            row = connection.execute(
                """
                SELECT state_json
                FROM session_compact_state
                WHERE session_id = ?
                """,
                (identifier,),
            ).fetchone()

        if row is None:
            return CompactState()
        try:
            return CompactState.model_validate_json(str(row["state_json"]))
        except (ValidationError, ValueError) as error:
            raise SessionDataError(
                "Stored session Compact state is invalid."
            ) from error

    def load_or_reset_compact_state(
        self,
        project: ProjectIdentity,
        session_id: str,
        *,
        lease_owner_id: str,
    ) -> CompactStateLoadResult:
        identifier = _validate_session_id(session_id)
        owner_id = _validate_lease_owner_id(lease_owner_id)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            _require_session(connection, project, identifier)
            _require_current_session_lease(
                connection,
                identifier,
                owner_id,
                now,
            )
            row = connection.execute(
                """
                SELECT state_json
                FROM session_compact_state
                WHERE session_id = ?
                """,
                (identifier,),
            ).fetchone()
            if row is None:
                return CompactStateLoadResult(state=CompactState())

            try:
                state = CompactState.model_validate_json(str(row["state_json"]))
            except (ValidationError, ValueError):
                message_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM session_messages
                        WHERE session_id = ?
                        """,
                        (identifier,),
                    ).fetchone()[0]
                )
                state = CompactState(
                    boundary=None,
                    consecutive_failure_count=1,
                    retry_after_message_count=(
                        message_count + DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES
                    ),
                    last_failure_reason="stored_compact_state_invalid",
                )
                connection.execute(
                    """
                    UPDATE session_compact_state
                    SET state_json = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        state.model_dump_json(),
                        _datetime_to_text(now),
                        identifier,
                    ),
                )
                return CompactStateLoadResult(
                    state=state,
                    recovered_invalid_state=True,
                )
        return CompactStateLoadResult(state=state)

    def save_compact_state(
        self,
        project: ProjectIdentity,
        session_id: str,
        state: CompactState,
        *,
        lease_owner_id: str | None = None,
    ) -> None:
        identifier = _validate_session_id(session_id)
        owner_id = (
            None if lease_owner_id is None else _validate_lease_owner_id(lease_owner_id)
        )
        serialized = state.model_dump_json()
        if len(serialized) > MAX_COMPACT_STATE_JSON_CHARS:
            raise SessionDataError(
                "Session Compact state exceeds the persisted size limit."
            )
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            _require_session(connection, project, identifier)
            if owner_id is not None:
                _require_current_session_lease(
                    connection,
                    identifier,
                    owner_id,
                    now,
                )
            else:
                _require_session_not_in_use(
                    connection,
                    identifier,
                    now,
                    subagent_event_limit=self.subagent_event_limit,
                    subagent_run_retention=self.subagent_run_retention,
                )

            if state.boundary is not None:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS message_count
                    FROM session_messages
                    WHERE session_id = ?
                    """,
                    (identifier,),
                ).fetchone()
                if state.boundary.covered_message_count > int(
                    row["message_count"]
                ):
                    raise SessionDataError(
                        "Compact boundary exceeds the persisted session history."
                    )

            connection.execute(
                """
                INSERT INTO session_compact_state (
                    session_id,
                    state_json,
                    updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    identifier,
                    serialized,
                    _datetime_to_text(now),
                ),
            )

    def rename_session(
        self,
        project: ProjectIdentity,
        session_id: str,
        title: str,
        *,
        lease_owner_id: str | None = None,
    ) -> SessionRecord:
        identifier = _validate_session_id(session_id)
        owner_id = (
            None if lease_owner_id is None else _validate_lease_owner_id(lease_owner_id)
        )
        normalized_title = _validate_title(title)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            _require_session(connection, project, identifier)
            if owner_id is not None:
                _require_current_session_lease(
                    connection,
                    identifier,
                    owner_id,
                    now,
                )
            else:
                _require_session_not_in_use(
                    connection,
                    identifier,
                    now,
                    subagent_event_limit=self.subagent_event_limit,
                    subagent_run_retention=self.subagent_run_retention,
                )
            connection.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (normalized_title, _datetime_to_text(now), identifier),
            )

        record = self.get_session(project, identifier)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {identifier}")
        return record

    def acquire_session_lease(
        self,
        project: ProjectIdentity,
        session_id: str,
        owner_id: str,
        *,
        lease_duration_seconds: float = DEFAULT_SESSION_LEASE_SECONDS,
    ) -> SessionRecord:
        identifier = _validate_session_id(session_id)
        normalized_owner_id = _validate_lease_owner_id(owner_id)
        duration = _validate_lease_duration(lease_duration_seconds)

        with self.database_maintenance_lock():
            with self.session_operation_lock(project, identifier):
                with self._connect() as connection:
                    now = self._begin_immediate_at(connection)
                    _require_session(connection, project, identifier)
                    existing = connection.execute(
                        """
                        SELECT owner_id, expires_at
                        FROM session_leases
                        WHERE session_id = ?
                        """,
                        (identifier,),
                    ).fetchone()
                    if existing is not None:
                        existing_owner_id = str(existing["owner_id"])
                        expires_at = _datetime_from_text(str(existing["expires_at"]))
                        if (
                            expires_at > now
                            and existing_owner_id != normalized_owner_id
                        ):
                            raise SessionInUseError(
                                "Session is already in use by another agent: "
                                f"{identifier}"
                            )
                        connection.execute(
                            "DELETE FROM session_leases WHERE session_id = ?",
                            (identifier,),
                        )
                    _interrupt_unfinished_subagent_runs_locked(
                        connection,
                        identifier,
                        now,
                        reason="parent_session_resumed",
                        event_limit=self.subagent_event_limit,
                        run_retention=self.subagent_run_retention,
                    )
                    _insert_session_lease(
                        connection,
                        identifier,
                        normalized_owner_id,
                        now,
                        duration,
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = 'active', updated_at = ?
                        WHERE id = ?
                        """,
                        (_datetime_to_text(now), identifier),
                    )
                record = self.get_session(project, identifier)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {identifier}")
        return record

    def renew_session_lease(
        self,
        project: ProjectIdentity,
        session_id: str,
        owner_id: str,
        *,
        lease_duration_seconds: float = DEFAULT_SESSION_LEASE_SECONDS,
    ) -> None:
        identifier = _validate_session_id(session_id)
        normalized_owner_id = _validate_lease_owner_id(owner_id)
        duration = _validate_lease_duration(lease_duration_seconds)

        # Heartbeats must not wait behind a potentially long artifact fsync.
        # BEGIN IMMEDIATE still serializes this renewal with creation of a
        # deletion tombstone, so either the lease is renewed first and deletion
        # observes it as live, or the tombstone commits first and renewal fails.
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            _require_session(connection, project, identifier)
            _require_current_session_lease(
                connection,
                identifier,
                normalized_owner_id,
                now,
            )
            connection.execute(
                """
                UPDATE session_leases
                SET heartbeat_at = ?, expires_at = ?
                WHERE session_id = ? AND owner_id = ?
                """,
                (
                    _datetime_to_text(now),
                    _datetime_to_text(now + timedelta(seconds=duration)),
                    identifier,
                    normalized_owner_id,
                )
            )

    def release_session_lease(
        self,
        project: ProjectIdentity,
        session_id: str,
        owner_id: str,
        status: SessionStatus,
    ) -> SessionRecord:
        if status not in {"closed", "interrupted"}:
            raise ValueError("Released sessions must be closed or interrupted.")
        identifier = _validate_session_id(session_id)
        normalized_owner_id = _validate_lease_owner_id(owner_id)

        with self.session_operation_lock(project, identifier):
            with self._connect() as connection:
                now = self._begin_immediate_at(connection)
                _require_session(connection, project, identifier)
                _require_current_session_lease(
                    connection,
                    identifier,
                    normalized_owner_id,
                    now,
                    allow_expired=True,
                )
                _interrupt_unfinished_subagent_runs_locked(
                    connection,
                    identifier,
                    now,
                    reason=f"parent_session_{status}",
                    event_limit=self.subagent_event_limit,
                    run_retention=self.subagent_run_retention,
                )
                connection.execute(
                    """
                    DELETE FROM session_leases
                    WHERE session_id = ? AND owner_id = ?
                    """,
                    (identifier, normalized_owner_id),
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, _datetime_to_text(now), identifier),
                )
            record = self.get_session(project, identifier)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {identifier}")
        return record

    def expire_session_leases(self, project: ProjectIdentity) -> int:
        now = _normalize_datetime(self._now())
        now_text = _datetime_to_text(now)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_leases.session_id
                FROM session_leases
                JOIN sessions ON sessions.id = session_leases.session_id
                WHERE sessions.project_key = ?
                  AND session_leases.expires_at <= ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM session_deletion_tasks AS deletion
                      WHERE deletion.session_id = sessions.id
                        AND deletion.project_key = sessions.project_key
                  )
                """,
                (project.key, now_text),
            ).fetchall()
        identifiers = [str(row["session_id"]) for row in rows]
        expired_count = 0
        for identifier in identifiers:
            with self.session_operation_lock(project, identifier):
                with self._connect() as connection:
                    transaction_now = self._begin_immediate_at(connection)
                    transaction_now_text = _datetime_to_text(transaction_now)
                    row = connection.execute(
                        """
                        SELECT 1
                        FROM session_leases
                        JOIN sessions ON sessions.id = session_leases.session_id
                        WHERE sessions.id = ?
                          AND sessions.project_key = ?
                          AND session_leases.expires_at <= ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM session_deletion_tasks AS deletion
                              WHERE deletion.session_id = sessions.id
                                AND deletion.project_key = sessions.project_key
                          )
                        """,
                        (identifier, project.key, transaction_now_text),
                    ).fetchone()
                    if row is None:
                        continue
                    _interrupt_unfinished_subagent_runs_locked(
                        connection,
                        identifier,
                        transaction_now,
                        reason="parent_lease_expired",
                        event_limit=self.subagent_event_limit,
                        run_retention=self.subagent_run_retention,
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = 'interrupted'
                        WHERE id = ?
                        """,
                        (identifier,),
                    )
                    connection.execute(
                        "DELETE FROM session_leases WHERE session_id = ?",
                        (identifier,),
                    )
                    expired_count += 1
        return expired_count

    def set_status(
        self,
        project: ProjectIdentity,
        session_id: str,
        status: SessionStatus,
    ) -> SessionRecord:
        if status not in SESSION_STATUSES:
            raise ValueError(f"Unsupported session status: {status}")
        if status == "active":
            raise ValueError("Use acquire_session_lease() to activate a session.")
        identifier = _validate_session_id(session_id)
        with self.session_operation_lock(project, identifier):
            with self._connect() as connection:
                now = self._begin_immediate_at(connection)
                _require_session(connection, project, identifier)
                _require_session_not_in_use(
                    connection,
                    identifier,
                    now,
                    subagent_event_limit=self.subagent_event_limit,
                    subagent_run_retention=self.subagent_run_retention,
                )
                _interrupt_unfinished_subagent_runs_locked(
                    connection,
                    identifier,
                    now,
                    reason=f"parent_session_{status}",
                    event_limit=self.subagent_event_limit,
                    run_retention=self.subagent_run_retention,
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, _datetime_to_text(now), identifier),
                )
            record = self.get_session(project, identifier)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {identifier}")
        return record

    def request_session_deletion(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> SessionDeletionRecord | None:
        identifier = _validate_session_id(session_id)
        with self.session_operation_lock(project, identifier):
            with self._connect() as connection:
                now = self._begin_immediate_at(connection)
                existing_task = connection.execute(
                    """
                    SELECT *
                    FROM session_deletion_tasks
                    WHERE session_id = ? AND project_key = ?
                    """,
                    (identifier, project.key),
                ).fetchone()
                if existing_task is not None:
                    return _session_deletion_from_row(existing_task)

                existing_session = connection.execute(
                    """
                    SELECT 1
                    FROM sessions
                    WHERE id = ? AND project_key = ?
                    """,
                    (identifier, project.key),
                ).fetchone()
                if existing_session is None:
                    return None
                _require_session_not_in_use(
                    connection,
                    identifier,
                    now,
                    subagent_event_limit=self.subagent_event_limit,
                    subagent_run_retention=self.subagent_run_retention,
                )
                deletion_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO session_deletion_tasks (
                        id,
                        session_id,
                        project_key,
                        stage,
                        artifact_present,
                        retry_count,
                        last_error_code,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, 'pending', NULL, 0, NULL, ?, ?)
                    """,
                    (
                        deletion_id,
                        identifier,
                        project.key,
                        _datetime_to_text(now),
                        _datetime_to_text(now),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT *
                    FROM session_deletion_tasks
                    WHERE id = ?
                    """,
                    (deletion_id,),
                ).fetchone()
        if row is None:
            raise SessionStoreError("Session deletion task was not persisted.")
        return _session_deletion_from_row(row)

    def get_session_deletion(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> SessionDeletionRecord | None:
        identifier = _validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM session_deletion_tasks
                WHERE session_id = ? AND project_key = ?
                """,
                (identifier, project.key),
            ).fetchone()
        return None if row is None else _session_deletion_from_row(row)

    def list_session_deletions(
        self,
        project: ProjectIdentity,
    ) -> list[SessionDeletionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM session_deletion_tasks
                WHERE project_key = ?
                ORDER BY created_at ASC, id ASC
                """,
                (project.key,),
            ).fetchall()
        return [_session_deletion_from_row(row) for row in rows]

    def get_session_deletion_by_id(
        self,
        deletion_id: str,
    ) -> SessionDeletionRecord | None:
        identifier = _validate_session_id(deletion_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM session_deletion_tasks
                WHERE id = ?
                """,
                (identifier,),
            ).fetchone()
        return None if row is None else _session_deletion_from_row(row)

    def list_all_session_deletions(self) -> list[SessionDeletionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM session_deletion_tasks
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [_session_deletion_from_row(row) for row in rows]

    def delete_session_records_for_deletion_locked(
        self,
        deletion_id: str,
        *,
        artifact_present: bool,
    ) -> SessionDeletionRecord:
        identifier = _validate_session_id(deletion_id)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            task = _require_session_deletion(connection, identifier)
            if task.stage != "pending":
                return task
            connection.execute(
                """
                DELETE FROM sessions
                WHERE id = ? AND project_key = ?
                """,
                (task.session_id, task.project_key),
            )
            connection.execute(
                """
                UPDATE session_deletion_tasks
                SET stage = 'database_deleted',
                    artifact_present = ?,
                    last_error_code = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if artifact_present else 0,
                    _datetime_to_text(now),
                    identifier,
                ),
            )
            row = connection.execute(
                "SELECT * FROM session_deletion_tasks WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise SessionStoreError("Session deletion task disappeared.")
        return _session_deletion_from_row(row)

    def record_session_deletion_failure(
        self,
        deletion_id: str,
        *,
        error_code: str,
    ) -> SessionDeletionRecord:
        identifier = _validate_session_id(deletion_id)
        safe_error_code = _validate_deletion_error_code(error_code)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            _require_session_deletion(connection, identifier)
            connection.execute(
                """
                UPDATE session_deletion_tasks
                SET retry_count = retry_count + 1,
                    last_error_code = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    safe_error_code,
                    _datetime_to_text(now),
                    identifier,
                ),
            )
            row = connection.execute(
                "SELECT * FROM session_deletion_tasks WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise SessionStoreError("Session deletion task disappeared.")
        return _session_deletion_from_row(row)

    def load_database_maintenance_state(self) -> DatabaseMaintenanceState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM database_maintenance_state WHERE id = 1"
            ).fetchone()
        if row is None:
            raise SessionStoreError("Database maintenance state is missing.")
        return _database_maintenance_state_from_row(row)

    def retire_session_deletion_for_scrub(
        self,
        deletion_id: str,
    ) -> bool:
        """Remove a target-bearing tombstone and atomically arm generic scrub."""
        identifier = _validate_session_id(deletion_id)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            cursor = connection.execute(
                "DELETE FROM session_deletion_tasks WHERE id = ?",
                (identifier,),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                """
                UPDATE database_maintenance_state
                SET post_delete_scrub_required = 1,
                    retry_count = 0,
                    last_error_code = NULL,
                    updated_at = ?
                WHERE id = 1
                """,
                (_datetime_to_text(now),),
            )
        return True

    def record_post_delete_scrub_failure(
        self,
        error_code: str,
    ) -> DatabaseMaintenanceState:
        safe_error_code = _validate_deletion_error_code(error_code)
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            connection.execute(
                """
                UPDATE database_maintenance_state
                SET post_delete_scrub_required = 1,
                    retry_count = retry_count + 1,
                    last_error_code = ?,
                    updated_at = ?
                WHERE id = 1
                """,
                (safe_error_code, _datetime_to_text(now)),
            )
            row = connection.execute(
                "SELECT * FROM database_maintenance_state WHERE id = 1"
            ).fetchone()
        if row is None:
            raise SessionStoreError("Database maintenance state is missing.")
        return _database_maintenance_state_from_row(row)

    def mark_post_delete_scrub_complete(self) -> None:
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            connection.execute(
                """
                UPDATE database_maintenance_state
                SET post_delete_scrub_required = 0,
                    retry_count = 0,
                    last_error_code = NULL,
                    updated_at = ?
                WHERE id = 1
                """,
                (_datetime_to_text(now),),
            )

    def has_active_session_leases_for_maintenance(self) -> bool:
        """Check live owners while holding SQLite's writer slot."""
        with self._connect() as connection:
            now = self._begin_immediate_at(connection)
            row = connection.execute(
                """
                SELECT 1
                FROM session_leases
                WHERE expires_at > ?
                LIMIT 1
                """,
                (_datetime_to_text(now),),
            ).fetchone()
        return row is not None

    def checkpoint_wal_truncate(self) -> None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
        except SessionDatabaseCorruptionError:
            raise
        except SessionStoreError as error:
            raise SessionMaintenanceError(
                "SQLite WAL checkpoint is temporarily unavailable."
            ) from error
        if row is None:
            raise SessionMaintenanceError(
                "SQLite WAL checkpoint returned no status."
            )
        busy, remaining_frames, checkpointed_frames = (
            int(row[0]),
            int(row[1]),
            int(row[2]),
        )
        if busy != 0 or remaining_frames != 0 or checkpointed_frames != 0:
            raise SessionMaintenanceError(
                "SQLite WAL checkpoint could not truncate the log."
            )
        wal_path = Path(f"{self.database_path}-wal")
        if wal_path.exists() and wal_path.stat().st_size != 0:
            raise SessionMaintenanceError(
                "SQLite WAL file remains non-empty after checkpoint."
            )

    def vacuum_database(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("VACUUM")
        except SessionDatabaseCorruptionError:
            raise
        except SessionStoreError as error:
            raise SessionMaintenanceError(
                "SQLite VACUUM is temporarily unavailable."
            ) from error

    def create_subagent_run(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        *,
        run_id: str,
        role: str,
        task_sha256: str,
        objective_chars: int,
        context_chars: int,
        scope_path_count: int,
        reason: str,
        occurred_at: datetime,
        lease_owner_id: str,
    ) -> SubAgentRunRecord:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        normalized_role = _validate_subagent_role(role)
        digest = _validate_sha256(task_sha256, field_name="task_sha256")
        objective_size = _validate_non_negative_int(
            objective_chars,
            field_name="objective_chars",
        )
        context_size = _validate_non_negative_int(
            context_chars,
            field_name="context_chars",
        )
        scope_count = _validate_non_negative_int(
            scope_path_count,
            field_name="scope_path_count",
        )
        event_reason = _validate_short_text(reason, field_name="reason", max_chars=200)
        owner_id = _validate_lease_owner_id(lease_owner_id)
        occurred = _normalize_datetime(occurred_at)
        occurred_text = _datetime_to_text(occurred)

        try:
            with self._connect() as connection:
                lease_now = self._begin_immediate_at(connection)
                _require_session(connection, project, parent_id)
                _require_current_session_lease(
                    connection,
                    parent_id,
                    owner_id,
                    lease_now,
                )
                connection.execute(
                    """
                    INSERT INTO subagent_runs (
                        id,
                        parent_session_id,
                        project_key,
                        role,
                        status,
                        task_sha256,
                        objective_chars,
                        context_chars,
                        scope_path_count,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        parent_id,
                        project.key,
                        normalized_role,
                        digest,
                        objective_size,
                        context_size,
                        scope_count,
                        occurred_text,
                        occurred_text,
                    ),
                )
                _append_subagent_event_locked(
                    connection,
                    identifier,
                    event_type="state",
                    state="running",
                    reason=event_reason,
                    data_json="{}",
                    occurred_at=occurred_text,
                    limit=self.subagent_event_limit,
                )
                _prune_subagent_runs_locked(
                    connection,
                    parent_id,
                    retention=self.subagent_run_retention,
                )
        except sqlite3.IntegrityError as error:
            raise SessionStoreError(
                f"SubAgent run already exists or violates its parent session: {identifier}"
            ) from error

        record = self.get_subagent_run(project, parent_id, identifier)
        if record is None:
            raise SessionStoreError(f"SubAgent run was not created: {identifier}")
        return record

    def append_subagent_state(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
        *,
        state: str,
        reason: str,
        occurred_at: datetime,
        lease_owner_id: str,
    ) -> None:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        normalized_state = _validate_subagent_status(state)
        event_reason = _validate_short_text(reason, field_name="reason", max_chars=200)
        owner_id = _validate_lease_owner_id(lease_owner_id)
        occurred = _normalize_datetime(occurred_at)
        occurred_text = _datetime_to_text(occurred)

        with self._connect() as connection:
            lease_now = self._begin_immediate_at(connection)
            _require_session(connection, project, parent_id)
            _require_current_session_lease(
                connection,
                parent_id,
                owner_id,
                lease_now,
            )
            _require_writable_subagent_run(
                connection,
                project,
                parent_id,
                identifier,
            )
            _append_subagent_event_locked(
                connection,
                identifier,
                event_type="state",
                state=normalized_state,
                reason=event_reason,
                data_json="{}",
                occurred_at=occurred_text,
                limit=self.subagent_event_limit,
            )
            if normalized_state in {"running", "awaiting_confirmation", "interrupted"}:
                completed_at = occurred_text if normalized_state == "interrupted" else None
                stop_reason = "interrupted" if normalized_state == "interrupted" else None
                connection.execute(
                    """
                    UPDATE subagent_runs
                    SET status = ?, stop_reason = COALESCE(?, stop_reason),
                        completed_at = COALESCE(?, completed_at), updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized_state,
                        stop_reason,
                        completed_at,
                        occurred_text,
                        identifier,
                    ),
                )

    def update_subagent_snapshot(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
        *,
        snapshot: dict[str, object],
        occurred_at: datetime,
        lease_owner_id: str,
    ) -> None:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        snapshot_json = _bounded_json_object(
            snapshot,
            field_name="snapshot",
            max_chars=MAX_SUBAGENT_SNAPSHOT_JSON_CHARS,
        )
        combined_sha256 = _validate_sha256(
            str(snapshot.get("combined_sha256", "")),
            field_name="snapshot combined_sha256",
        )
        snapshot_event_json = _bounded_json_object(
            {
                "combined_sha256": combined_sha256,
                "snapshot_chars": len(snapshot_json),
            },
            field_name="snapshot_event",
            max_chars=MAX_SUBAGENT_EVENT_DATA_CHARS,
        )
        owner_id = _validate_lease_owner_id(lease_owner_id)
        occurred = _normalize_datetime(occurred_at)
        occurred_text = _datetime_to_text(occurred)

        with self._connect() as connection:
            lease_now = self._begin_immediate_at(connection)
            _require_session(connection, project, parent_id)
            _require_current_session_lease(
                connection,
                parent_id,
                owner_id,
                lease_now,
            )
            _require_writable_subagent_run(
                connection,
                project,
                parent_id,
                identifier,
            )
            connection.execute(
                "UPDATE subagent_runs SET snapshot_json = ?, updated_at = ? WHERE id = ?",
                (snapshot_json, occurred_text, identifier),
            )
            _append_subagent_event_locked(
                connection,
                identifier,
                event_type="snapshot",
                state=None,
                reason="snapshot_frozen",
                data_json=snapshot_event_json,
                occurred_at=occurred_text,
                limit=self.subagent_event_limit,
            )

    def append_subagent_tool_audit(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
        *,
        tool_name: str,
        arguments_sha256: str,
        argument_summary: dict[str, object],
        ok: bool,
        exit_code: int | None,
        duration_ms: int | None,
        output_chars: int,
        truncated: bool,
        reason: str | None,
        occurred_at: datetime,
        lease_owner_id: str,
    ) -> None:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        normalized_tool_name = _validate_short_text(
            tool_name,
            field_name="tool_name",
            max_chars=100,
        )
        digest = _validate_sha256(
            arguments_sha256,
            field_name="arguments_sha256",
        )
        summary_json = _bounded_json_object(
            argument_summary,
            field_name="argument_summary",
            max_chars=MAX_SUBAGENT_ARGUMENT_SUMMARY_CHARS,
        )
        normalized_exit_code = _validate_optional_int(
            exit_code,
            field_name="exit_code",
        )
        normalized_duration = _validate_optional_non_negative_int(
            duration_ms,
            field_name="duration_ms",
        )
        output_size = _validate_non_negative_int(
            output_chars,
            field_name="output_chars",
        )
        normalized_reason = (
            None
            if reason is None
            else _validate_short_text(reason, field_name="reason", max_chars=100)
        )
        if not isinstance(ok, bool) or not isinstance(truncated, bool):
            raise ValueError("ok and truncated must be booleans.")
        owner_id = _validate_lease_owner_id(lease_owner_id)
        occurred = _normalize_datetime(occurred_at)
        occurred_text = _datetime_to_text(occurred)

        with self._connect() as connection:
            lease_now = self._begin_immediate_at(connection)
            _require_session(connection, project, parent_id)
            _require_current_session_lease(
                connection,
                parent_id,
                owner_id,
                lease_now,
            )
            _require_writable_subagent_run(
                connection,
                project,
                parent_id,
                identifier,
            )
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM subagent_tool_audits WHERE run_id = ?",
                (identifier,),
            ).fetchone()
            if int(row["count"]) >= self.subagent_tool_audit_limit:
                connection.execute(
                    """
                    UPDATE subagent_runs
                    SET omitted_tool_audit_count = omitted_tool_audit_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (occurred_text, identifier),
                )
                return
            sequence = int(row["count"])
            connection.execute(
                """
                INSERT INTO subagent_tool_audits (
                    run_id, sequence, tool_name, arguments_sha256,
                    argument_summary_json, ok, exit_code, duration_ms,
                    output_chars, truncated, reason, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    sequence,
                    normalized_tool_name,
                    digest,
                    summary_json,
                    int(ok),
                    normalized_exit_code,
                    normalized_duration,
                    output_size,
                    int(truncated),
                    normalized_reason,
                    occurred_text,
                ),
            )
            connection.execute(
                "UPDATE subagent_runs SET updated_at = ? WHERE id = ?",
                (occurred_text, identifier),
            )

    def finalize_subagent_run(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
        *,
        status: str,
        stop_reason: str,
        result: dict[str, object],
        context: dict[str, object] | None,
        token_usage: dict[str, object] | None,
        tool_call_count: int,
        validation_execution_count: int,
        occurred_at: datetime,
        lease_owner_id: str,
    ) -> None:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        normalized_status = _validate_subagent_status(status)
        if normalized_status not in {"completed", "failed"}:
            raise ValueError("Finalized SubAgent runs must be completed or failed.")
        normalized_stop_reason = _validate_short_text(
            stop_reason,
            field_name="stop_reason",
            max_chars=100,
        )
        result_json = _bounded_json_object(
            result,
            field_name="result",
            max_chars=MAX_SUBAGENT_RESULT_JSON_CHARS,
        )
        context_json = (
            None
            if context is None
            else _bounded_json_object(
                context,
                field_name="context",
                max_chars=MAX_SUBAGENT_CONTEXT_JSON_CHARS,
            )
        )
        token_usage_json = (
            None
            if token_usage is None
            else _bounded_json_object(
                token_usage,
                field_name="token_usage",
                max_chars=MAX_SUBAGENT_TOKEN_USAGE_JSON_CHARS,
            )
        )
        tool_count = _validate_non_negative_int(
            tool_call_count,
            field_name="tool_call_count",
        )
        validation_count = _validate_non_negative_int(
            validation_execution_count,
            field_name="validation_execution_count",
        )
        owner_id = _validate_lease_owner_id(lease_owner_id)
        occurred = _normalize_datetime(occurred_at)
        occurred_text = _datetime_to_text(occurred)
        result_event_json = _bounded_json_object(
            {
                "status": normalized_status,
                "stop_reason": normalized_stop_reason,
                "result_chars": len(result_json),
            },
            field_name="result_event",
            max_chars=MAX_SUBAGENT_EVENT_DATA_CHARS,
        )

        with self._connect() as connection:
            lease_now = self._begin_immediate_at(connection)
            _require_session(connection, project, parent_id)
            _require_current_session_lease(
                connection,
                parent_id,
                owner_id,
                lease_now,
            )
            _require_writable_subagent_run(
                connection,
                project,
                parent_id,
                identifier,
            )
            connection.execute(
                """
                UPDATE subagent_runs
                SET status = ?, stop_reason = ?, context_json = ?,
                    token_usage_json = ?, tool_call_count = ?,
                    validation_execution_count = ?, result_json = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    normalized_status,
                    normalized_stop_reason,
                    context_json,
                    token_usage_json,
                    tool_count,
                    validation_count,
                    result_json,
                    occurred_text,
                    occurred_text,
                    identifier,
                ),
            )
            _append_subagent_event_locked(
                connection,
                identifier,
                event_type="result",
                state=normalized_status,
                reason=normalized_stop_reason,
                data_json=result_event_json,
                occurred_at=occurred_text,
                limit=self.subagent_event_limit,
            )
            _prune_subagent_runs_locked(
                connection,
                parent_id,
                retention=self.subagent_run_retention,
            )

    def get_subagent_run(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
    ) -> SubAgentRunRecord | None:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        with self._connect() as connection:
            _require_session(connection, project, parent_id)
            row = connection.execute(
                """
                SELECT * FROM subagent_runs
                WHERE id = ? AND parent_session_id = ? AND project_key = ?
                """,
                (identifier, parent_id, project.key),
            ).fetchone()
        return None if row is None else _subagent_run_from_row(row)

    def list_subagent_runs(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        *,
        limit: int = 100,
    ) -> list[SubAgentRunRecord]:
        parent_id = _validate_session_id(parent_session_id)
        normalized_limit = _validate_positive_limit(limit, field_name="limit")
        with self._connect() as connection:
            _require_session(connection, project, parent_id)
            rows = connection.execute(
                """
                SELECT * FROM subagent_runs
                WHERE parent_session_id = ? AND project_key = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (parent_id, project.key, normalized_limit),
            ).fetchall()
        return [_subagent_run_from_row(row) for row in rows]

    def list_subagent_events(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
    ) -> list[SubAgentEventRecord]:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        with self._connect() as connection:
            _require_session(connection, project, parent_id)
            _require_subagent_run(connection, project, parent_id, identifier)
            rows = connection.execute(
                """
                SELECT * FROM subagent_events
                WHERE run_id = ? ORDER BY sequence ASC
                """,
                (identifier,),
            ).fetchall()
        return [_subagent_event_from_row(row) for row in rows]

    def list_subagent_tool_audits(
        self,
        project: ProjectIdentity,
        parent_session_id: str,
        run_id: str,
    ) -> list[SubAgentToolAuditRecord]:
        parent_id = _validate_session_id(parent_session_id)
        identifier = _validate_subagent_run_id(run_id)
        with self._connect() as connection:
            _require_session(connection, project, parent_id)
            _require_subagent_run(connection, project, parent_id, identifier)
            rows = connection.execute(
                """
                SELECT * FROM subagent_tool_audits
                WHERE run_id = ? ORDER BY sequence ASC
                """,
                (identifier,),
            ).fetchall()
        return [_subagent_tool_audit_from_row(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            version = int(row[0])
            if version > SESSION_SCHEMA_VERSION:
                raise UnsupportedSessionSchemaError(
                    "Session database schema is newer than this MyCode version: "
                    f"database={version}, supported={SESSION_SCHEMA_VERSION}"
                )
            journal_mode = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).casefold()
            if journal_mode != SQLITE_JOURNAL_MODE:
                raise SessionStoreError(
                    "Session database could not enable WAL journal mode: "
                    f"{journal_mode}"
                )
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                connection.execute(
                    """
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY,
                        project_key TEXT NOT NULL,
                        workspace_root TEXT NOT NULL,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('active', 'closed', 'interrupted')
                        ),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX sessions_project_updated_idx
                    ON sessions(project_key, updated_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE session_messages (
                        session_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence >= 0),
                        role TEXT NOT NULL CHECK (
                            role IN ('user', 'assistant', 'tool')
                        ),
                        content TEXT NOT NULL,
                        tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        tool_call_id TEXT,
                        reasoning_content TEXT,
                        reasoning_state TEXT NOT NULL DEFAULT 'absent' CHECK (
                            reasoning_state IN (
                                'absent',
                                'present_empty',
                                'present_nonempty'
                            )
                        ),
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, sequence),
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                            ON DELETE CASCADE
                    )
                    """
                )
            if version <= 1:
                connection.execute(
                    """
                    CREATE TABLE session_leases (
                        session_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                if version == 1:
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = 'interrupted'
                        WHERE status = 'active'
                        """
                    )
            if version <= 2:
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS sessions_id_project_idx
                    ON sessions(id, project_key)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE subagent_runs (
                        id TEXT PRIMARY KEY,
                        parent_session_id TEXT NOT NULL,
                        project_key TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (
                            role IN ('explorer', 'tester', 'reviewer')
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'running', 'awaiting_confirmation',
                                'completed', 'failed', 'interrupted'
                            )
                        ),
                        stop_reason TEXT,
                        task_sha256 TEXT NOT NULL,
                        objective_chars INTEGER NOT NULL CHECK (objective_chars >= 0),
                        context_chars INTEGER NOT NULL CHECK (context_chars >= 0),
                        scope_path_count INTEGER NOT NULL CHECK (scope_path_count >= 0),
                        snapshot_json TEXT,
                        context_json TEXT,
                        token_usage_json TEXT,
                        tool_call_count INTEGER NOT NULL DEFAULT 0
                            CHECK (tool_call_count >= 0),
                        validation_execution_count INTEGER NOT NULL DEFAULT 0
                            CHECK (validation_execution_count >= 0),
                        omitted_event_count INTEGER NOT NULL DEFAULT 0
                            CHECK (omitted_event_count >= 0),
                        omitted_tool_audit_count INTEGER NOT NULL DEFAULT 0
                            CHECK (omitted_tool_audit_count >= 0),
                        result_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        FOREIGN KEY (parent_session_id, project_key)
                            REFERENCES sessions(id, project_key) ON DELETE CASCADE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX subagent_runs_parent_created_idx
                    ON subagent_runs(parent_session_id, created_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE subagent_events (
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence >= 0),
                        event_type TEXT NOT NULL CHECK (
                            event_type IN ('state', 'snapshot', 'result')
                        ),
                        state TEXT CHECK (
                            state IS NULL OR state IN (
                                'running', 'awaiting_confirmation',
                                'completed', 'failed', 'interrupted'
                            )
                        ),
                        reason TEXT,
                        data_json TEXT NOT NULL DEFAULT '{}',
                        occurred_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, sequence),
                        FOREIGN KEY (run_id) REFERENCES subagent_runs(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE subagent_tool_audits (
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence >= 0),
                        tool_name TEXT NOT NULL,
                        arguments_sha256 TEXT NOT NULL,
                        argument_summary_json TEXT NOT NULL,
                        ok INTEGER NOT NULL CHECK (ok IN (0, 1)),
                        exit_code INTEGER,
                        duration_ms INTEGER CHECK (
                            duration_ms IS NULL OR duration_ms >= 0
                        ),
                        output_chars INTEGER NOT NULL CHECK (output_chars >= 0),
                        truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
                        reason TEXT,
                        occurred_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, sequence),
                        FOREIGN KEY (run_id) REFERENCES subagent_runs(id)
                            ON DELETE CASCADE
                    )
                    """
                )
            if version <= 3:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_compact_state (
                        session_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                            ON DELETE CASCADE
                    )
                    """
                )
            if version <= 4:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_deletion_tasks (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        project_key TEXT NOT NULL,
                        stage TEXT NOT NULL CHECK (
                            stage IN (
                                'pending',
                                'database_deleted',
                                'initial_checkpoint_complete',
                                'vacuum_complete',
                                'final_checkpoint_complete'
                            )
                        ),
                        artifact_present INTEGER CHECK (
                            artifact_present IS NULL
                            OR artifact_present IN (0, 1)
                        ),
                        retry_count INTEGER NOT NULL DEFAULT 0
                            CHECK (retry_count >= 0),
                        last_error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (session_id, project_key)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        session_deletion_tasks_project_created_idx
                    ON session_deletion_tasks(project_key, created_at ASC)
                    """
                )
            if version <= 5:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS database_maintenance_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        post_delete_scrub_required INTEGER NOT NULL DEFAULT 0
                            CHECK (post_delete_scrub_required IN (0, 1)),
                        retry_count INTEGER NOT NULL DEFAULT 0
                            CHECK (retry_count >= 0),
                        last_error_code TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO database_maintenance_state (
                        id,
                        post_delete_scrub_required,
                        retry_count,
                        last_error_code,
                        updated_at
                    ) VALUES (1, 0, 0, NULL, ?)
                    """,
                    ("1970-01-01T00:00:00+00:00",),
                )
            if version <= 6:
                session_message_columns = {
                    str(column[1])
                    for column in connection.execute(
                        "PRAGMA table_info(session_messages)"
                    ).fetchall()
                }
                if (
                    session_message_columns
                    and "reasoning_content" not in session_message_columns
                ):
                    connection.execute(
                        "ALTER TABLE session_messages "
                        "ADD COLUMN reasoning_content TEXT"
                    )
            if version <= 7:
                session_message_columns = {
                    str(column[1])
                    for column in connection.execute(
                        "PRAGMA table_info(session_messages)"
                    ).fetchall()
                }
                if (
                    session_message_columns
                    and "reasoning_state" not in session_message_columns
                ):
                    connection.execute(
                        "ALTER TABLE session_messages "
                        "ADD COLUMN reasoning_state TEXT NOT NULL "
                        "DEFAULT 'absent' CHECK (reasoning_state IN ("
                        "'absent', 'present_empty', 'present_nonempty'))"
                    )
                if session_message_columns:
                    connection.execute(
                        "UPDATE session_messages SET reasoning_state = CASE "
                        "WHEN reasoning_content = '' THEN 'present_empty' "
                        "WHEN reasoning_content IS NOT NULL "
                        "THEN 'present_nonempty' ELSE 'absent' END"
                    )
            connection.execute(f"PRAGMA user_version = {SESSION_SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=self.busy_timeout_seconds,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA secure_delete = ON")
            secure_delete = int(
                connection.execute("PRAGMA secure_delete").fetchone()[0]
            )
            if secure_delete != 1:
                raise SessionStoreError(
                    "Session database could not enable secure_delete."
                )
            connection.execute(
                f"PRAGMA busy_timeout = {round(self.busy_timeout_seconds * 1000)}"
            )
            with connection:
                yield connection
        except sqlite3.IntegrityError:
            raise
        except sqlite3.OperationalError as error:
            raise SessionStoreError(
                f"Session database operation failed: {error}"
            ) from error
        except sqlite3.DatabaseError as error:
            raise SessionDatabaseCorruptionError(
                "Session database is unreadable or corrupt; "
                "MyCode did not repair or replace it."
            ) from error
        finally:
            if connection is not None:
                connection.close()


@dataclass(frozen=True)
class _PreparedMessage:
    role: str
    content: str
    tool_calls_json: str
    tool_call_id: str | None
    reasoning_content: str | None = field(repr=False)
    reasoning_state: ReasoningState


def _prepare_message(message: Message) -> _PreparedMessage:
    if message.role == "system":
        raise SessionDataError(
            "System messages are regenerated at resume time and must not be persisted."
        )
    if message.role not in PERSISTED_MESSAGE_ROLES:
        raise SessionDataError(f"Unsupported persisted message role: {message.role}")
    if not isinstance(message.content, str):
        raise SessionDataError("Message content must be a string.")
    if message.tool_calls and message.role != "assistant":
        raise SessionDataError("Only assistant messages can contain tool calls.")
    if message.role == "tool" and not message.tool_call_id:
        raise SessionDataError("Tool result messages must contain tool_call_id.")
    if message.tool_call_id is not None and message.role != "tool":
        raise SessionDataError("Only tool messages can contain tool_call_id.")
    if message.reasoning_state != "absent" and (
        message.role != "assistant" or not message.tool_calls
    ):
        raise SessionDataError(
            "Reasoning content requires an assistant tool-call message."
        )

    try:
        tool_calls_json = json.dumps(
            [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SessionDataError("Tool call arguments must be valid JSON data.") from error

    return _PreparedMessage(
        role=message.role,
        content=message.content,
        tool_calls_json=tool_calls_json,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
        reasoning_state=message.reasoning_state,
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    role = str(row["role"])
    if role not in PERSISTED_MESSAGE_ROLES:
        raise SessionDataError(f"Unsupported stored message role: {role}")

    try:
        raw_tool_calls = json.loads(str(row["tool_calls_json"]))
    except json.JSONDecodeError as error:
        raise SessionDataError("Stored tool_calls_json is invalid JSON.") from error
    if not isinstance(raw_tool_calls, list):
        raise SessionDataError("Stored tool_calls_json must contain a list.")

    tool_calls: list[AgentToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            raise SessionDataError("Stored tool call must be an object.")
        identifier = raw_tool_call.get("id")
        name = raw_tool_call.get("name")
        arguments = raw_tool_call.get("arguments")
        if not isinstance(identifier, str) or not isinstance(name, str):
            raise SessionDataError("Stored tool call id and name must be strings.")
        if not isinstance(arguments, dict):
            raise SessionDataError("Stored tool call arguments must be an object.")
        tool_calls.append(
            AgentToolCall(
                id=identifier,
                name=name,
                arguments=arguments,
            )
        )

    tool_call_id = row["tool_call_id"]
    reasoning_content = row["reasoning_content"]
    raw_reasoning_state = str(row["reasoning_state"])
    if raw_reasoning_state not in {
        "absent",
        "present_empty",
        "present_nonempty",
    }:
        raise SessionDataError("Stored reasoning_state is invalid.")
    if (
        (
            raw_reasoning_state in {"absent", "present_empty"}
            and reasoning_content is not None
        )
        or (
            raw_reasoning_state == "present_nonempty"
            and (not isinstance(reasoning_content, str) or reasoning_content == "")
        )
    ):
        raise SessionDataError(
            "Stored reasoning state and content are inconsistent."
        )
    reasoning_state = cast(ReasoningState, raw_reasoning_state)
    message = Message(
        role=role,  # type: ignore[arg-type]
        content=str(row["content"]),
        tool_calls=tuple(tool_calls),
        tool_call_id=None if tool_call_id is None else str(tool_call_id),
        reasoning_content=(
            None if reasoning_content is None else str(reasoning_content)
        ),
        reasoning_state=reasoning_state,
    )
    _prepare_message(message)
    return message


def _database_maintenance_state_from_row(
    row: sqlite3.Row,
) -> DatabaseMaintenanceState:
    error_code = row["last_error_code"]
    return DatabaseMaintenanceState(
        post_delete_scrub_required=bool(row["post_delete_scrub_required"]),
        retry_count=int(row["retry_count"]),
        last_error_code=None if error_code is None else str(error_code),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _subagent_run_from_row(row: sqlite3.Row) -> SubAgentRunRecord:
    status = _validate_subagent_status(str(row["status"]))
    completed_at = row["completed_at"]
    stop_reason = row["stop_reason"]
    return SubAgentRunRecord(
        id=str(row["id"]),
        parent_session_id=str(row["parent_session_id"]),
        project_key=str(row["project_key"]),
        role=_validate_subagent_role(str(row["role"])),
        status=status,
        stop_reason=None if stop_reason is None else str(stop_reason),
        task_sha256=_validate_sha256(
            str(row["task_sha256"]),
            field_name="stored task_sha256",
        ),
        objective_chars=int(row["objective_chars"]),
        context_chars=int(row["context_chars"]),
        scope_path_count=int(row["scope_path_count"]),
        snapshot=_optional_json_object(row["snapshot_json"], field_name="snapshot"),
        context=_optional_json_object(row["context_json"], field_name="context"),
        token_usage=_optional_json_object(
            row["token_usage_json"],
            field_name="token_usage",
        ),
        tool_call_count=int(row["tool_call_count"]),
        validation_execution_count=int(row["validation_execution_count"]),
        omitted_event_count=int(row["omitted_event_count"]),
        omitted_tool_audit_count=int(row["omitted_tool_audit_count"]),
        result=_optional_json_object(row["result_json"], field_name="result"),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
        completed_at=(
            None
            if completed_at is None
            else _datetime_from_text(str(completed_at))
        ),
    )


def _subagent_event_from_row(row: sqlite3.Row) -> SubAgentEventRecord:
    state = row["state"]
    reason = row["reason"]
    return SubAgentEventRecord(
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        event_type=str(row["event_type"]),
        state=None if state is None else str(state),
        reason=None if reason is None else str(reason),
        data=_json_object_from_text(str(row["data_json"]), field_name="event data"),
        occurred_at=_datetime_from_text(str(row["occurred_at"])),
    )


def _subagent_tool_audit_from_row(row: sqlite3.Row) -> SubAgentToolAuditRecord:
    reason = row["reason"]
    exit_code = row["exit_code"]
    duration_ms = row["duration_ms"]
    return SubAgentToolAuditRecord(
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        tool_name=str(row["tool_name"]),
        arguments_sha256=_validate_sha256(
            str(row["arguments_sha256"]),
            field_name="stored arguments_sha256",
        ),
        argument_summary=_json_object_from_text(
            str(row["argument_summary_json"]),
            field_name="argument summary",
        ),
        ok=bool(row["ok"]),
        exit_code=None if exit_code is None else int(exit_code),
        duration_ms=None if duration_ms is None else int(duration_ms),
        output_chars=int(row["output_chars"]),
        truncated=bool(row["truncated"]),
        reason=None if reason is None else str(reason),
        occurred_at=_datetime_from_text(str(row["occurred_at"])),
    )


def _require_subagent_run(
    connection: sqlite3.Connection,
    project: ProjectIdentity,
    parent_session_id: str,
    run_id: str,
) -> None:
    row = connection.execute(
        """
        SELECT 1 FROM subagent_runs
        WHERE id = ? AND parent_session_id = ? AND project_key = ?
        """,
        (run_id, parent_session_id, project.key),
    ).fetchone()
    if row is None:
        raise SessionNotFoundError(
            f"SubAgent run not found in current project session: {run_id}"
        )


def _require_writable_subagent_run(
    connection: sqlite3.Connection,
    project: ProjectIdentity,
    parent_session_id: str,
    run_id: str,
) -> None:
    row = connection.execute(
        """
        SELECT status FROM subagent_runs
        WHERE id = ? AND parent_session_id = ? AND project_key = ?
        """,
        (run_id, parent_session_id, project.key),
    ).fetchone()
    if row is None:
        raise SessionNotFoundError(
            f"SubAgent run not found in current project session: {run_id}"
        )
    status = str(row["status"])
    if status in SUBAGENT_TERMINAL_STATUSES:
        raise SessionStoreError(
            f"SubAgent run is already terminal and cannot be changed: {run_id}"
        )


def _append_subagent_event_locked(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    event_type: str,
    state: str | None,
    reason: str | None,
    data_json: str,
    occurred_at: str,
    limit: int,
) -> None:
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM subagent_events WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    count = int(row["count"])
    if count >= limit:
        connection.execute(
            """
            UPDATE subagent_runs
            SET omitted_event_count = omitted_event_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (occurred_at, run_id),
        )
        return
    connection.execute(
        """
        INSERT INTO subagent_events (
            run_id, sequence, event_type, state, reason, data_json, occurred_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, count, event_type, state, reason, data_json, occurred_at),
    )


def _interrupt_unfinished_subagent_runs_locked(
    connection: sqlite3.Connection,
    parent_session_id: str,
    occurred_at: datetime,
    *,
    reason: str,
    event_limit: int,
    run_retention: int,
) -> int:
    occurred_text = _datetime_to_text(occurred_at)
    rows = connection.execute(
        """
        SELECT id FROM subagent_runs
        WHERE parent_session_id = ?
          AND status IN ('running', 'awaiting_confirmation')
        ORDER BY created_at ASC, id ASC
        """,
        (parent_session_id,),
    ).fetchall()
    for row in rows:
        run_id = str(row["id"])
        _append_subagent_event_locked(
            connection,
            run_id,
            event_type="state",
            state="interrupted",
            reason=reason,
            data_json="{}",
            occurred_at=occurred_text,
            limit=event_limit,
        )
        connection.execute(
            """
            UPDATE subagent_runs
            SET status = 'interrupted', stop_reason = 'interrupted',
                completed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (occurred_text, occurred_text, run_id),
        )
    _prune_subagent_runs_locked(
        connection,
        parent_session_id,
        retention=run_retention,
    )
    return len(rows)


def _prune_subagent_runs_locked(
    connection: sqlite3.Connection,
    parent_session_id: str,
    *,
    retention: int,
) -> None:
    rows = connection.execute(
        """
        SELECT id, status FROM subagent_runs
        WHERE parent_session_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (parent_session_id,),
    ).fetchall()
    removable = [
        str(row["id"])
        for row in rows[retention:]
        if str(row["status"]) in SUBAGENT_TERMINAL_STATUSES
    ]
    if not removable:
        return
    placeholders = ", ".join("?" for _ in removable)
    connection.execute(
        f"DELETE FROM subagent_runs WHERE id IN ({placeholders})",
        removable,
    )


def _bounded_json_object(
    value: dict[str, object],
    *,
    field_name: str,
    max_chars: int,
) -> str:
    if not isinstance(value, dict):
        raise SessionDataError(f"{field_name} must be a JSON object.")
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SessionDataError(f"{field_name} must contain valid JSON data.") from error
    if len(serialized) > max_chars:
        raise SessionDataError(
            f"{field_name} exceeds its storage limit: {len(serialized)}/{max_chars}."
        )
    return serialized


def _optional_json_object(
    value: object,
    *,
    field_name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    return _json_object_from_text(str(value), field_name=field_name)


def _json_object_from_text(value: str, *, field_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SessionDataError(f"Stored {field_name} is invalid JSON.") from error
    if not isinstance(parsed, dict):
        raise SessionDataError(f"Stored {field_name} must be a JSON object.")
    return parsed


def _require_session(
    connection: sqlite3.Connection,
    project: ProjectIdentity,
    session_id: str,
) -> None:
    row = connection.execute(
        "SELECT 1 FROM sessions WHERE id = ? AND project_key = ?",
        (session_id, project.key),
    ).fetchone()
    if row is None:
        raise SessionNotFoundError(
            f"Session not found in current project: {session_id}"
        )
    deletion = connection.execute(
        """
        SELECT 1
        FROM session_deletion_tasks
        WHERE session_id = ? AND project_key = ?
        """,
        (session_id, project.key),
    ).fetchone()
    if deletion is not None:
        raise SessionDeletingError(
            f"Session deletion is already in progress: {session_id}"
        )


def _require_no_session_deletion(
    connection: sqlite3.Connection,
    project: ProjectIdentity,
    session_id: str,
) -> None:
    row = connection.execute(
        """
        SELECT 1
        FROM session_deletion_tasks
        WHERE session_id = ? AND project_key = ?
        """,
        (session_id, project.key),
    ).fetchone()
    if row is not None:
        raise SessionDeletingError(
            f"Session deletion is already in progress: {session_id}"
        )


def _require_session_deletion(
    connection: sqlite3.Connection,
    deletion_id: str,
) -> SessionDeletionRecord:
    row = connection.execute(
        """
        SELECT *
        FROM session_deletion_tasks
        WHERE id = ?
        """,
        (deletion_id,),
    ).fetchone()
    if row is None:
        raise SessionNotFoundError(f"Session deletion task not found: {deletion_id}")
    return _session_deletion_from_row(row)


def _insert_session_lease(
    connection: sqlite3.Connection,
    session_id: str,
    owner_id: str,
    now: datetime,
    duration_seconds: float,
) -> None:
    now_text = _datetime_to_text(now)
    connection.execute(
        """
        INSERT INTO session_leases (
            session_id,
            owner_id,
            acquired_at,
            heartbeat_at,
            expires_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            owner_id,
            now_text,
            now_text,
            _datetime_to_text(now + timedelta(seconds=duration_seconds)),
        ),
    )


def _require_current_session_lease(
    connection: sqlite3.Connection,
    session_id: str,
    owner_id: str,
    now: datetime,
    *,
    allow_expired: bool = False,
) -> None:
    row = connection.execute(
        """
        SELECT owner_id, expires_at
        FROM session_leases
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None or str(row["owner_id"]) != owner_id:
        raise SessionLeaseLostError(
            f"Session lease is no longer owned by this agent: {session_id}"
        )
    expires_at = _datetime_from_text(str(row["expires_at"]))
    if not allow_expired and expires_at <= now:
        raise SessionLeaseLostError(f"Session lease has expired: {session_id}")


def _require_session_not_in_use(
    connection: sqlite3.Connection,
    session_id: str,
    now: datetime,
    *,
    subagent_event_limit: int,
    subagent_run_retention: int,
) -> None:
    row = connection.execute(
        "SELECT expires_at FROM session_leases WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return
    expires_at = _datetime_from_text(str(row["expires_at"]))
    if expires_at > now:
        raise SessionInUseError(
            f"Session is already in use by another agent: {session_id}"
        )
    connection.execute(
        "DELETE FROM session_leases WHERE session_id = ?",
        (session_id,),
    )
    connection.execute(
        "UPDATE sessions SET status = 'interrupted' WHERE id = ?",
        (session_id,),
    )
    _interrupt_unfinished_subagent_runs_locked(
        connection,
        session_id,
        now,
        reason="parent_lease_expired",
        event_limit=subagent_event_limit,
        run_retention=subagent_run_retention,
    )


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    status = str(row["status"])
    if status not in SESSION_STATUSES:
        raise SessionDataError(f"Unsupported stored session status: {status}")
    return SessionRecord(
        id=str(row["id"]),
        project_key=str(row["project_key"]),
        workspace_root=Path(str(row["workspace_root"])),
        title=str(row["title"]),
        status=status,  # type: ignore[arg-type]
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _session_deletion_from_row(row: sqlite3.Row) -> SessionDeletionRecord:
    stage = str(row["stage"])
    if stage not in SESSION_DELETION_STAGES:
        raise SessionDataError(f"Unsupported session deletion stage: {stage}")
    artifact_present = row["artifact_present"]
    retry_count = int(row["retry_count"])
    if retry_count < 0:
        raise SessionDataError("Stored session deletion retry count is invalid.")
    last_error_code = row["last_error_code"]
    try:
        deletion_id = _validate_session_id(str(row["id"]))
        session_id = _validate_session_id(str(row["session_id"]))
        project_key = _validate_sha256(
            str(row["project_key"]),
            field_name="stored project_key",
        )
        validated_error_code = (
            None
            if last_error_code is None
            else _validate_deletion_error_code(str(last_error_code))
        )
    except ValueError as error:
        raise SessionDataError(
            "Stored session deletion identity or error code is invalid."
        ) from error
    return SessionDeletionRecord(
        id=deletion_id,
        session_id=session_id,
        project_key=project_key,
        stage=stage,  # type: ignore[arg-type]
        artifact_present=(
            None if artifact_present is None else bool(artifact_present)
        ),
        retry_count=retry_count,
        last_error_code=validated_error_code,
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _validate_session_id(session_id: str) -> str:
    identifier = session_id.strip()
    if identifier == "":
        raise ValueError("session_id must not be empty.")
    return identifier


def _validate_subagent_run_id(run_id: str) -> str:
    identifier = run_id.strip()
    if identifier == "":
        raise ValueError("SubAgent run_id must not be empty.")
    if len(identifier) > 100:
        raise ValueError("SubAgent run_id must not exceed 100 characters.")
    return identifier


def _validate_subagent_role(role: str) -> str:
    normalized = role.strip()
    if normalized not in SUBAGENT_ROLES:
        raise ValueError(f"Unsupported SubAgent role: {role}")
    return normalized


def _validate_subagent_status(status: str) -> SubAgentRunStatus:
    normalized = status.strip()
    if normalized not in SUBAGENT_RUN_STATUSES:
        raise ValueError(f"Unsupported SubAgent status: {status}")
    return normalized  # type: ignore[return-value]


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")
    return normalized


def _validate_short_text(
    value: str,
    *,
    field_name: str,
    max_chars: int,
) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} must not exceed {max_chars} characters.")
    return normalized


def _validate_positive_limit(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer at least 1.")
    return value


def _validate_non_negative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_optional_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or None.")
    return value


def _validate_optional_non_negative_int(
    value: int | None,
    *,
    field_name: str,
) -> int | None:
    parsed = _validate_optional_int(value, field_name=field_name)
    if parsed is not None and parsed < 0:
        raise ValueError(f"{field_name} must not be negative.")
    return parsed


def _validate_lease_owner_id(owner_id: str) -> str:
    normalized = owner_id.strip()
    if normalized == "":
        raise ValueError("lease owner_id must not be empty.")
    return normalized


def _validate_lease_duration(duration_seconds: float) -> float:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("lease_duration_seconds must be a finite number above 0.")
    return duration_seconds


def _validate_title(title: str) -> str:
    normalized = " ".join(title.split())
    if normalized == "":
        raise ValueError("Session title must not be empty.")
    if len(normalized) > MAX_SESSION_TITLE_CHARS:
        raise ValueError(
            f"Session title must not exceed {MAX_SESSION_TITLE_CHARS} characters."
        )
    return normalized


def _validate_deletion_error_code(error_code: str) -> str:
    normalized = error_code.strip()
    if (
        normalized == ""
        or len(normalized) > MAX_SESSION_DELETION_ERROR_CODE_CHARS
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in normalized
        )
    ):
        raise ValueError(
            "Session deletion error code must be a short lowercase code."
        )
    return normalized


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Session timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _datetime_from_text(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SessionDataError(f"Invalid stored session timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise SessionDataError(f"Stored session timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)
