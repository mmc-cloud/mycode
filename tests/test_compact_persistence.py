from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from mycode.context_budget import ContextBudget, TokenEstimator
from mycode.context_compact import (
    COMPACT_SUMMARY_MARKER,
    DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES,
    CompactBoundary,
    CompactPolicy,
    CompactState,
    CompactSummary,
    ConversationCompactor,
)
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_runtime import SessionStartRequest, start_project_session
from mycode.session_store import (
    SESSION_SCHEMA_VERSION,
    SessionDataError,
    SessionLeaseLostError,
    SessionStore,
)


def compact_boundary() -> CompactBoundary:
    return CompactBoundary(
        boundary_id="boundary-1",
        covered_message_count=2,
        covered_turn_count=1,
        summary=CompactSummary(
            objective="Continue implementation",
            progress=("Located the runtime.",),
            decisions=("Keep canonical history.",),
            constraints=("Do not split tool groups.",),
            open_items=("Run tests.",),
            references=("mycode/context_budget.py",),
        ),
        source_estimated_tokens=800,
        summary_prompt_tokens=200,
        summary_completion_tokens=50,
        created_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )


def test_session_store_persists_latest_boundary_and_failure_state(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")
    store.append_messages(
        project,
        session.id,
        [
            Message(role="user", content="old request"),
            Message(role="assistant", content="old reply"),
            Message(role="user", content="current request"),
        ],
    )
    state = CompactState(
        boundary=compact_boundary(),
        consecutive_failure_count=1,
        retry_after_message_count=10,
        last_failure_reason="temporary provider failure",
    )

    store.save_compact_state(project, session.id, state)

    assert store.load_compact_state(project, session.id) == state


def test_session_store_rejects_boundary_past_persisted_history(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")

    with pytest.raises(SessionDataError, match="exceeds"):
        store.save_compact_state(
            project,
            session.id,
            CompactState(boundary=compact_boundary()),
        )


def test_compact_state_write_requires_current_session_lease(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        session_id="session",
        lease_owner_id="owner-a",
    )
    store.append_messages(
        project,
        session.id,
        [
            Message(role="user", content="old request"),
            Message(role="assistant", content="old reply"),
        ],
        lease_owner_id="owner-a",
    )

    with pytest.raises(SessionLeaseLostError):
        store.save_compact_state(
            project,
            session.id,
            CompactState(boundary=compact_boundary()),
            lease_owner_id="owner-b",
        )

    store.save_compact_state(
        project,
        session.id,
        CompactState(boundary=compact_boundary()),
        lease_owner_id="owner-a",
    )
    assert store.load_compact_state(project, session.id).boundary is not None


def test_resume_rebuilds_compact_summary_plus_recent_tail(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda _message: None,
    )
    assert active is not None
    for message in (
        Message(role="user", content="old request"),
        Message(role="assistant", content="old reply"),
        Message(role="user", content="current request"),
        Message(role="assistant", content="current reply"),
    ):
        active.persist_message(message)
    active.persist_compact_state(CompactState(boundary=compact_boundary()))

    restored_history = active.load_history()
    restored_state = active.load_compact_state()
    compactor = ConversationCompactor(
        llm_client=NeverSummaryClient(),
        policy=CompactPolicy(trigger_ratio=1.0),
        state=restored_state,
    )
    prepared = compactor.prepare(
        restored_history,
        ContextBudget(
            context_window_tokens=10000,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        token_estimator=TokenEstimator(),
    )

    visible = prepared.conversation.get_messages()
    assert COMPACT_SUMMARY_MARKER in visible[0].content
    assert visible[1:] == restored_history.get_messages()[2:]
    assert restored_history.get_messages()[0].content == "old request"
    active.close()


def test_invalid_compact_json_is_reset_under_lease_with_persisted_cooldown(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda _message: None,
    )
    assert active is not None
    active.persist_message(Message(role="user", content="canonical history"))
    active.persist_compact_state(CompactState())
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE session_compact_state
            SET state_json = '{invalid-json'
            WHERE session_id = ?
            """,
            (active.record.id,),
        )

    recovered = active.load_compact_state()

    assert active.compact_state_recovered is True
    assert recovered.boundary is None
    assert recovered.consecutive_failure_count == 1
    assert recovered.retry_after_message_count == (
        1 + DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES
    )
    assert recovered.last_failure_reason == "stored_compact_state_invalid"
    assert store.load_compact_state(project, active.record.id) == recovered
    assert active.load_history().get_messages() == [
        Message(role="user", content="canonical history")
    ]
    active.close()


def test_schema_v3_migrates_to_compact_state_table(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                project_key TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 3")

    SessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table_count = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'session_compact_state'
            """
        ).fetchone()[0]
    assert version == SESSION_SCHEMA_VERSION
    assert table_count == 1


class NeverSummaryClient:
    def complete(self, conversation: Conversation) -> Message:
        raise AssertionError("Summary model should not be called.")
