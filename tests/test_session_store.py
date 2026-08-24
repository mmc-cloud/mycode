from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier
import time

import pytest

from mycode.agent import AgentToolCall
from mycode.context_compact import CompactState
from mycode.messages import Message
from mycode.session_deletion import SessionDeletionManager
from mycode.session_store import (
    DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS,
    ProjectIdentity,
    SESSION_SCHEMA_VERSION,
    SessionDataError,
    SessionDatabaseCorruptionError,
    SessionInUseError,
    SessionLeaseLostError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
    UnsupportedSessionSchemaError,
)


def test_project_identity_is_stable_for_same_workspace(tmp_path: Path) -> None:
    first = ProjectIdentity.from_workspace(tmp_path)
    second = ProjectIdentity.from_workspace(tmp_path / ".")

    assert first == second
    assert len(first.key) == 64
    assert first.workspace_root == tmp_path.resolve()


def test_session_store_creates_parent_and_versioned_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "state.sqlite3"

    store = SessionStore(database_path)

    assert database_path.exists()
    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert version == SESSION_SCHEMA_VERSION
    assert journal_mode == "wal"
    assert "session_leases" in tables
    assert "subagent_runs" in tables
    assert "subagent_events" in tables
    assert "subagent_tool_audits" in tables
    assert "session_compact_state" in tables
    with store._connect() as connection:
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        secure_delete = connection.execute("PRAGMA secure_delete").fetchone()[0]
    assert busy_timeout == round(DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)
    assert secure_delete == 1
    assert "session_deletion_tasks" in tables
    assert "database_maintenance_state" in tables
    with sqlite3.connect(database_path) as connection:
        message_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(session_messages)"
            ).fetchall()
        }
    assert "reasoning_content" in message_columns
    assert "reasoning_state" in message_columns


def test_corrupt_sqlite_database_is_not_replaced_or_silently_repaired(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    original = b"not-a-sqlite-database\x00private-bytes"
    database_path.write_bytes(original)

    with pytest.raises(
        SessionDatabaseCorruptionError,
        match="did not repair or replace",
    ):
        SessionStore(database_path)

    assert database_path.read_bytes() == original


def test_schema_v4_migrates_deletion_tasks_without_losing_session_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    original = SessionStore(database_path)
    original.create_session(project, session_id="v4-session")
    original.save_compact_state(
        project,
        "v4-session",
        CompactState(
            consecutive_failure_count=1,
            retry_after_message_count=8,
            last_failure_reason="legacy_compact_failure",
        ),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE session_deletion_tasks")
        connection.execute("PRAGMA user_version = 4")

    migrated = SessionStore(database_path)

    assert migrated.get_session(project, "v4-session") is not None
    assert (
        migrated.load_compact_state(project, "v4-session").last_failure_reason
        == "legacy_compact_failure"
    )
    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        deletion_table_count = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'session_deletion_tasks'
            """
        ).fetchone()[0]
    assert version == SESSION_SCHEMA_VERSION
    assert deletion_table_count == 1


def test_schema_v5_adds_anonymous_database_maintenance_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    original = SessionStore(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE database_maintenance_state")
        connection.execute("PRAGMA user_version = 5")

    migrated = SessionStore(database_path)
    state = migrated.load_database_maintenance_state()

    assert state.post_delete_scrub_required is False
    assert state.retry_count == 0
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == SESSION_SCHEMA_VERSION
        )


def test_schema_v6_adds_reasoning_content_without_losing_messages(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    original = SessionStore(database_path)
    session = original.create_session(project, session_id="v6-session")
    original.append_message(
        project,
        session.id,
        Message(role="user", content="legacy message"),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE session_messages DROP COLUMN reasoning_content"
        )
        connection.execute("PRAGMA user_version = 6")

    migrated = SessionStore(database_path)

    assert migrated.load_conversation(project, session.id).get_messages() == [
        Message(role="user", content="legacy message")
    ]
    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(session_messages)"
            ).fetchall()
        }
    assert version == SESSION_SCHEMA_VERSION
    assert "reasoning_content" in columns
    assert "reasoning_state" in columns


def test_schema_v7_adds_reasoning_state_and_preserves_nonempty_reasoning(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    original = SessionStore(database_path)
    session = original.create_session(project, session_id="v7-session")
    tool_call = AgentToolCall(
        id="call-v7",
        name="read_file",
        arguments={"path": "README.md"},
    )
    original.append_message(
        project,
        session.id,
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE session_messages DROP COLUMN reasoning_state"
        )
        connection.execute("PRAGMA user_version = 7")

    migrated = SessionStore(database_path)
    restored = migrated.load_conversation(project, session.id).get_messages()[0]

    assert restored.reasoning_state == "present_nonempty"
    assert restored.reasoning_content == "private synthetic reasoning"


def test_schema_v7_reasoning_state_migration_is_idempotent_for_partial_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    original = SessionStore(database_path)
    session = original.create_session(project, session_id="partial-v7-session")
    tool_call = AgentToolCall(
        id="call-partial-v7",
        name="read_file",
        arguments={"path": "README.md"},
    )
    original.append_message(
        project,
        session.id,
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE session_messages SET reasoning_state = 'absent' "
            "WHERE session_id = ?",
            (session.id,),
        )
        connection.execute("PRAGMA user_version = 7")

    migrated = SessionStore(database_path)
    restored = migrated.load_conversation(project, session.id).get_messages()[0]

    assert restored.reasoning_state == "present_nonempty"
    assert restored.reasoning_content == "private synthetic reasoning"


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_session_store_rejects_invalid_busy_timeout(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        SessionStore(tmp_path / "state.sqlite3", busy_timeout_seconds=value)


def test_session_store_rejects_newer_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"PRAGMA user_version = {SESSION_SCHEMA_VERSION + 1}")

    with pytest.raises(UnsupportedSessionSchemaError, match="newer"):
        SessionStore(database_path)


def test_schema_v1_migration_marks_unowned_active_sessions_interrupted(
    tmp_path: Path,
) -> None:
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
        connection.execute(
            """
            INSERT INTO sessions VALUES (
                'legacy', 'project', 'C:/project', 'Legacy', 'active',
                '2026-07-12T00:00:00+00:00',
                '2026-07-12T00:00:00+00:00'
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")

    SessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        status = connection.execute(
            "SELECT status FROM sessions WHERE id = 'legacy'"
        ).fetchone()[0]
        lease_table = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'session_leases'
            """
        ).fetchone()[0]
    assert version == SESSION_SCHEMA_VERSION
    assert status == "interrupted"
    assert lease_table == 1


def test_session_lease_blocks_second_owner_and_allows_expired_takeover(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    current = [datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)]
    first_store = SessionStore(database_path, now=lambda: current[0])
    second_store = SessionStore(database_path, now=lambda: current[0])
    session = first_store.create_session(
        project,
        session_id="session",
        lease_owner_id="owner-a",
        lease_duration_seconds=30,
    )

    with pytest.raises(SessionInUseError, match="another agent"):
        second_store.acquire_session_lease(
            project,
            session.id,
            "owner-b",
            lease_duration_seconds=30,
        )
    with pytest.raises(SessionInUseError, match="another agent"):
        second_store.append_message(
            project,
            session.id,
            Message(role="user", content="unowned writer"),
        )
    with pytest.raises(SessionInUseError, match="another agent"):
        second_store.rename_session(project, session.id, "Unsafe rename")
    with pytest.raises(SessionInUseError, match="another agent"):
        second_store.set_status(project, session.id, "closed")
    with pytest.raises(SessionInUseError, match="another agent"):
        SessionDeletionManager(second_store).request_and_process(
            project,
            session.id,
        )

    current[0] += timedelta(seconds=31)
    resumed = second_store.acquire_session_lease(
        project,
        session.id,
        "owner-b",
        lease_duration_seconds=30,
    )
    assert resumed.status == "active"
    with pytest.raises(SessionLeaseLostError, match="no longer owned"):
        first_store.append_message(
            project,
            session.id,
            Message(role="user", content="stale writer"),
            lease_owner_id="owner-a",
        )

    second_store.append_message(
        project,
        session.id,
        Message(role="user", content="current writer"),
        lease_owner_id="owner-b",
    )
    released = second_store.release_session_lease(
        project,
        session.id,
        "owner-b",
        "closed",
    )
    assert released.status == "closed"


def test_renew_samples_time_after_waiting_for_sqlite_write_lock(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 7, 27, tzinfo=timezone.utc)
    current = [base]
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(database_path, now=lambda: current[0])
    session = store.create_session(
        project,
        session_id="lock-wait-renew",
        lease_owner_id="owner",
        lease_duration_seconds=1,
    )
    current[0] = base + timedelta(seconds=0.2)
    blocker = sqlite3.connect(database_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.renew_session_lease,
            project,
            session.id,
            "owner",
            lease_duration_seconds=1,
        )
        time.sleep(0.05)
        current[0] = base + timedelta(seconds=0.9)
        blocker.rollback()
        blocker.close()
        future.result(timeout=2)

    with sqlite3.connect(database_path) as connection:
        expires_at = datetime.fromisoformat(
            connection.execute(
                "SELECT expires_at FROM session_leases WHERE session_id = ?",
                (session.id,),
            ).fetchone()[0]
        )
    assert expires_at == base + timedelta(seconds=1.9)

    current[0] = base + timedelta(seconds=1.3)
    store.renew_session_lease(
        project,
        session.id,
        "owner",
        lease_duration_seconds=1,
    )


def test_expired_session_cleanup_is_project_scoped(tmp_path: Path) -> None:
    project_a_root = tmp_path / "project-a"
    project_b_root = tmp_path / "project-b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    current = [datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)]
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    session_a = store.create_session(
        project_a,
        session_id="a",
        lease_owner_id="owner-a",
        lease_duration_seconds=10,
    )
    session_b = store.create_session(
        project_b,
        session_id="b",
        lease_owner_id="owner-b",
        lease_duration_seconds=10,
    )
    current[0] += timedelta(seconds=11)

    assert store.expire_session_leases(project_a) == 1
    assert store.get_session(project_a, session_a.id).status == "interrupted"
    assert store.get_session(project_b, session_b.id).status == "active"


def test_independent_processes_share_project_but_not_same_session(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    ready_path = tmp_path / "owner-ready"
    release_path = tmp_path / "owner-release"
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(database_path)
    owned = store.create_session(project, session_id="owned")
    store.set_status(project, owned.id, "closed")
    owner_script = """
import sys
import time
from pathlib import Path
from mycode.project import ProjectIdentity
from mycode.session_store import SessionStore

database_path, workspace_path, ready_path, release_path = map(Path, sys.argv[1:])
project = ProjectIdentity.from_workspace(workspace_path)
store = SessionStore(database_path)
store.acquire_session_lease(
    project,
    "owned",
    "process-owner",
    lease_duration_seconds=10,
)
ready_path.touch()
deadline = time.monotonic() + 8
while not release_path.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("release signal timed out")
    time.sleep(0.01)
store.release_session_lease(project, "owned", "process-owner", "closed")
"""
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            owner_script,
            str(database_path),
            str(tmp_path),
            str(ready_path),
            str(release_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_path.exists() and owner.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("owner process did not acquire its lease in time")
            time.sleep(0.01)
        if owner.poll() is not None:
            stdout, stderr = owner.communicate()
            pytest.fail(f"owner process exited early: {stdout}\n{stderr}")

        second_store = SessionStore(database_path)
        independent = second_store.create_session(
            project,
            session_id="independent",
            lease_owner_id="second-process",
        )
        with pytest.raises(SessionInUseError, match="another agent"):
            second_store.acquire_session_lease(
                project,
                owned.id,
                "second-process",
            )
        second_store.release_session_lease(
            project,
            independent.id,
            "second-process",
            "closed",
        )
    finally:
        release_path.touch()
        stdout, stderr = owner.communicate(timeout=10)
        if owner.returncode != 0:
            pytest.fail(f"owner process failed: {stdout}\n{stderr}")

    resumed = store.acquire_session_lease(project, owned.id, "after-release")
    assert resumed.status == "active"
    store.release_session_lease(project, owned.id, "after-release", "closed")


def test_session_can_be_recovered_after_owner_process_exits_without_release(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(database_path)
    session = store.create_session(project, session_id="crashed")
    store.set_status(project, session.id, "closed")
    crash_script = """
import sys
from pathlib import Path
from mycode.project import ProjectIdentity
from mycode.session_store import SessionStore

database_path, workspace_path = map(Path, sys.argv[1:])
project = ProjectIdentity.from_workspace(workspace_path)
SessionStore(database_path).acquire_session_lease(
    project,
    "crashed",
    "crashed-owner",
    lease_duration_seconds=0.3,
)
"""

    result = subprocess.run(
        [sys.executable, "-c", crash_script, str(database_path), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(SessionInUseError, match="another agent"):
        store.acquire_session_lease(project, session.id, "recovery-owner")

    time.sleep(0.5)
    recovered = store.acquire_session_lease(
        project,
        session.id,
        "recovery-owner",
    )
    assert recovered.status == "active"
    store.release_session_lease(
        project,
        session.id,
        "recovery-owner",
        "closed",
    )


def test_create_and_list_sessions_are_project_scoped(tmp_path: Path) -> None:
    project_a_root = tmp_path / "project-a"
    project_b_root = tmp_path / "project-b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    now = datetime(2026, 7, 12, 1, 0, tzinfo=timezone.utc)
    times = iter([now, now + timedelta(minutes=1)])
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: next(times))

    session_a = store.create_session(project_a, title="Project A", session_id="a")
    store.create_session(project_b, title="Project B", session_id="b")

    assert session_a.status == "closed"
    assert store.list_sessions(project_a) == [session_a]
    assert store.get_session(project_a, "b") is None


def test_append_and_load_conversation_preserves_tool_relationships(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")
    tool_call = AgentToolCall(
        id="call-1",
        name="read_file",
        arguments={"path": "README.md", "max_lines": 20},
    )
    messages = [
        Message(role="user", content="Read README"),
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
        Message(role="tool", content="OK", tool_call_id="call-1"),
        Message(role="assistant", content="Done"),
    ]

    sequences = store.append_messages(project, session.id, messages)
    restored = store.load_conversation(project, session.id)

    assert sequences == [0, 1, 2, 3]
    assert restored.get_messages() == messages


def test_append_and_load_conversation_preserves_present_empty_reasoning(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="empty-reasoning")
    tool_call = AgentToolCall(
        id="call-empty",
        name="read_file",
        arguments={"path": "README.md"},
    )
    message = Message(
        role="assistant",
        content="",
        tool_calls=(tool_call,),
        reasoning_state="present_empty",
    )

    store.append_message(project, session.id, message)
    restored = store.load_conversation(project, session.id).get_messages()[0]

    assert restored == message
    assert restored.to_model_dict()["reasoning_content"] is None


def test_load_conversation_rejects_inconsistent_reasoning_state(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="invalid-reasoning")
    tool_call = AgentToolCall(
        id="call-invalid",
        name="read_file",
        arguments={"path": "README.md"},
    )
    store.append_message(
        project,
        session.id,
        Message(
            role="assistant",
            content="",
            tool_calls=(tool_call,),
            reasoning_content="private synthetic reasoning",
        ),
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE session_messages SET reasoning_state = 'absent' "
            "WHERE session_id = ?",
            (session.id,),
        )

    with pytest.raises(SessionDataError, match="inconsistent"):
        store.load_conversation(project, session.id)


def test_concurrent_writers_allocate_unique_sequences(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    primary_store = SessionStore(database_path)
    session = primary_store.create_session(project, session_id="session")
    writer_count = 8
    stores = [SessionStore(database_path) for _ in range(writer_count)]
    barrier = Barrier(writer_count)

    def append_from_writer(index: int) -> int:
        barrier.wait()
        return stores[index].append_message(
            project,
            session.id,
            Message(role="user", content=f"writer-{index}"),
        )

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        sequences = list(executor.map(append_from_writer, range(writer_count)))

    assert sorted(sequences) == list(range(writer_count))
    restored = primary_store.load_conversation(project, session.id).get_messages()
    assert len(restored) == writer_count
    assert {message.content for message in restored} == {
        f"writer-{index}" for index in range(writer_count)
    }


def test_lock_timeout_is_reported_as_session_store_error(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    project = ProjectIdentity.from_workspace(tmp_path)
    primary_store = SessionStore(database_path)
    session = primary_store.create_session(project, session_id="session")
    blocked_store = SessionStore(database_path, busy_timeout_seconds=0.01)
    blocker = sqlite3.connect(database_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SessionStoreError, match="database is locked"):
            blocked_store.append_message(
                project,
                session.id,
                Message(role="user", content="blocked"),
            )
    finally:
        blocker.rollback()
        blocker.close()


def test_append_rejects_system_message_without_partial_write(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")

    with pytest.raises(SessionDataError, match="System messages"):
        store.append_messages(
            project,
            session.id,
            [
                Message(role="user", content="kept out"),
                Message(role="system", content="do not persist"),
            ],
        )

    assert store.load_conversation(project, session.id).get_messages() == []


def test_append_rejects_non_json_tool_arguments(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")
    tool_call = AgentToolCall(
        id="call-1",
        name="fake",
        arguments={"invalid": object()},
    )

    with pytest.raises(SessionDataError, match="valid JSON"):
        store.append_message(
            project,
            session.id,
            Message(role="assistant", content="", tool_calls=(tool_call,)),
        )


def test_incomplete_tool_chain_is_preserved_without_fabricated_result(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="session")
    assistant_message = Message(
        role="assistant",
        content="",
        tool_calls=(
            AgentToolCall(id="call-1", name="read_file", arguments={"path": "x"}),
        ),
    )

    store.append_message(project, session.id, assistant_message)

    assert store.load_conversation(project, session.id).get_messages() == [
        assistant_message
    ]


def test_session_management_updates_and_deletes_project_session(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, title="Old", session_id="session")
    tool_call = AgentToolCall(
        id="call-delete",
        name="read_file",
        arguments={"path": "README.md"},
    )
    store.append_messages(
        project,
        session.id,
        [
            Message(role="user", content="hello"),
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            Message(
                role="tool",
                content="OK\nartifact reference",
                tool_call_id=tool_call.id,
            ),
        ],
    )
    store.save_compact_state(project, session.id, CompactState())

    renamed = store.rename_session(project, session.id, "  New   title  ")
    closed = store.set_status(project, session.id, "closed")

    assert renamed.title == "New title"
    assert closed.status == "closed"
    deleted = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )
    deleted_again = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )
    assert deleted.completed is True
    assert deleted.already_absent is False
    assert deleted_again.already_absent is True
    assert store.get_session(project, session.id) is None

    with sqlite3.connect(store.database_path) as connection:
        message_count = connection.execute(
            "SELECT COUNT(*) FROM session_messages"
        ).fetchone()[0]
        compact_state_count = connection.execute(
            "SELECT COUNT(*) FROM session_compact_state"
        ).fetchone()[0]
    assert message_count == 0
    assert compact_state_count == 0


def test_cross_project_append_and_load_are_rejected(tmp_path: Path) -> None:
    project_a_root = tmp_path / "project-a"
    project_b_root = tmp_path / "project-b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project_a, session_id="session")

    with pytest.raises(SessionNotFoundError, match="current project"):
        store.append_message(
            project_b,
            session.id,
            Message(role="user", content="cross project"),
        )
    with pytest.raises(SessionNotFoundError, match="current project"):
        store.load_conversation(project_b, session.id)


def test_duplicate_session_id_is_rejected(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    store.create_session(project, session_id="same")

    with pytest.raises(SessionStoreError, match="already exists"):
        store.create_session(project, session_id="same")
