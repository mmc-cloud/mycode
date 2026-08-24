from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Event

import pytest

import mycode.session_deletion as session_deletion_module
from mycode.artifacts import (
    ArtifactCleanupError,
    ToolResultArtifactStore,
    artifact_directory_for_session,
    artifact_quarantine_directory,
)
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_deletion import SessionDeletionManager
from mycode.session_lock import SessionLockTimeoutError
from mycode.session_store import (
    SessionMaintenanceError,
    SessionNotFoundError,
    SessionStore,
)


def test_hard_delete_scrubs_sqlite_wal_artifacts_and_deletion_task(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    database = tmp_path / "state.sqlite3"
    store = SessionStore(database)
    session = store.create_session(project, session_id="delete-me")
    marker = ("MYCODE_HARD_DELETE_UNIQUE_BODY_" + "x" * 40) * 32
    store.append_message(
        project,
        session.id,
        Message(role="user", content=marker),
    )
    store.checkpoint_wal_truncate()
    assert marker.encode("utf-8") in database.read_bytes()
    held_connection = sqlite3.connect(database)
    assert held_connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    held_connection.execute("SELECT COUNT(*) FROM sessions").fetchone()

    artifact_directory = artifact_directory_for_session(
        database.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    (artifact_directory / f"{'a' * 64}.txt").write_text(
        "artifact body",
        encoding="utf-8",
    )

    try:
        result = SessionDeletionManager(store).request_and_process(
            project,
            session.id,
        )
    finally:
        held_connection.close()

    assert result.completed is True
    assert result.artifact_removed is True
    assert store.get_session(project, session.id) is None
    assert store.get_session_deletion(project, session.id) is None
    assert not artifact_directory.exists()
    assert marker.encode("utf-8") not in database.read_bytes()
    assert session.id.encode("utf-8") not in database.read_bytes()
    wal_path = Path(f"{database}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM session_messages WHERE session_id = ?",
                (session.id,),
            ).fetchone()[0]
            == 0
        )
        maintenance = connection.execute(
            """
            SELECT post_delete_scrub_required
            FROM database_maintenance_state
            WHERE id = 1
            """
        ).fetchone()[0]
        assert maintenance == 0
        assert (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM session_deletion_tasks
                WHERE session_id = ?
                """,
                (session.id,),
            ).fetchone()[0]
            == 0
        )


def test_expired_owner_cannot_recreate_artifact_after_completed_delete(
    tmp_path: Path,
) -> None:
    current = [datetime(2026, 7, 26, tzinfo=timezone.utc)]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    session = store.create_session(
        project,
        session_id="stale",
        lease_owner_id="stale-owner",
        lease_duration_seconds=1,
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    stale_store = ToolResultArtifactStore(
        root=artifact_directory,
        threshold_chars=8,
        write_guard=lambda: store.artifact_write_guard(
            project,
            session.id,
            "stale-owner",
        ),
    )
    stale_store.externalize(
        tool_name="read_file",
        tool_call_id="before-delete",
        content="payload before deletion",
    )
    current[0] += timedelta(seconds=2)

    result = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert result.completed is True
    assert not artifact_directory.exists()
    with pytest.raises(SessionNotFoundError):
        stale_store.externalize(
            tool_name="read_file",
            tool_call_id="after-delete",
            content="payload written by an expired owner after deletion",
        )
    assert not artifact_directory.exists()


def test_artifact_write_guard_blocks_cross_process_deletion_tombstone(
    tmp_path: Path,
) -> None:
    current = [datetime(2026, 7, 26, tzinfo=timezone.utc)]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = ProjectIdentity.from_workspace(workspace)
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    session = store.create_session(
        project,
        session_id="locked-write",
        lease_owner_id="writer",
        lease_duration_seconds=1,
    )
    future_now = (current[0] + timedelta(seconds=2)).isoformat()
    script = (
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "from mycode.project import ProjectIdentity\n"
        "from mycode.session_deletion import SessionDeletionManager\n"
        "from mycode.session_lock import SessionLockTimeoutError\n"
        "from mycode.session_store import SessionStore\n"
        f"database = Path({str(store.database_path)!r})\n"
        f"workspace = Path({str(workspace)!r})\n"
        f"future_now = datetime.fromisoformat({future_now!r})\n"
        "store = SessionStore(database, now=lambda: future_now, "
        "session_lock_timeout_seconds=0.1)\n"
        "try:\n"
        "    SessionDeletionManager(store).request_and_process("
        "ProjectIdentity.from_workspace(workspace), 'locked-write')\n"
        "    print('unexpected-delete')\n"
        "except SessionLockTimeoutError:\n"
        "    print('lock-timeout')\n"
    )

    with store.artifact_write_guard(project, session.id, "writer"):
        attempted = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )

    assert attempted.stdout.strip() == "lock-timeout"
    assert store.get_session_deletion(project, session.id) is None
    assert store.get_session(project, session.id) is not None
    current[0] += timedelta(seconds=2)

    completed = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert completed.completed is True
    assert store.get_session(project, session.id) is None


def test_artifact_cleanup_failure_keeps_durable_task_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="cleanup-retry")
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'b' * 64}.txt"
    artifact_file.write_text("tracked residue", encoding="utf-8")
    original_delete = session_deletion_module.delete_quarantined_artifacts
    attempts = 0

    def fail_once(*args, **kwargs) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ArtifactCleanupError("synthetic private detail")
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(
        session_deletion_module,
        "delete_quarantined_artifacts",
        fail_once,
    )

    first = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert first.completed is False
    assert first.pending_stage == "database_deleted"
    assert first.error_code == "artifact_cleanup_failed"
    task = store.get_session_deletion(project, session.id)
    assert task is not None
    quarantine = artifact_quarantine_directory(
        store.database_path.parent,
        deletion_id=task.id,
    )
    assert not artifact_file.exists()
    assert (quarantine / artifact_file.name).is_file()

    restarted_store = SessionStore(store.database_path)
    retried = SessionDeletionManager(restarted_store).retry_all_pending()

    assert len(retried) == 1
    assert retried[0].completed is True
    assert restarted_store.list_session_deletions(project) == []
    assert not quarantine.exists()


def test_quarantine_failure_hides_session_but_preserves_tracked_data_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="quarantine-retry")
    store.append_message(
        project,
        session.id,
        Message(role="user", content="preserve until retry"),
    )
    original_quarantine = session_deletion_module.quarantine_session_artifacts
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ArtifactCleanupError("synthetic private detail")
        return original_quarantine(*args, **kwargs)

    monkeypatch.setattr(
        session_deletion_module,
        "quarantine_session_artifacts",
        fail_once,
    )

    first = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert first.completed is False
    assert first.pending_stage == "pending"
    assert first.error_code == "artifact_quarantine_failed"
    assert store.get_session(project, session.id) is None
    assert store.list_sessions(project) == []
    with closing(sqlite3.connect(store.database_path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?",
                (session.id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM session_messages WHERE session_id = ?",
                (session.id,),
            ).fetchone()[0]
            == 1
        )

    retried = SessionDeletionManager(store).retry_all_pending()

    assert retried[0].completed is True
    with closing(sqlite3.connect(store.database_path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?",
                (session.id,),
            ).fetchone()[0]
            == 0
        )


def test_vacuum_failure_remains_persisted_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="vacuum-retry")
    original_vacuum = store.vacuum_database
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SessionMaintenanceError("synthetic private detail")
        original_vacuum()

    monkeypatch.setattr(store, "vacuum_database", fail_once)

    first = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert first.completed is False
    assert first.pending_stage == "post_delete_scrub"
    assert first.error_code == "sqlite_maintenance_pending"
    assert store.get_session_deletion(project, session.id) is None
    assert store.load_database_maintenance_state().post_delete_scrub_required

    retried = SessionDeletionManager(store).retry_all_pending()

    assert retried[0].completed is True
    assert store.get_session_deletion(project, session.id) is None


def test_post_delete_scrub_is_deferred_while_unrelated_lease_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [datetime(2026, 7, 27, tzinfo=timezone.utc)]
    project_a_root = tmp_path / "project-a"
    project_b_root = tmp_path / "project-b"
    project_a_root.mkdir()
    project_b_root.mkdir()
    project_a = ProjectIdentity.from_workspace(project_a_root)
    project_b = ProjectIdentity.from_workspace(project_b_root)
    store = SessionStore(tmp_path / "state.sqlite3", now=lambda: current[0])
    target = store.create_session(project_a, session_id="delete-target")
    active = store.create_session(
        project_b,
        session_id="unrelated-active",
        lease_owner_id="owner-b",
        lease_duration_seconds=1,
    )
    vacuum_called = False
    original_vacuum = store.vacuum_database

    def recording_vacuum() -> None:
        nonlocal vacuum_called
        vacuum_called = True
        original_vacuum()

    monkeypatch.setattr(store, "vacuum_database", recording_vacuum)

    pending = SessionDeletionManager(store).request_and_process(
        project_a,
        target.id,
    )

    assert pending.completed is False
    assert pending.pending_stage == "post_delete_scrub"
    assert pending.error_code == "active_session_leases"
    assert vacuum_called is False
    assert store.get_session(project_a, target.id) is None
    assert store.get_session_deletion(project_a, target.id) is None

    current[0] += timedelta(seconds=0.4)
    store.renew_session_lease(
        project_b,
        active.id,
        "owner-b",
        lease_duration_seconds=1,
    )
    store.release_session_lease(project_b, active.id, "owner-b", "closed")

    retried = SessionDeletionManager(store).retry_all_pending()

    assert len(retried) == 1
    assert retried[0].maintenance_only is True
    assert retried[0].completed is True
    assert vacuum_called is True


def test_new_owner_cannot_race_with_post_delete_vacuum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    database = tmp_path / "state.sqlite3"
    store = SessionStore(database)
    target = store.create_session(project, session_id="delete-target")
    resumable = store.create_session(project, session_id="resumable")
    vacuum_started = Event()
    allow_vacuum = Event()
    original_vacuum = store.vacuum_database

    def slow_vacuum() -> None:
        vacuum_started.set()
        assert allow_vacuum.wait(timeout=2)
        original_vacuum()

    monkeypatch.setattr(store, "vacuum_database", slow_vacuum)
    manager = SessionDeletionManager(store)

    with ThreadPoolExecutor(max_workers=1) as executor:
        deletion = executor.submit(
            manager.request_and_process,
            project,
            target.id,
        )
        assert vacuum_started.wait(timeout=2)
        contender = SessionStore(
            database,
            session_lock_timeout_seconds=0.05,
        )
        with pytest.raises(SessionLockTimeoutError):
            contender.acquire_session_lease(
                project,
                resumable.id,
                "racing-owner",
            )
        allow_vacuum.set()
        assert deletion.result(timeout=2).completed is True

    acquired = contender.acquire_session_lease(
        project,
        resumable.id,
        "racing-owner",
    )
    assert acquired.status == "active"


def test_final_checkpoint_failure_rearms_anonymous_scrub_for_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(project, session_id="final-checkpoint")
    original_checkpoint = store.checkpoint_wal_truncate
    calls = 0

    def fail_final_checkpoint_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise SessionMaintenanceError("synthetic private detail")
        original_checkpoint()

    monkeypatch.setattr(
        store,
        "checkpoint_wal_truncate",
        fail_final_checkpoint_once,
    )

    pending = SessionDeletionManager(store).request_and_process(
        project,
        session.id,
    )

    assert pending.completed is False
    assert pending.pending_stage == "post_delete_scrub"
    assert store.get_session_deletion(project, session.id) is None
    assert store.load_database_maintenance_state().post_delete_scrub_required
    assert session.id.encode() not in store.database_path.read_bytes()

    retried = SessionDeletionManager(store).retry_all_pending()

    assert retried[0].completed is True
    assert not store.load_database_maintenance_state().post_delete_scrub_required


@pytest.mark.parametrize(
    "legacy_stage",
    [
        "pending",
        "database_deleted",
        "initial_checkpoint_complete",
        "vacuum_complete",
        "final_checkpoint_complete",
    ],
)
def test_schema_v5_deletion_stage_resumes_through_anonymous_scrub(
    tmp_path: Path,
    legacy_stage: str,
) -> None:
    case_root = tmp_path / legacy_stage
    case_root.mkdir()
    project = ProjectIdentity.from_workspace(case_root)
    database = case_root / "state.sqlite3"
    store = SessionStore(database)
    session = store.create_session(project, session_id=f"legacy-{legacy_stage}")
    task = store.request_session_deletion(project, session.id)
    assert task is not None
    with sqlite3.connect(database) as connection:
        if legacy_stage != "pending":
            connection.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session.id,),
            )
        connection.execute(
            "UPDATE session_deletion_tasks SET stage = ? WHERE id = ?",
            (legacy_stage, task.id),
        )
        connection.execute("DROP TABLE database_maintenance_state")
        connection.execute("PRAGMA user_version = 5")

    migrated = SessionStore(database)
    results = SessionDeletionManager(migrated).retry_all_pending()

    assert len(results) == 1
    assert results[0].completed is True
    assert migrated.get_session_deletion(project, session.id) is None
    assert not migrated.load_database_maintenance_state().post_delete_scrub_required
