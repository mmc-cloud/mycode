from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from threading import Event

import pytest

import mycode.session_deletion as session_deletion_module
from mycode.artifacts import (
    ArtifactCleanupError,
    artifact_directory_for_session,
    artifact_quarantine_directory,
)
from mycode.context_compact import CompactState
from mycode.messages import Message
from mycode.session_runtime import (
    ActiveProjectSession,
    SessionStartRequest,
    start_project_session,
)
from mycode.session_deletion import SessionDeletionManager
from mycode.session_lock import SessionLockTimeoutError
from mycode.session_store import (
    DEFAULT_SESSION_TITLE,
    ProjectIdentity,
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
)


def test_start_project_session_creates_new_when_project_has_no_history(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    outputs: list[str] = []

    active = start_project_session(
        store,
        project,
        input_func=lambda prompt: pytest.fail("selection input should not be requested"),
        output_func=outputs.append,
    )

    assert active is not None
    assert active.record.title == DEFAULT_SESSION_TITLE
    assert outputs[0] == f"session> 项目 {tmp_path.resolve()}"
    assert "没有历史会话" in outputs[1]


def test_interactive_selection_only_lists_current_project_sessions(
    tmp_path: Path,
) -> None:
    current_root = tmp_path / "current"
    other_root = tmp_path / "other"
    current_root.mkdir()
    other_root.mkdir()
    current = ProjectIdentity.from_workspace(current_root)
    other = ProjectIdentity.from_workspace(other_root)
    store = SessionStore(tmp_path / "state.sqlite3")
    current_session = store.create_session(
        current,
        title="Current project session",
        session_id="current-session",
    )
    store.set_status(current, current_session.id, "closed")
    store.create_session(other, title="Other project secret", session_id="other-session")
    inputs = iter(["invalid", "1"])
    outputs: list[str] = []

    active = start_project_session(
        store,
        current,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is not None
    assert active.record.id == current_session.id
    assert active.record.status == "active"
    assert any("Current project session" in output for output in outputs)
    assert not any("Other project secret" in output for output in outputs)
    assert "session> 选择无效，请重新输入" in outputs


def test_continue_resumes_latest_project_session(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    first = store.create_session(project, title="First", session_id="first")
    store.set_status(project, first.id, "closed")
    second = store.create_session(project, title="Second", session_id="second")
    store.set_status(project, second.id, "closed")

    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="continue"),
        output_func=lambda message: None,
    )

    assert active is not None
    assert active.record.id == second.id


def test_resume_rejects_session_from_other_project(tmp_path: Path) -> None:
    current_root = tmp_path / "current"
    other_root = tmp_path / "other"
    current_root.mkdir()
    other_root.mkdir()
    current = ProjectIdentity.from_workspace(current_root)
    other = ProjectIdentity.from_workspace(other_root)
    store = SessionStore(tmp_path / "state.sqlite3")
    current_session = store.create_session(current, session_id="current-session")
    store.create_session(other, session_id="other-session")

    with pytest.raises(SessionNotFoundError, match="current project"):
        start_project_session(
            store,
            current,
            request=SessionStartRequest(
                mode="resume",
                session_id="other-session",
            ),
        )

    assert store.get_session(current, current_session.id).status == "closed"


def test_resume_rejects_missing_id_without_interrupting_active_session(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    active = store.create_session(project, session_id="active-session")

    with pytest.raises(SessionNotFoundError, match="current project"):
        start_project_session(
            store,
            project,
            request=SessionStartRequest(
                mode="resume",
                session_id="missing-session",
            ),
        )

    assert store.get_session(project, active.id).status == "closed"


def test_interactive_selection_can_quit_without_creating_session(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    existing = store.create_session(project, session_id="existing")

    active = start_project_session(
        store,
        project,
        input_func=lambda prompt: "q",
        output_func=lambda message: None,
    )

    assert active is None
    assert len(store.list_sessions(project)) == 1
    assert store.get_session(project, existing.id).status == "closed"


def test_interactive_delete_requires_exact_confirmation(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        title="Keep this session",
        session_id="keep-session",
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'a' * 64}.txt"
    artifact_file.write_text("keep", encoding="utf-8")
    inputs = iter(["d", "1", "DELETE wrong-session", "q"])
    outputs: list[str] = []

    active = start_project_session(
        store,
        project,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is None
    assert store.get_session(project, session.id) is not None
    assert artifact_file.read_text(encoding="utf-8") == "keep"
    assert any("确认文本不匹配" in output for output in outputs)


def test_interactive_delete_removes_session_state_and_exact_artifacts(
    tmp_path: Path,
) -> None:
    current_root = tmp_path / "current"
    other_root = tmp_path / "other"
    current_root.mkdir()
    other_root.mkdir()
    current = ProjectIdentity.from_workspace(current_root)
    other = ProjectIdentity.from_workspace(other_root)
    store = SessionStore(tmp_path / "state.sqlite3")
    survivor = store.create_session(
        current,
        title="Survivor",
        session_id="s",
    )
    target = store.create_session(
        current,
        title="Delete target",
        session_id="t",
    )
    other_session = store.create_session(
        other,
        title="Other project",
        session_id="o",
    )
    store.append_message(
        current,
        target.id,
        Message(role="user", content="delete this history"),
    )
    store.save_compact_state(current, target.id, CompactState())

    target_artifacts = artifact_directory_for_session(
        store.database_path.parent,
        project_key=current.key,
        session_id=target.id,
    )
    survivor_artifacts = artifact_directory_for_session(
        store.database_path.parent,
        project_key=current.key,
        session_id=survivor.id,
    )
    other_artifacts = artifact_directory_for_session(
        store.database_path.parent,
        project_key=other.key,
        session_id=other_session.id,
    )
    for directory, content in (
        (target_artifacts, "target"),
        (survivor_artifacts, "survivor"),
        (other_artifacts, "other"),
    ):
        directory.mkdir(parents=True)
        (directory / f"{content[0] * 64}.txt").write_text(
            content,
            encoding="utf-8",
        )

    listed = store.list_sessions(current)
    target_index = next(
        index
        for index, session in enumerate(listed, start=1)
        if session.id == target.id
    )
    inputs = iter(
        [
            "d",
            str(target_index),
            f"DELETE {target.id}",
            "q",
        ]
    )
    outputs: list[str] = []

    active = start_project_session(
        store,
        current,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is None
    assert store.get_session(current, target.id) is None
    assert store.get_session(current, survivor.id) is not None
    assert store.get_session(other, other_session.id) is not None
    assert not target_artifacts.exists()
    assert (survivor_artifacts / f"{'s' * 64}.txt").is_file()
    assert (other_artifacts / f"{'o' * 64}.txt").is_file()
    with sqlite3.connect(store.database_path) as connection:
        message_count = connection.execute(
            "SELECT COUNT(*) FROM session_messages WHERE session_id = ?",
            (target.id,),
        ).fetchone()[0]
        compact_count = connection.execute(
            "SELECT COUNT(*) FROM session_compact_state WHERE session_id = ?",
            (target.id,),
        ).fetchone()[0]
    assert message_count == 0
    assert compact_count == 0
    assert any("已永久删除" in output for output in outputs)
    assert "session> artifact 清理完成" in outputs
    assert not any("Other project" in output for output in outputs)


def test_interactive_delete_refuses_session_with_live_lease(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        title="In use",
        session_id="in-use",
        lease_owner_id="owner",
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'a' * 64}.txt"
    artifact_file.write_text("keep", encoding="utf-8")
    inputs = iter(["d", "1", f"DELETE {session.id}", "q"])
    outputs: list[str] = []

    active = start_project_session(
        SessionStore(store.database_path),
        project,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is None
    assert store.get_session(project, session.id) is not None
    assert artifact_file.is_file()
    assert any("当前无法删除" in output for output in outputs)
    store.release_session_lease(project, session.id, "owner", "closed")


def test_interactive_delete_reports_artifact_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        title="Cleanup failure",
        session_id="cleanup-failure",
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'a' * 64}.txt"
    artifact_file.write_text("residue", encoding="utf-8")

    def fail_cleanup(*args, **kwargs) -> bool:
        raise ArtifactCleanupError("synthetic cleanup failure")

    monkeypatch.setattr(
        "mycode.session_deletion.delete_quarantined_artifacts",
        fail_cleanup,
    )
    inputs = iter(["d", "1", f"DELETE {session.id}", "q"])
    outputs: list[str] = []

    active = start_project_session(
        store,
        project,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is None
    assert store.get_session(project, session.id) is None
    deletion = store.get_session_deletion(project, session.id)
    assert deletion is not None
    assert deletion.stage == "database_deleted"
    assert deletion.last_error_code == "artifact_cleanup_failed"
    quarantine_directory = artifact_quarantine_directory(
        store.database_path.parent,
        deletion_id=deletion.id,
    )
    assert not artifact_file.exists()
    assert (quarantine_directory / artifact_file.name).is_file()
    assert any(
        "已记录删除请求，物理清理仍待完成" in output
        for output in outputs
    )


def test_session_start_retries_persisted_artifact_cleanup_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        title="Retry on startup",
        session_id="startup-cleanup",
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'c' * 64}.txt"
    artifact_file.write_text("tracked until startup retry", encoding="utf-8")
    original_delete = session_deletion_module.delete_quarantined_artifacts

    def fail_cleanup(*args, **kwargs) -> bool:
        raise ArtifactCleanupError("synthetic cleanup failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            session_deletion_module,
            "delete_quarantined_artifacts",
            fail_cleanup,
        )
        first = SessionDeletionManager(store).request_and_process(
            project,
            session.id,
        )

    assert first.completed is False
    deletion = store.get_session_deletion(project, session.id)
    assert deletion is not None
    quarantine_directory = artifact_quarantine_directory(
        store.database_path.parent,
        deletion_id=deletion.id,
    )
    assert quarantine_directory.exists()
    assert original_delete is session_deletion_module.delete_quarantined_artifacts
    outputs: list[str] = []

    active = start_project_session(
        SessionStore(store.database_path),
        project,
        request=SessionStartRequest(mode="new"),
        output_func=outputs.append,
    )

    assert active is not None
    assert store.get_session_deletion(project, session.id) is None
    assert not quarantine_directory.exists()
    assert any("已完成待处理的删除任务" in output for output in outputs)
    active.close()


def test_session_start_retries_deletion_for_removed_other_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_a = tmp_path / "project-a"
    workspace_b = tmp_path / "project-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    project_a = ProjectIdentity.from_workspace(workspace_a)
    project_b = ProjectIdentity.from_workspace(workspace_b)
    database_path = tmp_path / "state.sqlite3"
    store = SessionStore(database_path)
    session = store.create_session(
        project_a,
        title="Removed workspace cleanup",
        session_id="gone",
    )
    artifact_directory = artifact_directory_for_session(
        store.database_path.parent,
        project_key=project_a.key,
        session_id=session.id,
    )
    artifact_directory.mkdir(parents=True)
    artifact_file = artifact_directory / f"{'d' * 64}.txt"
    artifact_file.write_text("quarantined residue", encoding="utf-8")

    def fail_cleanup(*args, **kwargs) -> bool:
        raise ArtifactCleanupError("synthetic cleanup failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            session_deletion_module,
            "delete_quarantined_artifacts",
            fail_cleanup,
        )
        first = SessionDeletionManager(store).request_and_process(
            project_a,
            session.id,
        )

    assert first.completed is False
    deletion = store.get_session_deletion(project_a, session.id)
    assert deletion is not None
    quarantine_directory = artifact_quarantine_directory(
        store.database_path.parent,
        deletion_id=deletion.id,
    )
    assert quarantine_directory.exists()

    workspace_a.rmdir()
    with pytest.raises(ValueError, match="does not exist"):
        ProjectIdentity.from_workspace(workspace_a)

    outputs: list[str] = []
    active = start_project_session(
        SessionStore(database_path),
        project_b,
        request=SessionStartRequest(mode="new"),
        output_func=outputs.append,
    )

    assert active is not None
    assert store.get_session_deletion(project_a, session.id) is None
    assert not quarantine_directory.exists()
    assert any(
        "已完成待处理的删除任务" in output
        and project_a.key[:8] in output
        and session.id in output
        for output in outputs
    )
    active.close()


def test_interactive_delete_succeeds_when_session_has_no_artifacts(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    session = store.create_session(
        project,
        title="No artifacts",
        session_id="no-artifacts",
    )
    inputs = iter(["d", "1", f"DELETE {session.id}", "q"])
    outputs: list[str] = []

    active = start_project_session(
        store,
        project,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert active is None
    assert store.get_session(project, session.id) is None
    assert "session> 没有需要清理的 artifact 文件" in outputs
    assert "session> 当前没有历史会话" in outputs


def test_confirmed_new_session_preserves_other_active_session(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    existing = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )
    assert existing is not None

    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )

    assert active is not None
    assert active.record.id != existing.record.id
    assert store.get_session(project, existing.record.id).status == "active"
    assert store.get_session(project, active.record.id).status == "active"
    active.close()
    existing.close()


def test_same_session_cannot_be_resumed_by_two_agents(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    database_path = tmp_path / "state.sqlite3"
    first_store = SessionStore(database_path)
    second_store = SessionStore(database_path)
    first = start_project_session(
        first_store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )
    assert first is not None

    with pytest.raises(SessionInUseError, match="another agent"):
        start_project_session(
            second_store,
            project,
            request=SessionStartRequest(
                mode="resume",
                session_id=first.record.id,
            ),
            output_func=lambda message: None,
        )

    assert first_store.get_session(project, first.record.id).status == "active"
    first.close()


def test_interactive_menu_keeps_running_after_in_use_session_choice(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    first = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )
    assert first is not None
    inputs = iter(["1", "n"])
    outputs: list[str] = []

    second = start_project_session(
        SessionStore(store.database_path),
        project,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert second is not None
    assert second.record.id != first.record.id
    assert any("session> 当前不可用：" in output for output in outputs)
    second.close()
    first.close()


def test_active_session_heartbeat_renews_owned_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    record = store.create_session(
        project,
        session_id="heartbeat",
        lease_owner_id="owner",
    )
    renewed = Event()
    original_renew = store.renew_session_lease

    def recording_renew(*args, **kwargs) -> None:
        original_renew(*args, **kwargs)
        renewed.set()

    monkeypatch.setattr(store, "renew_session_lease", recording_renew)
    active = ActiveProjectSession(
        store=store,
        project=project,
        record=record,
        lease_owner_id="owner",
        heartbeat_interval_seconds=0.01,
    )

    active.start_heartbeat()

    assert renewed.wait(timeout=1)
    active.close()


def test_artifact_write_lock_does_not_block_heartbeat_or_expire_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_now = datetime(2026, 7, 26, tzinfo=timezone.utc)
    current = [initial_now]
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(
        tmp_path / "state.sqlite3",
        now=lambda: current[0],
    )
    record = store.create_session(
        project,
        session_id="long-artifact-write",
        lease_owner_id="owner",
        lease_duration_seconds=0.2,
    )
    original_renew = store.renew_session_lease
    renew_count = 0
    renewed_across_original_expiry = Event()

    def recording_renew(*args, **kwargs) -> None:
        nonlocal renew_count
        current[0] += timedelta(seconds=0.1)
        original_renew(*args, **kwargs)
        renew_count += 1
        if renew_count >= 4:
            renewed_across_original_expiry.set()

    monkeypatch.setattr(store, "renew_session_lease", recording_renew)
    active = ActiveProjectSession(
        store=store,
        project=project,
        record=record,
        lease_owner_id="owner",
        lease_duration_seconds=0.2,
        heartbeat_interval_seconds=0.01,
    )
    contender = SessionStore(
        store.database_path,
        now=lambda: current[0],
        session_lock_timeout_seconds=0.02,
    )

    with active.artifact_write_guard():
        active.start_heartbeat()
        assert renewed_across_original_expiry.wait(timeout=1)
        assert current[0] > initial_now + timedelta(seconds=0.2)
        with pytest.raises(SessionLockTimeoutError):
            SessionDeletionManager(contender).request_and_process(
                project,
                record.id,
            )
        with pytest.raises(SessionLockTimeoutError):
            contender.acquire_session_lease(
                project,
                record.id,
                "second-owner",
                lease_duration_seconds=0.2,
            )

    with pytest.raises(SessionInUseError):
        SessionDeletionManager(contender).request_and_process(
            project,
            record.id,
        )
    with pytest.raises(SessionInUseError):
        contender.acquire_session_lease(
            project,
            record.id,
            "second-owner",
            lease_duration_seconds=0.2,
        )

    active.close()


def test_active_session_persists_messages_titles_and_closes(tmp_path: Path) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )
    assert active is not None

    active.persist_message(
        Message(role="user", content="  Implement   session persistence  ")
    )
    active.persist_message(Message(role="assistant", content="Done"))
    active.close()

    stored = store.get_session(project, active.record.id)
    assert stored is not None
    assert stored.title == "Implement session persistence"
    assert stored.status == "closed"
    assert active.load_history().get_messages() == [
        Message(role="user", content="  Implement   session persistence  "),
        Message(role="assistant", content="Done"),
    ]


def test_explicit_session_title_is_not_overwritten_by_first_user_message(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "state.sqlite3")
    record = store.create_session(
        project,
        title="Manual title",
        session_id="manual-title",
    )
    store.set_status(project, record.id, "closed")
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="resume", session_id=record.id),
        output_func=lambda message: None,
    )
    assert isinstance(active, ActiveProjectSession)

    active.persist_message(Message(role="user", content="First user request"))

    assert store.get_session(project, record.id).title == "Manual title"
    active.close()


def test_session_request_validates_resume_identifier() -> None:
    with pytest.raises(ValueError, match="requires session_id"):
        SessionStartRequest(mode="resume")
    with pytest.raises(ValueError, match="must not include"):
        SessionStartRequest(mode="new", session_id="unexpected")
    with pytest.raises(ValueError, match="Unsupported"):
        SessionStartRequest(mode="invalid")  # type: ignore[arg-type]
