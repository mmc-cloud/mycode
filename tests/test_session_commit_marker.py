import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mycode.context.compact import CompactBoundary, CompactState, CompactSummary, DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.application.sessions import SessionStartRequest, start_project_session
from mycode.persistence.session_store import SessionDataError, SessionInUseError, SessionStore, SessionStoreError
import mycode.persistence.session_store as persistence


@pytest.fixture
def session(tmp_path):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    store.create_session(project, session_id="one")
    storage = store.project_storage(project)
    return SimpleNamespace(store=store, project=project, storage=storage, layout=storage.session("one"))


def test_orphans_and_non_session_entries_do_not_poison_select(session):
    orphan = session.storage.session("orphan", create=True)
    orphan.transcript_path.touch()
    (session.storage.sessions_directory / "ordinary-file").touch()
    (session.storage.sessions_directory / "invalid name").mkdir()
    assert session.store.get_session(session.project, "missing") is None
    assert session.store.get_session(session.project, "orphan") is None
    assert [r.id for r in session.store.list_sessions(session.project)] == ["one"]
    active = start_project_session(
        session.store,
        session.project,
        request=SessionStartRequest(mode="resume", session_id="one"),
    )
    try:
        assert active.record.id == "one"
    finally:
        active.close()


@pytest.mark.parametrize("damage", ["json", "schema", "identity"])
def test_committed_metadata_corruption_is_not_an_orphan(session, damage):
    data = json.loads(session.layout.meta_path.read_text())
    if damage == "schema":
        data["title"] = []
    elif damage == "identity":
        data["session_id"] = "different"
    payload = "{" if damage == "json" else json.dumps(data)
    session.layout.meta_path.write_text(payload)
    for operation in (
        lambda: session.store.get_session(session.project, "one"),
        lambda: session.store.list_sessions(session.project),
        lambda: session.store.create_session(session.project, session_id="one"),
    ):
        with pytest.raises(SessionDataError):
            operation()
        assert session.layout.meta_path.read_text() == payload


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_create_failure_is_invisible_and_same_id_can_be_retried(session, monkeypatch, cleanup_fails):
    orphan = session.storage.session("retry")
    original_write = persistence.write_json_snapshot

    def fail_commit(root, path, data):
        if path == orphan.meta_path:
            assert orphan.transcript_path.is_file()
            assert orphan.artifacts_directory.is_dir()
            assert orphan.subagents_directory.is_dir()
            raise PermissionError("commit failed")
        return original_write(root, path, data)

    with monkeypatch.context() as fault:
        fault.setattr(persistence, "write_json_snapshot", fail_commit)
        if cleanup_fails:
            fault.setattr(persistence.shutil, "rmtree", lambda _: (_ for _ in ()).throw(OSError("cleanup failed")))
        with pytest.raises(SessionStoreError):
            session.store.create_session(session.project, session_id="retry")
    assert orphan.root.exists() == cleanup_fails
    assert session.store.get_session(session.project, "retry") is None
    assert [r.id for r in session.store.list_sessions(session.project)] == ["one"]
    session.store.create_session(session.project, session_id="retry")
    assert session.store.get_session(session.project, "retry").id == "retry"
    with pytest.raises(SessionDataError, match="already exists"):
        session.store.create_session(session.project, session_id="retry")


def test_delete_revoke_marker_before_failed_cleanup_then_retry(session, monkeypatch):
    with session.store.open_session(session.project, "one"):
        with pytest.raises(SessionInUseError):
            session.store.delete_session(session.project, "one")
        assert session.layout.meta_path.is_file()

    def partial_cleanup(root):
        assert not session.layout.meta_path.exists()
        session.layout.transcript_path.unlink()
        raise OSError("crash during physical cleanup")

    with monkeypatch.context() as fault:
        fault.setattr(persistence.shutil, "rmtree", partial_cleanup)
        with pytest.raises(SessionStoreError):
            session.store.delete_session(session.project, "one")
    assert session.layout.root.is_dir()
    assert session.store.get_session(session.project, "one") is None
    assert session.store.list_sessions(session.project) == []
    assert session.store.delete_session(session.project, "one")
    assert not session.layout.root.exists()
    assert session.layout.lock_path.is_file()
    assert not session.store.delete_session(session.project, "one")


@pytest.mark.parametrize("orphan", [False, True])
def test_reparse_deletion_rejected_before_marker_removal(session, monkeypatch, orphan):
    if orphan:
        session.layout.meta_path.unlink()
    child = session.layout.root / "reparse"
    child.touch()
    original = Path.lstat

    def reparse_info(path, *args, **kwargs):
        if path == child:
            return SimpleNamespace(st_file_attributes=0x400, st_mode=original(path).st_mode)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", reparse_info)
    # Windows reparse attribute is also simulated on Unix CI.
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False)
    with pytest.raises(SessionDataError):
        session.store.delete_session(session.project, "one")
    assert session.layout.meta_path.exists() == (not orphan)
    if orphan:
        with pytest.raises(SessionDataError):
            session.store.create_session(session.project, session_id="one")
    assert child.exists()


@pytest.mark.parametrize("damage", ["json", "schema", "boundary", "size"])
def test_only_compact_damage_recovers_with_cooldown(session, monkeypatch, damage):
    with session.store.open_session(session.project, "one") as writer:
        writer.append_messages([Message(role="user", content="history")])
        data = CompactState().model_dump(mode="json")
        if damage == "schema":
            data["consecutive_failure_count"] = -1
        if damage == "boundary":
            data["boundary"] = CompactBoundary(
                boundary_id="too-far", covered_message_count=2, covered_turn_count=1,
                summary=CompactSummary(objective="Continue", progress=(), decisions=(), constraints=(), open_items=(), references=()),
                source_estimated_tokens=100, summary_prompt_tokens=20,
                summary_completion_tokens=10, created_at=datetime.now(timezone.utc),
            ).model_dump(mode="json")
        if damage == "size":
            monkeypatch.setattr(persistence, "MAX_COMPACT_STATE_JSON_CHARS", 300)
            data["last_failure_reason"] = "x" * 301
        session.layout.compact_path.write_text("{" if damage == "json" else json.dumps(data))
        monkeypatch.setattr(session.store, "load_conversation", lambda *_: pytest.fail("writer must use existing message count"))
        recovered = writer.load_or_reset_compact_state()
        assert recovered.recovered_invalid_state
        assert recovered.state.boundary is None
        assert recovered.state.retry_after_message_count == 1 + DEFAULT_COMPACT_FAILURE_COOLDOWN_MESSAGES
        assert recovered.state.consecutive_failure_count == 1
        assert json.loads(session.layout.compact_path.read_text())["last_failure_reason"] == "stored_compact_state_invalid"


@pytest.mark.parametrize("damage", ["transcript", "meta", "project"])
def test_resume_rejects_non_compact_corruption_without_overwriting_compact(session, damage):
    session.layout.compact_path.write_text("{invalid compact too")
    target = {"transcript": session.layout.transcript_path, "meta": session.layout.meta_path,
              "project": session.storage.metadata_path}[damage]
    target.write_text("bad\n")
    with pytest.raises(SessionDataError):
        with session.store.open_session(session.project, "one") as writer:
            writer.load_or_reset_compact_state()
    with pytest.raises(SessionDataError):
        session.store.load_compact_state(session.project, "one")
    assert session.layout.compact_path.read_text() == "{invalid compact too"


@pytest.mark.parametrize("damage", ["meta", "project", "io", "boundary"])
def test_writer_does_not_reset_compact_on_unrelated_error(session, monkeypatch, damage):
    with session.store.open_session(session.project, "one") as writer:
        session.layout.compact_path.write_text("{invalid compact too")
        if damage in {"meta", "project"}:
            target = session.layout.meta_path if damage == "meta" else session.storage.metadata_path
            target.write_text("bad")
        else:
            from mycode.persistence.filesystem import JsonSnapshotError, StorageBoundaryError

            def fail_read(root, path):
                if path == session.layout.compact_path:
                    if damage == "boundary":
                        raise StorageBoundaryError("unsafe path")
                    raise JsonSnapshotError("cannot read") from PermissionError("denied")
                return original_read(root, path)

            original_read = persistence.read_json_snapshot
            monkeypatch.setattr(persistence, "read_json_snapshot", fail_read)
        with pytest.raises(SessionDataError):
            writer.load_or_reset_compact_state()
        assert session.layout.compact_path.read_text() == "{invalid compact too"
