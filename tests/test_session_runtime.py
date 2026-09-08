import pytest
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.application.sessions import SessionStartRequest, start_project_session
from mycode.persistence.session_store import SessionStore, SessionInUseError, SessionNotFoundError


@pytest.mark.parametrize("mode", ["new", "continue"])
def test_new_close_and_resume(tmp_path, mode):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    active = start_project_session(
        store, project, request=SessionStartRequest(mode=mode),
    )
    active.persist_message(Message(role="user", content="first title"))
    active.close()
    active.close()
    assert store.get_session(project, active.record.id).status == "closed"
    resumed = start_project_session(
        store, project,
        request=SessionStartRequest(mode="resume", session_id=active.record.id),
    )
    assert resumed.load_history().get_messages() == [Message(role="user", content="first title")]
    assert resumed.record.title == "first title"
    resumed.interrupt()
    assert store.get_session(project, active.record.id).status == "interrupted"


def test_resume_and_continue_only_current_project(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = ProjectIdentity.from_workspace(other_path)
    store.create_session(other, title="secret", session_id="other")
    first = store.create_session(project, title="current", session_id="first")
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="resume", session_id=first.id),
    )
    assert active.record.id == first.id
    active.close()
    active = start_project_session(
        store, project, request=SessionStartRequest(mode="continue"),
    )
    assert active.record.id == first.id
    active.close()
    with pytest.raises(SessionNotFoundError):
        start_project_session(store, project, request=SessionStartRequest(mode="resume", session_id="other"))


def test_lifecycle_release_even_if_metadata_write_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    active = start_project_session(store, project, request=SessionStartRequest(mode="new"))
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(active.writer, "_save_metadata", fail)
    with pytest.raises(Exception):
        active.close()
    with store.open_session(project, active.record.id):
        pass
    resumed = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="resume", session_id=active.record.id),
    )
    resumed.interrupt()
    with store.open_session(project, active.record.id):
        pass
