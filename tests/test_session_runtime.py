import pytest
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_runtime import SessionStartRequest, start_project_session
from mycode.session_store import SessionStore, SessionInUseError, SessionNotFoundError


@pytest.mark.parametrize("mode", ["new", "continue", "select"])
def test_new_close_and_resume(tmp_path, mode):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    active = start_project_session(
        store, project, request=SessionStartRequest(mode=mode),
        output_func=lambda _: None,
    )
    active.persist_message(Message(role="user", content="first title"))
    active.close()
    active.close()
    assert store.get_session(project, active.record.id).status == "closed"
    resumed = start_project_session(
        store, project,
        request=SessionStartRequest(mode="resume", session_id=active.record.id),
        output_func=lambda _: None,
    )
    assert resumed.load_history().get_messages() == [Message(role="user", content="first title")]
    assert resumed.record.title == "first title"
    resumed.interrupt()
    assert store.get_session(project, active.record.id).status == "interrupted"


def test_select_and_continue_only_current_project(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = ProjectIdentity.from_workspace(other_path)
    store.create_session(other, title="secret", session_id="other")
    first = store.create_session(project, title="current", session_id="first")
    outputs = []
    inputs = iter(["invalid", "1"])
    active = start_project_session(
        store, project, input_func=lambda _: next(inputs), output_func=outputs.append,
    )
    assert active.record.id == first.id
    assert not any("secret" in line for line in outputs)
    active.close()
    active = start_project_session(
        store, project, request=SessionStartRequest(mode="continue"), output_func=lambda _: None,
    )
    assert active.record.id == first.id
    active.close()
    with pytest.raises(SessionNotFoundError):
        start_project_session(store, project, request=SessionStartRequest(mode="resume", session_id="other"))


@pytest.mark.parametrize("confirm,deleted", [("DELETE one", True), ("wrong", False)])
def test_select_delete_requires_exact_confirmation(tmp_path, confirm, deleted):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="one")
    inputs = iter(["d", "1", confirm, "q"])
    assert start_project_session(
        store, project, input_func=lambda _: next(inputs), output_func=lambda _: None,
    ) is None
    assert (store.get_session(project, "one") is None) == deleted


def test_lifecycle_release_even_if_metadata_write_or_start_output_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    active = start_project_session(store, project, request=SessionStartRequest(mode="new"), output_func=lambda _: None)
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(active.writer, "_save_metadata", fail)
    with pytest.raises(Exception):
        active.close()
    with store.open_session(project, active.record.id):
        pass
    def bad_output(_):
        raise RuntimeError("output failed")
    with pytest.raises(RuntimeError):
        start_project_session(
            store, project, request=SessionStartRequest(mode="resume", session_id=active.record.id),
            output_func=bad_output,
        )
    with store.open_session(project, active.record.id):
        pass
