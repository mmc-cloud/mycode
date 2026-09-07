import pytest
from mycode.artifacts import ToolResultArtifactStore, ReadArtifactTool
from mycode.project import ProjectIdentity
from mycode.session_store import SessionStore, SessionInUseError, SessionDataError


def test_delete_removes_all_session_data_but_preserves_lock_and_other_session(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    with store.open_session(project, "one", create=True) as one:
        artifact = ToolResultArtifactStore(one.layout.artifacts_directory, 1)
        ref = artifact.externalize(tool_name="read", tool_call_id="a", content="large result")
        one.append_subagent_event("run", {"type": "state", "state": "running"})
        target = one.layout
        with pytest.raises(SessionInUseError):
            store.delete_session(project, "one")
    with store.open_session(project, "two", create=True) as two:
        other = two.layout
        with pytest.raises(ValueError):
            ToolResultArtifactStore(other.artifacts_directory, 1).rehydrate(
                tool_name="read", tool_call_id="a", content=ref,
            )
    assert store.delete_session(project, "one")
    assert not target.root.exists()
    assert target.lock_path.is_file()
    assert other.root.exists()
    assert not store.delete_session(project, "one")


def test_delete_rejects_traversal_and_symlink(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    with pytest.raises(SessionDataError):
        store.delete_session(project, "../escape")
    store.create_session(project, session_id="one")
    root = store.project_storage(project).session("one").root
    outside = tmp_path / "valuable"
    outside.write_text("keep")
    try:
        (root / "link").symlink_to(outside)
    except OSError as error:
        pytest.skip(str(error))
    with pytest.raises(SessionDataError):
        store.delete_session(project, "one")
    assert outside.read_text() == "keep"
