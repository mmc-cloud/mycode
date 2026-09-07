from dataclasses import replace
import json

import pytest

from mycode.agent import AgentToolCall
from mycode.context_compact import CompactState
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.project_storage import ProjectStorageError, validate_storage_component
from mycode.session_runtime import SessionStartRequest, start_project_session
from mycode.session_store import SessionStore, SessionStoreError
import mycode.filesystem as filesystem
import mycode.session_store as persistence


@pytest.mark.parametrize("tail", ["normal", "no-newline", "partial"])
def test_writable_startup_parses_once_and_load_history_uses_writer(tmp_path, monkeypatch, tail):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    history = [Message(role="user", content="first"), Message(role="assistant", content="reply")]
    store.create_session(project, session_id="one")
    store.append_messages(project, "one", history)
    layout = store.project_storage(project).session("one")
    original_bytes = layout.transcript_path.read_bytes()
    if tail == "no-newline":
        layout.transcript_path.write_bytes(original_bytes.rstrip(b"\n"))
    if tail == "partial":
        layout.transcript_path.write_bytes(original_bytes + b'{"sequence":')
    calls = []
    original_read = filesystem.read_jsonl_records

    def counted_read(*args, **kwargs):
        calls.append(args[1])
        return original_read(*args, **kwargs)

    monkeypatch.setattr(filesystem, "read_jsonl_records", counted_read)
    monkeypatch.setattr(persistence, "read_jsonl_records", counted_read)
    active = start_project_session(store, project, request=SessionStartRequest(mode="resume", session_id="one"), output_func=lambda _: None)
    try:
        assert active.load_history().get_messages() == history
        active.load_history().clear()
        assert active.load_history().get_messages() == history
        active.load_compact_state()
        active.persist_message(Message(role="user", content="next"))
        assert active.load_history().get_messages() == history + [Message(role="user", content="next")]
        assert calls == [layout.transcript_path]
    finally:
        active.close()
    assert store.load_conversation(project, "one").get_messages() == history + [Message(role="user", content="next")]
    assert len(calls) == 2  # The public read-only API still reads from disk.


def test_writer_history_tracks_only_successful_appends(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    original_append = persistence.append_jsonl_record
    with store.open_session(project, "one", create=True) as writer:
        def fail_second(root, path, record):
            if record["sequence"] == 2:
                raise OSError("disk failure")
            return original_append(root, path, record)

        with monkeypatch.context() as fault:
            fault.setattr(persistence, "append_jsonl_record", fail_second)
            with pytest.raises(SessionStoreError):
                writer.append_messages([Message(role="user", content="saved"), Message(role="assistant", content="not saved")])
        assert writer.load_history().get_messages() == [Message(role="user", content="saved")]
        assert store.load_conversation(project, "one").get_messages() == writer.load_history().get_messages()
        writer.append_messages([Message(role="assistant", content="retry")])
        assert [json.loads(line)["sequence"] for line in writer.layout.transcript_path.read_text().splitlines()] == [1, 2]
        assert store.load_conversation(project, "one").get_messages() == writer.load_history().get_messages()


def test_incomplete_tool_group_is_preserved_without_replay(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    pending = Message(role="assistant", content="", tool_calls=(AgentToolCall(id="call", name="write_file", arguments={"path": "x"}),))
    store.create_session(project, session_id="one")
    store.append_message(project, "one", pending)
    with store.open_session(project, "one") as writer:
        assert writer.load_history().get_messages() == [pending]
        assert store.load_conversation(project, "one").get_messages() == [pending]
        assert not (tmp_path / "x").exists()


@pytest.fixture
def projects(tmp_path):
    store = SessionStore(tmp_path / "projects")
    storage = []
    for name in ("A", "B"):
        workspace = tmp_path / name
        workspace.mkdir()
        project = ProjectIdentity.from_workspace(workspace)
        store.create_session(project, session_id="one")
        store.append_message(project, "one", Message(role="user", content=name))
        store.save_compact_state(project, "one", CompactState())
        storage.append(store.project_storage(project))
    return store, *storage


@pytest.mark.parametrize("operation", ["read-json", "write-json", "read-jsonl", "append-jsonl", "prepare-jsonl"])
def test_project_primitive_cannot_access_sibling_project(projects, operation):
    _, a, b = projects
    layout = b.session("one")
    path = layout.meta_path if operation.endswith("json") else layout.transcript_path
    before = path.read_bytes()
    actions = {
        "read-json": lambda: filesystem.read_json_snapshot(a.project_directory, path),
        "write-json": lambda: filesystem.write_json_snapshot(a.project_directory, path, {}),
        "read-jsonl": lambda: filesystem.read_jsonl_records(a.project_directory, path),
        "append-jsonl": lambda: filesystem.append_jsonl_record(a.project_directory, path, {}),
        "prepare-jsonl": lambda: filesystem.prepare_jsonl_for_append(a.project_directory, path),
    }
    with pytest.raises(filesystem.StorageBoundaryError):
        actions[operation]()
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["read", "delete", "create"])
def test_redirected_session_directory_cannot_cross_project_boundary(projects, monkeypatch, operation):
    store, a, b = projects
    redirected = replace(a, sessions_directory=b.sessions_directory)
    monkeypatch.setattr(store, "project_storage", lambda _: redirected)
    before = b.session("one").meta_path.read_bytes()
    with pytest.raises(SessionStoreError):
        if operation == "read":
            store.load_conversation(a.identity, "one")
        elif operation == "delete":
            store.delete_session(a.identity, "one")
        else:
            store.create_session(a.identity, session_id="new")
    assert b.session("one").meta_path.read_bytes() == before
    assert not b.session("new").root.exists()


@pytest.mark.parametrize("field,operation", [("transcript_path", "append"), ("meta_path", "rename"), ("compact_path", "compact"), ("subagents_directory", "subagent")])
def test_writer_primitives_reject_foreign_project_paths(projects, field, operation):
    store, a, b = projects
    foreign = b.session("one")
    with store.open_session(a.identity, "one") as writer:
        writer.layout = replace(writer.layout, **{field: getattr(foreign, field)})
        actions = {
            "append": lambda: writer.append_messages([Message(role="assistant", content="escape")]),
            "rename": lambda: writer.rename("escape"),
            "compact": lambda: writer.save_compact_state(CompactState()),
            "subagent": lambda: writer.append_subagent_event("escape", {"type": "state"}),
        }
        before = {p: p.read_bytes() for p in foreign.root.rglob("*") if p.is_file()}
        with pytest.raises(SessionStoreError):
            actions[operation]()
        assert {p: p.read_bytes() for p in foreign.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("value", ["CON", "com1.txt", "a/b", "a\\b", "..", "bad.", "a:stream"])
def test_public_component_validator_keeps_portable_rules(value):
    with pytest.raises(ProjectStorageError):
        validate_storage_component(value, field_name="session_id")
    validate_storage_component("run-123_abc", field_name="run_id")
