import json
from pathlib import Path
import subprocess
import sys

import pytest

from mycode.agent import AgentToolCall
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_store import SessionStore, SessionDataError, SessionInUseError, SessionStoreError


def test_full_message_round_trip_and_project_isolation(tmp_path):
    project = ProjectIdentity.from_workspace(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = ProjectIdentity.from_workspace(other_root)
    store = SessionStore(tmp_path / "projects")
    record = store.create_session(project, session_id="one")
    messages = [
        Message(role="user", content="你好"),
        Message(role="assistant", content="", tool_calls=(
            AgentToolCall(id="call", name="read_file", arguments={"path": "a"}),
        ), reasoning_content="private reasoning", reasoning_state="present_nonempty"),
        Message(role="tool", content="result", tool_call_id="call"),
        Message(role="assistant", content="done"),
    ]
    store.append_messages(project, record.id, messages)
    assert store.load_conversation(project, record.id).get_messages() == messages
    assert store.list_sessions(other) == []
    assert store.get_session(other, record.id) is None
    assert store.rename_session(project, record.id, "Renamed").title == "Renamed"
    layout = store.project_storage(project).session(record.id)
    raw = layout.transcript_path.read_text(encoding="utf-8")
    assert "reasoning_content" in raw
    assert [json.loads(line)["sequence"] for line in raw.splitlines()] == [1, 2, 3, 4]
    assert "active" not in layout.meta_path.read_text()
    assert not layout.lock_path.is_relative_to(layout.root)


@pytest.mark.parametrize("tail", [b"", b'{"unfinished":'])
def test_writable_resume_normalizes_tail_before_append(tmp_path, tail):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    store.create_session(project, session_id="one")
    store.append_message(project, "one", Message(role="user", content="first"))
    path = store.project_storage(project).session("one").transcript_path
    good = path.read_bytes()
    path.write_bytes(good.rstrip(b"\n") if not tail else good + tail)
    with store.open_session(project, "one") as writer:
        writer.append_messages([Message(role="assistant", content="second")])
    assert [m.content for m in store.load_conversation(project, "one").get_messages()] == [
        "first", "second",
    ]
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("bad", [b"bad\n", b"bad\r\n", b"bad\n{}\n"])
def test_resume_corruption_fails_closed_and_releases_lock(tmp_path, bad):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    store.create_session(project, session_id="one")
    path = store.project_storage(project).session("one").transcript_path
    path.write_bytes(bad)
    with pytest.raises(SessionDataError):
        with store.open_session(project, "one"):
            pytest.fail("corrupt session opened")
    assert path.read_bytes() == bad
    path.write_bytes(b"")
    with store.open_session(project, "one"):
        pass


@pytest.mark.parametrize("field,value", [
    ("role", "system"), ("content", 3), ("sequence", 2),
    ("tool_calls", [{"id": "x", "name": "tool", "arguments": []}]),
    ("reasoning_state", "invalid"),
])
def test_transcript_domain_validation(tmp_path, field, value):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    store.create_session(project, session_id="one")
    store.append_message(project, "one", Message(role="user", content="hi"))
    path = store.project_storage(project).session("one").transcript_path
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(SessionDataError):
        store.load_conversation(project, "one")


def test_writer_is_scoped_and_same_process_second_owner_is_rejected(tmp_path):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    with store.open_session(project, "one", create=True) as writer:
        assert store.get_session(project, "one").status == "active"
        with pytest.raises(SessionInUseError):
            store.append_message(project, "one", Message(role="user", content="other"))
        with pytest.raises(SessionInUseError):
            store.delete_session(project, "one")
    with pytest.raises(SessionStoreError, match="closed"):
        writer.append_messages([Message(role="user", content="late")])


def test_cross_process_owner_crash_releases_lock(tmp_path):
    project = ProjectIdentity.from_workspace(tmp_path)
    store = SessionStore(tmp_path / "projects")
    store.create_session(project, session_id="one")
    script = (
        "import sys\n"
        "from mycode.project import ProjectIdentity\n"
        "from mycode.session_store import SessionStore\n"
        f"store = SessionStore({str(tmp_path / 'projects')!r})\n"
        f"project = ProjectIdentity.from_workspace({str(tmp_path)!r})\n"
        "with store.open_session(project, 'one'):\n"
        " print('locked', flush=True)\n"
        " sys.stdin.read()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(SessionInUseError):
            with store.open_session(project, "one"):
                pass
    finally:
        child.kill()
        child.communicate(timeout=10)
    with store.open_session(project, "one") as resumed:
        resumed.append_messages([Message(role="user", content="after crash")])
    assert store.load_conversation(project, "one").get_messages()[0].content == "after crash"


def test_owner_context_preserves_callers_exception(tmp_path):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    failure = OSError("model transport failure")
    with pytest.raises(OSError) as caught:
        with store.open_session(project, "one", create=True):
            raise failure
    assert caught.value is failure
    with store.open_session(project, "one"):
        pass


@pytest.mark.parametrize("identifier", ["", "../x", "a/b", "a\\b", "CON", "a:stream"])
def test_invalid_session_ids_are_rejected(tmp_path, identifier):
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    with pytest.raises(SessionDataError):
        store.create_session(project, session_id=identifier)
