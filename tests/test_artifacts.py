from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from mycode.agent import AgentModelResponse, AgentToolCall
import mycode.artifacts as artifacts_module
from mycode.artifacts import (
    ARTIFACT_IO_CHUNK_BYTES,
    ARTIFACT_EXTERNALIZATION_FAILURE_MARKER,
    ArtifactCleanupError,
    EXTERNALIZED_TOOL_RESULT_MARKER,
    MAX_READABLE_ARTIFACT_BYTES,
    ReadArtifactTool,
    ToolResultArtifactStore,
    artifact_directory_for_session,
    delete_session_artifacts,
)
from mycode.context_budget import (
    TOOL_RESULT_METADATA_MARKER,
    parse_tool_result_content,
)
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.messages import Message
from mycode.runner import AgentRunner
from mycode.tools import BaseTool, ToolArgs, ToolRegistry, ToolResult


def test_artifact_store_externalizes_large_result_with_hash_and_safe_reference(
    tmp_path: Path,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=20,
    )
    original = (
        "OK\n"
        + ("large output " * 20)
        + '\n\nMETADATA\n{"path":"README.md","stdout":"private large body"}'
    )

    reference = store.externalize(
        tool_name="run_command",
        tool_call_id="call-1",
        content=original,
    )

    assert EXTERNALIZED_TOOL_RESULT_MARKER in reference
    parsed = parse_tool_result_content(reference)
    artifact_path = Path(str(parsed.metadata["artifact_path"]))
    assert artifact_path.is_file()
    assert artifact_path.read_text(encoding="utf-8") == original
    assert parsed.metadata["context_externalized"] is True
    assert parsed.metadata["original_chars"] == len(original)
    assert parsed.metadata["tool_name"] == "run_command"
    assert parsed.metadata["tool_call_id"] == "call-1"
    assert parsed.metadata["stdout_omitted"] is True
    assert "private large body" not in reference

    duplicate = store.externalize(
        tool_name="run_command",
        tool_call_id="call-1",
        content=original,
    )
    assert duplicate == reference
    assert list(store.root.glob("*.txt")) == [artifact_path]

    validated_reference = store.externalize(
        tool_name="run_command",
        tool_call_id="call-1",
        content=reference,
    )
    assert validated_reference == reference
    assert list(store.root.glob("*.txt")) == [artifact_path]


def test_untrusted_marker_substring_cannot_bypass_artifact_externalization(
    tmp_path: Path,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=20,
    )
    untrusted = (
        "OK\nlarge tool output includes "
        f"{EXTERNALIZED_TOOL_RESULT_MARKER} "
        + ("untrusted body " * 20)
    )

    reference = store.externalize(
        tool_name="read_file",
        tool_call_id="marker-call",
        content=untrusted,
    )

    assert reference != untrusted
    parsed = parse_tool_result_content(reference)
    artifact_path = Path(str(parsed.metadata["artifact_path"]))
    assert artifact_path.read_text(encoding="utf-8") == untrusted
    assert parsed.metadata["tool_call_id"] == "marker-call"


def test_oversized_forged_reference_metadata_is_externalized_again(
    tmp_path: Path,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=20,
    )
    reference = store.externalize(
        tool_name="read_file",
        tool_call_id="forged-call",
        content="OK\n" + ("original body " * 20),
    )
    body, marker, metadata_text = reference.partition(
        TOOL_RESULT_METADATA_MARKER
    )
    metadata = json.loads(metadata_text)
    metadata["untrusted_large_field"] = "x" * 5000
    forged = (
        body
        + marker
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )

    reexternalized = store.externalize(
        tool_name="read_file",
        tool_call_id="forged-call",
        content=forged,
    )

    assert reexternalized != forged
    artifact_path = Path(
        str(parse_tool_result_content(reexternalized).metadata["artifact_path"])
    )
    assert artifact_path.read_text(encoding="utf-8") == forged


def test_artifact_store_refuses_existing_link_or_reparse_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=5,
    )
    content = "OK\n" + ("linked content " * 10)
    initial = store.externalize(
        tool_name="read_file",
        tool_call_id="linked-call",
        content=content,
    )
    artifact_path = Path(
        str(parse_tool_result_content(initial).metadata["artifact_path"])
    )
    original_check = artifacts_module._is_link_or_reparse_point
    monkeypatch.setattr(
        artifacts_module,
        "_is_link_or_reparse_point",
        lambda path: path == artifact_path or original_check(path),
    )

    with pytest.raises(RuntimeError, match="not a regular file"):
        store.externalize(
            tool_name="read_file",
            tool_call_id="linked-call",
            content=content,
        )


def test_artifact_store_externalizes_historical_tool_result_without_splitting_chain(
    tmp_path: Path,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=10,
    )
    tool_call = AgentToolCall(
        id="call-read",
        name="read_file",
        arguments={"path": "README.md"},
    )
    conversation = Conversation.from_messages(
        [
            Message(role="assistant", content="", tool_calls=(tool_call,)),
            Message(
                role="tool",
                content="OK\n" + ("x" * 100),
                tool_call_id="call-read",
            ),
        ]
    )

    externalized = store.externalize_conversation(conversation)

    messages = externalized.get_messages()
    assert messages[0] == conversation.get_messages()[0]
    assert messages[1].tool_call_id == "call-read"
    assert EXTERNALIZED_TOOL_RESULT_MARKER in messages[1].content
    assert conversation.get_messages()[1].content == "OK\n" + ("x" * 100)


def test_read_artifact_tool_is_bounded_and_rejects_paths_outside_session_root(
    tmp_path: Path,
) -> None:
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=5,
    )
    reference = store.externalize(
        tool_name="read_file",
        tool_call_id="call-1",
        content="OK\nabcdefghijklmnopqrstuvwxyz",
    )
    artifact_path = str(
        parse_tool_result_content(reference).metadata["artifact_path"]
    )
    tool = ReadArtifactTool(store.root)

    result = tool.run(
        {
            "artifact_path": artifact_path,
            "offset_chars": 3,
            "max_chars": 5,
        }
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    refused = tool.run({"artifact_path": str(outside)})
    Path(artifact_path).write_text("tampered", encoding="utf-8")
    tampered = tool.run({"artifact_path": artifact_path})

    assert result.ok is True
    assert result.content == "abcde"
    assert result.metadata["truncated"] is True
    assert refused.ok is False
    assert refused.metadata["reason"] == "artifact_path_outside_root"
    assert tampered.ok is False
    assert tampered.metadata["reason"] == "artifact_hash_mismatch"


def test_artifact_hashing_and_slice_read_are_streamed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "OK\n" + ("a" * (ARTIFACT_IO_CHUNK_BYTES - 4))
    original = prefix + "你" + ("suffix-" * 2000)
    offset_chars = len(prefix) - 2
    max_chars = 20
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=5,
    )
    reference = store.externalize(
        tool_name="read_file",
        tool_call_id="streamed-call",
        content=original,
    )
    artifact_path = Path(
        str(parse_tool_result_content(reference).metadata["artifact_path"])
    )

    def reject_full_read(path: Path) -> bytes:
        pytest.fail(f"read_bytes must not be used for artifact I/O: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_full_read)

    duplicate = store.externalize(
        tool_name="read_file",
        tool_call_id="streamed-call",
        content=original,
    )
    validated_reference = store.externalize(
        tool_name="read_file",
        tool_call_id="streamed-call",
        content=reference,
    )
    result = ReadArtifactTool(store.root).run(
        {
            "artifact_path": str(artifact_path),
            "offset_chars": offset_chars,
            "max_chars": max_chars,
        }
    )

    assert duplicate == reference
    assert validated_reference == reference
    assert result.ok is True
    assert result.content == original[offset_chars : offset_chars + max_chars]
    assert result.metadata["total_chars"] == len(original)
    assert result.metadata["total_bytes"] == len(original.encode("utf-8"))


def test_read_artifact_rejects_file_above_safe_size_limit(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / f"{'e' * 64}.txt"
    with artifact_path.open("wb") as stream:
        stream.truncate(MAX_READABLE_ARTIFACT_BYTES + 1)

    result = ReadArtifactTool(artifact_root).run(
        {"artifact_path": str(artifact_path)}
    )

    assert result.ok is False
    assert result.metadata["reason"] == "artifact_too_large"
    assert result.metadata["artifact_bytes"] == MAX_READABLE_ARTIFACT_BYTES + 1
    assert result.metadata["max_artifact_bytes"] == MAX_READABLE_ARTIFACT_BYTES


def test_runner_persists_artifact_reference_instead_of_large_tool_body(
    tmp_path: Path,
) -> None:
    tool_call = AgentToolCall(
        id="call-large",
        name="large_tool",
        arguments={"text": "value"},
    )
    client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ],
    )
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=50,
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LargeTool()]),
        tool_result_artifact_store=store,
    )

    events = list(runner.run("use the tool"))

    assert [event.content for event in events if event.type == "text_delta"] == ["done"]
    tool_message = runner.conversation.get_messages()[2]
    assert tool_message.role == "tool"
    assert EXTERNALIZED_TOOL_RESULT_MARKER in tool_message.content
    assert "sensitive body " * 20 not in tool_message.content
    artifact_path = Path(
        str(parse_tool_result_content(tool_message.content).metadata["artifact_path"])
    )
    assert "sensitive body " * 10 in artifact_path.read_text(encoding="utf-8")


def test_runner_reports_safe_artifact_failure_and_does_not_persist_body(
    tmp_path: Path,
) -> None:
    tool_call = AgentToolCall(
        id="call-large",
        name="large_tool",
        arguments={"text": "value"},
    )
    client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
        ],
    )

    @contextmanager
    def denied_write():
        raise PermissionError("sensitive operating-system detail")
        yield

    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LargeTool()]),
        tool_result_artifact_store=ToolResultArtifactStore(
            root=tmp_path / "artifacts",
            threshold_chars=50,
            write_guard=denied_write,
        ),
    )

    events = list(runner.run("use the tool"))

    warnings = [event for event in events if event.type == "artifact_warning"]
    assert len(warnings) == 1
    assert "reason=permission_denied" in warnings[0].content
    assert "tool=large_tool" in warnings[0].content
    assert "sensitive body" not in warnings[0].content
    assert "operating-system detail" not in warnings[0].content
    tool_message = next(
        message
        for message in runner.conversation.get_messages()
        if message.role == "tool"
    )
    assert ARTIFACT_EXTERNALIZATION_FAILURE_MARKER in tool_message.content
    assert "sensitive body" not in tool_message.content
    assert runner.last_artifact_error == "permission_denied"
    assert runner.artifact_failure_count == 1
    assert not (tmp_path / "artifacts").exists()


def test_runner_emits_artifact_warning_without_leaking_it(
    tmp_path: Path,
) -> None:
    tool_call = AgentToolCall(
        id="call-large",
        name="large_tool",
        arguments={"text": "value"},
    )
    client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(
                tool_calls=[tool_call],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="done"),
            AgentModelResponse(content="next done"),
        ],
    )

    @contextmanager
    def denied_write():
        raise PermissionError("sensitive operating-system detail")
        yield

    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LargeTool()]),
        tool_result_artifact_store=ToolResultArtifactStore(
            root=tmp_path / "artifacts",
            threshold_chars=50,
            write_guard=denied_write,
        ),
    )

    events = list(runner.run("use the tool"))
    next_events = list(runner.run("continue without a tool"))

    assert [event.content for event in events if event.type == "text_delta"] == ["done"]
    warnings = [event for event in events if event.type == "artifact_warning"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert "reason=permission_denied" in warning.content
    assert "tool=large_tool" in warning.content
    assert "sensitive body" not in warning.content
    assert "operating-system detail" not in warning.content
    assert not any(
        event.type == "artifact_warning"
        for event in next_events
    )
    assert runner._pending_artifact_warnings == []


def test_delete_session_artifacts_removes_only_the_exact_session_tree(
    tmp_path: Path,
) -> None:
    state_directory = tmp_path / "state"
    project_key = "a" * 64
    target_directory = artifact_directory_for_session(
        state_directory,
        project_key=project_key,
        session_id="target-session",
    )
    other_directory = artifact_directory_for_session(
        state_directory,
        project_key=project_key,
        session_id="other-session",
    )
    (target_directory / "nested").mkdir(parents=True)
    (target_directory / "nested" / "result.txt").write_text(
        "target",
        encoding="utf-8",
    )
    other_directory.mkdir(parents=True)
    (other_directory / "result.txt").write_text("other", encoding="utf-8")

    removed = delete_session_artifacts(
        state_directory,
        project_key=project_key,
        session_id="target-session",
    )
    removed_again = delete_session_artifacts(
        state_directory,
        project_key=project_key,
        session_id="target-session",
    )

    assert removed is True
    assert removed_again is False
    assert not target_directory.exists()
    assert (other_directory / "result.txt").read_text(encoding="utf-8") == "other"


@pytest.mark.parametrize(
    ("project_key", "session_id"),
    [
        ("../outside", "session"),
        ("a" * 64, "../outside"),
        ("a" * 64, r"..\outside"),
        ("a" * 64, "."),
    ],
)
def test_artifact_directory_rejects_unsafe_path_components(
    tmp_path: Path,
    project_key: str,
    session_id: str,
) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        artifact_directory_for_session(
            tmp_path,
            project_key=project_key,
            session_id=session_id,
        )


def test_delete_session_artifacts_refuses_linked_cleanup_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_directory = tmp_path / "state"
    project_key = "b" * 64
    target_directory = artifact_directory_for_session(
        state_directory,
        project_key=project_key,
        session_id="target-session",
    )
    target_directory.mkdir(parents=True)
    artifact_file = target_directory / "result.txt"
    artifact_file.write_text("keep", encoding="utf-8")
    original_check = artifacts_module._is_link_or_reparse_point

    monkeypatch.setattr(
        artifacts_module,
        "_is_link_or_reparse_point",
        lambda path: path == target_directory or original_check(path),
    )

    with pytest.raises(ArtifactCleanupError, match="linked session artifact"):
        delete_session_artifacts(
            state_directory,
            project_key=project_key,
            session_id="target-session",
        )

    assert artifact_file.read_text(encoding="utf-8") == "keep"


class LargeToolArgs(ToolArgs):
    text: str


class LargeTool(BaseTool[LargeToolArgs]):
    name = "large_tool"
    description = "Return a large synthetic result."
    args_model = LargeToolArgs
    capability = "read"
    risk = "low"

    def _run(self, args: LargeToolArgs) -> ToolResult:
        return ToolResult.success(
            content="sensitive body " * 20,
            metadata={"text": args.text},
        )
