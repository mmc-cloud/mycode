from contextlib import contextmanager
import hashlib
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
    ContextBudget,
    TOOL_RESULT_METADATA_MARKER,
    parse_tool_result_content,
)
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.runner import AgentRunner
from mycode.session_store import SessionStore
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


def test_runner_separates_canonical_artifact_refs_from_latest_model_view_and_resume(
    tmp_path: Path,
) -> None:
    project = ProjectIdentity.from_workspace(tmp_path)
    session_store = SessionStore(tmp_path / "state.sqlite3")
    session = session_store.create_session(project, session_id="artifact-contract")
    artifact_root = artifact_directory_for_session(
        session_store.database_path.parent,
        project_key=project.key,
        session_id=session.id,
    )
    artifact_store = ToolResultArtifactStore(
        root=artifact_root,
        threshold_chars=50,
    )
    first_call = AgentToolCall(
        id="call-large-1",
        name="large_tool",
        arguments={"text": "first"},
    )
    second_call = AgentToolCall(
        id="call-large-2",
        name="large_tool",
        arguments={"text": "second"},
    )
    client = RecordingArtifactLLMClient(
        [
            AgentModelResponse(tool_calls=[first_call], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[second_call], stop_reason="tool_calls"),
            AgentModelResponse(content="done"),
        ]
    )
    conversation = Conversation.from_messages(
        [],
        on_message_added=lambda message: session_store.append_message(
            project,
            session.id,
            message,
        ),
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LargeTool()]),
        conversation=conversation,
        context_budget=ContextBudget(recent_tool_result_groups_to_keep=1),
        tool_result_artifact_store=artifact_store,
    )

    list(runner.run("produce two large results"))

    canonical = session_store.load_conversation(project, session.id)
    canonical_tool_messages = [
        message for message in canonical.get_messages() if message.role == "tool"
    ]
    assert len(canonical_tool_messages) == 2
    assert all(
        EXTERNALIZED_TOOL_RESULT_MARKER in message.content
        for message in canonical_tool_messages
    ), [message.content for message in canonical_tool_messages]
    assert all(
        "sensitive body " * 20 not in message.content
        for message in canonical_tool_messages
    )

    artifact_contents: list[str] = []
    artifact_paths: list[Path] = []
    for message in canonical_tool_messages:
        parsed = parse_tool_result_content(message.content)
        artifact_path = Path(str(parsed.metadata["artifact_path"]))
        artifact_content = artifact_path.read_text(encoding="utf-8")
        artifact_paths.append(artifact_path)
        artifact_contents.append(artifact_content)
        assert artifact_path.is_file()
        assert "sensitive body " * 20 in artifact_content
        assert len(artifact_content) == parsed.metadata["original_chars"]
        assert hashlib.sha256(artifact_content.encode("utf-8")).hexdigest() == (
            parsed.metadata["artifact_sha256"]
        )

    first_next_turn = client.seen_conversations[1].get_messages()
    first_visible = next(
        message
        for message in first_next_turn
        if message.tool_call_id == "call-large-1"
    )
    assert first_visible.content == artifact_contents[0]
    assert EXTERNALIZED_TOOL_RESULT_MARKER not in first_visible.content

    second_next_turn = client.seen_conversations[2].get_messages()
    old_visible = next(
        message
        for message in second_next_turn
        if message.tool_call_id == "call-large-1"
    )
    latest_visible = next(
        message
        for message in second_next_turn
        if message.tool_call_id == "call-large-2"
    )
    assert old_visible.content == canonical_tool_messages[0].content
    assert EXTERNALIZED_TOOL_RESULT_MARKER in old_visible.content
    assert latest_visible.content == artifact_contents[1]
    assert EXTERNALIZED_TOOL_RESULT_MARKER not in latest_visible.content

    resume_client = RecordingArtifactLLMClient(
        [
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call-read-old",
                        name="read_artifact",
                        arguments={
                            "artifact_path": artifact_paths[0].as_posix(),
                            "offset_chars": 0,
                            "max_chars": 3000,
                        },
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="resumed"),
        ]
    )
    resumed_conversation = Conversation.from_messages(
        canonical.get_messages(),
        on_message_added=lambda message: session_store.append_message(
            project,
            session.id,
            message,
        ),
    )
    resumed_runner = AgentRunner(
        llm_client=resume_client,
        tool_registry=ToolRegistry.from_tools([ReadArtifactTool(artifact_root)]),
        conversation=resumed_conversation,
        tool_result_artifact_store=artifact_store,
    )

    resume_events = list(resumed_runner.run("continue from persisted evidence"))

    resume_first_context = resume_client.seen_conversations[0].get_messages()
    resumed_old_result = next(
        message
        for message in resume_first_context
        if message.tool_call_id == "call-large-1"
    )
    assert EXTERNALIZED_TOOL_RESULT_MARKER in resumed_old_result.content
    assert "sensitive body " * 20 not in resumed_old_result.content
    recovered = next(
        event.tool_result
        for event in resume_events
        if event.type == "tool_result" and event.tool_result is not None
    )
    assert recovered.ok is True
    assert recovered.content == artifact_contents[0]


def test_runner_keeps_two_ephemeral_tool_result_groups_without_changing_canonical(
    tmp_path: Path,
) -> None:
    calls = [
        AgentToolCall(
            id=call_id,
            name="large_tool",
            arguments={"text": call_id},
        )
        for call_id in ("call-old", "call-middle-a", "call-middle-b", "call-latest")
    ]
    client = RecordingArtifactLLMClient(
        [
            AgentModelResponse(tool_calls=[calls[0]], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=calls[1:3], stop_reason="tool_calls"),
            AgentModelResponse(tool_calls=[calls[3]], stop_reason="tool_calls"),
            AgentModelResponse(content="done"),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LargeTool()]),
        context_budget=ContextBudget(recent_tool_result_groups_to_keep=2),
        tool_result_artifact_store=ToolResultArtifactStore(
            root=tmp_path / "artifacts",
            threshold_chars=50,
        ),
    )

    list(runner.run("produce three result groups"))

    canonical_results = {
        message.tool_call_id: message
        for message in runner.conversation.get_messages()
        if message.role == "tool"
    }
    assert set(canonical_results) == {call.id for call in calls}
    assert all(
        EXTERNALIZED_TOOL_RESULT_MARKER in message.content
        for message in canonical_results.values()
    )
    assert all(
        "sensitive body " * 20 not in message.content
        for message in canonical_results.values()
    )

    third_next_turn = {
        message.tool_call_id: message
        for message in client.seen_conversations[3].get_messages()
        if message.role == "tool"
    }
    assert EXTERNALIZED_TOOL_RESULT_MARKER in third_next_turn["call-old"].content
    for call_id in ("call-middle-a", "call-middle-b", "call-latest"):
        assert EXTERNALIZED_TOOL_RESULT_MARKER not in third_next_turn[call_id].content
        assert "sensitive body " * 20 in third_next_turn[call_id].content

    assert [set(group) for group in runner._ephemeral_tool_result_groups] == [
        {"call-middle-a", "call-middle-b"},
        {"call-latest"},
    ]


def test_runner_falls_back_to_artifact_ref_when_latest_body_exceeds_budget(
    tmp_path: Path,
) -> None:
    tool_call = AgentToolCall(
        id="call-huge",
        name="huge_tool",
        arguments={"text": "value"},
    )
    client = RecordingArtifactLLMClient(
        [
            AgentModelResponse(tool_calls=[tool_call], stop_reason="tool_calls"),
            AgentModelResponse(content="done"),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([HugeTool()]),
        context_budget=ContextBudget(
            context_window_tokens=3000,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
            tool_result_compression_threshold_chars=50,
            recent_tool_result_groups_to_keep=2,
        ),
        tool_result_artifact_store=ToolResultArtifactStore(
            root=tmp_path / "artifacts",
            threshold_chars=50,
        ),
    )

    list(runner.run("produce an oversized result"))

    next_turn_tool_message = next(
        message
        for message in client.seen_conversations[1].get_messages()
        if message.tool_call_id == "call-huge"
    )
    assert EXTERNALIZED_TOOL_RESULT_MARKER in next_turn_tool_message.content
    assert "huge body " * 100 not in next_turn_tool_message.content
    assert runner._ephemeral_tool_result_groups == []


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


class HugeTool(LargeTool):
    name = "huge_tool"
    description = "Return a result too large for the active model budget."

    def _run(self, args: LargeToolArgs) -> ToolResult:
        return ToolResult.success(
            content="huge body " * 3000,
            metadata={"text": args.text},
        )


class RecordingArtifactLLMClient(FakeLLMClient):
    def __init__(self, tool_responses: list[AgentModelResponse]) -> None:
        super().__init__(responses=[], tool_responses=tool_responses)
        self.seen_conversations: list[Conversation] = []

    def stream_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, object]],
    ):
        self.seen_conversations.append(
            Conversation.from_messages(conversation.get_messages())
        )
        yield from super().stream_with_tools(conversation, tools)
