from pathlib import Path

import pytest

from mycode.agent import AgentToolCall
from mycode.artifacts import (
    EXTERNALIZED_TOOL_RESULT_MARKER, ToolResultArtifactStore, artifact_reference_info,
)
from mycode.context_budget import ContextBudget, MemoryContextStats, TokenEstimator, estimate_conversation
from mycode.context_builder import ContextBuilder
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.tool_result_format import COMPRESSED_TOOL_RESULT_MARKER, parse_tool_result_content
from mycode.tool_result_retention import ToolResultRetentionPolicy, TurnLocalFullGroup


def _history(store, *, batches=1, width=3, body_size=12000):
    messages = [Message(role="system", content="system"), Message(role="user", content="current request")]
    originals = {}
    for batch in range(batches):
        calls = tuple(AgentToolCall(id=f"b{batch}-{i}", name="read_file", arguments={"path": str(i)})
                      for i in range(width))
        messages.append(Message(role="assistant", content="", tool_calls=calls))
        for call in calls:
            full = f"OK\n{call.id}: " + "x" * body_size
            originals[call.id] = full
            reference = store.externalize(tool_name=call.name, tool_call_id=call.id, content=full)
            messages.append(Message(role="tool", content=reference, tool_call_id=call.id))
    return Conversation.from_messages(messages), originals


def _budget(tokens=100000, keep=1):
    return ContextBudget(context_window_tokens=tokens, reserved_output_tokens=0,
                         safety_margin_tokens=0, recent_tool_result_groups_to_keep=keep)


@pytest.mark.parametrize("keep", [0, 1, 2])
def test_projection_recency_is_by_batch_and_never_mutates_canonical(tmp_path, keep):
    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, originals = _history(store, batches=3)
    canonical = history.get_messages()
    context = ContextBuilder(_budget(keep=keep), TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(_budget(keep=keep), store)).build(history).context
    for message in context.messages:
        if message.role == "tool":
            batch = int(message.tool_call_id[1])
            if batch >= 3 - keep:
                assert message.content == originals[message.tool_call_id]
            else:
                assert EXTERNALIZED_TOOL_RESULT_MARKER in message.content
    assert context.retention_stats.full_groups == keep
    assert context.retention_stats.artifact_groups == 3 - keep
    assert history.get_messages() == canonical


@pytest.mark.parametrize("reuse", [False, True])
def test_budget_downgrades_entire_batch_with_memory_guidance_and_schemas(tmp_path, monkeypatch, reuse):
    import mycode.context_builder as module

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, originals = _history(store)
    messages = history.get_messages()
    handoff = TurnLocalFullGroup(messages[2], {
        m.tool_call_id: (m.content, originals[m.tool_call_id])
        for m in messages if m.role == "tool"
    })
    rehydrates = []
    real_rehydrate = ToolResultArtifactStore.rehydrate

    def rehydrate(store, **kwargs):
        rehydrates.append(kwargs["tool_call_id"])
        return real_rehydrate(store, **kwargs)

    monkeypatch.setattr(ToolResultArtifactStore, "rehydrate", rehydrate)
    memory = Message(role="system", content="remember the requirements")
    guidance = "verify the requested change"
    tools = [{"name": "read_file", "parameters": {"type": "object"}}]
    assembled_refs = Conversation.from_messages([
        *history.get_messages(), memory, Message(role="system", content=guidance),
    ])
    budget = _budget(estimate_conversation(assembled_refs, tools=tools).estimated_input_tokens + 5)
    real_budget = module.budget_model_context
    requests = []

    def budget_once(messages, *args, **kwargs):
        requests.append(messages)
        assert all(EXTERNALIZED_TOOL_RESULT_MARKER not in m.content for m in messages if m.role == "tool")
        return real_budget(messages, *args, **kwargs)

    monkeypatch.setattr(module, "budget_model_context", budget_once)
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget, store)).build(
        history, memory_message=memory, memory_stats=MemoryContextStats(selected_entry_count=1),
        guidance=(guidance,), tools=tools,
        turn_local_full_group=handoff if reuse else None,
    ).context
    assert len(requests) == 1
    assert len(rehydrates) == (0 if reuse else 3)
    results = [m for m in context.messages if m.role == "tool"]
    assert len(results) == 3
    assert all(EXTERNALIZED_TOOL_RESULT_MARKER in m.content for m in results)
    assert context.retention_stats.budget_downgraded_groups == 1
    assert context.retention_stats.artifact_groups == 1
    assert context.retention_stats.turn_local_full_groups == (1 if reuse else 0)
    assert context.retention_stats.artifact_rehydrated_groups == (0 if reuse else 1)
    assert context.retention_stats.artifact_rehydrate_count == (0 if reuse else 3)
    assert context.memory_stats.included_entry_count == 1
    assert memory in context.messages
    assert any(m.content == guidance for m in context.messages)
    assert any(m.content == "current request" for m in context.messages)
    assert context.estimate.tool_schema_chars > 0
    assert not context.estimate.over_budget


def test_budget_compresses_older_reference_batch_before_latest(tmp_path):
    from mycode.tool_result_format import _group_non_system_messages, _flatten_groups

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, originals = _history(store, batches=2, width=2)
    policy = ToolResultRetentionPolicy(_budget(keep=0), store)
    groups = _group_non_system_messages(tuple(history.get_messages()))
    index, compressed = next(policy.metadata_candidates(groups))
    groups[index] = compressed
    candidate = Conversation.from_messages([history.get_messages()[0], *_flatten_groups(groups)])
    budget = _budget(estimate_conversation(candidate).estimated_input_tokens + 1, keep=0)
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget, store)).build(history).context
    results = [m for m in context.messages if m.role == "tool"]
    assert len(results) == 4
    for message in results[:2]:
        assert COMPRESSED_TOOL_RESULT_MARKER in message.content
        parsed = parse_tool_result_content(message.content)
        assert parsed.status == "OK"
        assert parsed.metadata["original_chars"] == len(originals[message.tool_call_id])
        assert Path(parsed.metadata["artifact_path"]).is_file()
        assert len(parsed.metadata["artifact_sha256"]) == 64
        assert "result_preview" in parsed.metadata
    assert all(EXTERNALIZED_TOOL_RESULT_MARKER in m.content for m in results[2:])
    assert context.retention_stats.metadata_groups == 1
    assert context.retention_stats.artifact_groups == 1
    assert not context.estimate.over_budget


@pytest.mark.parametrize("failure", ["missing", "hash", "size", "utf8", "foreign", "reparse", "malformed"])
def test_rehydrate_failure_keeps_whole_batch_and_canonical_refs(tmp_path, monkeypatch, failure):
    import mycode.artifacts as module

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, _ = _history(store, width=2)
    canonical = history.get_messages()
    reference = canonical[-1]
    path, _, _ = artifact_reference_info(tool_name="read_file", tool_call_id=reference.tool_call_id,
                                         content=reference.content)
    if failure == "malformed":
        from dataclasses import replace
        canonical[-1] = replace(reference, content=reference.content.replace('"original_chars":', '"wrong_chars":'))
        history = Conversation.from_messages(canonical)
    elif failure == "missing":
        path.unlink()
    elif failure == "hash":
        path.write_text("tampered", encoding="utf-8")
    elif failure == "size":
        monkeypatch.setattr(module, "MAX_READABLE_ARTIFACT_BYTES", 10)
    elif failure == "utf8":
        path.write_bytes(b"\xff")
    elif failure == "foreign":
        store = ToolResultArtifactStore(tmp_path / "different-root", 50)
    elif failure == "reparse":
        real_check = module._is_link_or_reparse_point
        monkeypatch.setattr(module, "_is_link_or_reparse_point",
                            lambda p: p == path or real_check(p))
    context = ContextBuilder(_budget(), TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(_budget(), store)).build(history).context
    assert context.retention_stats.rehydration_failures >= 1
    assert context.retention_stats.artifact_groups == 1
    assert context.messages == tuple(canonical)
    assert history.get_messages() == canonical
    assert not context.estimate.over_budget


def test_rehydrate_validates_reference_fields_before_reading(tmp_path):
    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, originals = _history(store, width=1)
    reference = history.get_messages()[-1]
    assert store.rehydrate(tool_name="read_file", tool_call_id="b0-0",
                           content=reference.content) == originals["b0-0"]
    with pytest.raises(ValueError, match="Invalid artifact reference"):
        store.rehydrate(tool_name="read_file", tool_call_id="wrong-id", content=reference.content)
    with pytest.raises(ValueError, match="Invalid artifact reference"):
        store.rehydrate(tool_name="read_file", tool_call_id="b0-0",
                        content=reference.content.replace('"original_chars":', '"wrong_chars":'))


def test_no_artifact_budget_omits_memory_then_trims_history_but_preserves_user():
    current = Message(role="user", content="CURRENT")
    call = AgentToolCall(id="large", name="read_file", arguments={"path": "file"})
    history = Conversation.from_messages([
        Message(role="system", content="SYSTEM"),
        Message(role="user", content="OLD" * 4000), current,
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(role="tool", content="OK\n" + "x" * 20000, tool_call_id="large"),
    ])
    memory = Message(role="system", content="memory" * 4000)
    budget = _budget(700)
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget)).build(
        history, memory_message=memory, memory_stats=MemoryContextStats(selected_entry_count=1),
        guidance=("GUIDANCE",),
    ).context
    assert current in context.messages
    assert any(m.content == "SYSTEM" for m in context.messages)
    assert any(m.content == "GUIDANCE" for m in context.messages)
    assert context.retention_stats.metadata_groups == 1
    assert context.memory_stats.included_entry_count == 0
    assert memory not in context.messages
    assert not any(m.content.startswith("OLD") for m in context.messages)
    assert not context.estimate.over_budget
    oversized = Conversation.from_messages([Message(role="user", content="request" * 4000)])
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget)).build(oversized).context
    assert context.estimate.over_budget
    assert context.messages == tuple(oversized.get_messages())

    checkpoint = Message(role="user", content="summarize progress for max-turn stop")
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget)).build(
        oversized, request_messages=(checkpoint,),
    ).context
    assert context.estimate.over_budget
    assert context.messages == (*oversized.get_messages(), checkpoint)


def test_reused_tool_call_ids_do_not_cross_batch_boundaries(tmp_path):
    store = ToolResultArtifactStore(tmp_path / "a", 50)
    call = AgentToolCall(id="same-id", name="read_file", arguments={})
    messages = [Message(role="user", content="current")]
    for body in ("old body " * 1000, "new body " * 1000):
        messages.extend([
            Message(role="assistant", content="", tool_calls=(call,)),
            Message(role="tool", tool_call_id=call.id, content=store.externalize(
                tool_name=call.name, tool_call_id=call.id, content="OK\n" + body,
            )),
        ])
    history = Conversation.from_messages(messages)
    context = ContextBuilder(_budget(), TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(_budget(), store)).build(history).context
    assert context.messages[2] == messages[2]
    assert EXTERNALIZED_TOOL_RESULT_MARKER in context.messages[2].content
    assert context.messages[4].content == "OK\n" + "new body " * 1000


def test_rehydrate_byte_limit_also_bounds_multibyte_content(tmp_path, monkeypatch):
    import mycode.artifacts as module

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    body = "OK\n" + "汉字" * 100
    ref = store.externalize(tool_name="read_file", tool_call_id="unicode", content=body)
    monkeypatch.setattr(module, "MAX_READABLE_ARTIFACT_BYTES", len(body) + 1)
    with pytest.raises(RuntimeError):
        store.rehydrate(tool_name="read_file", tool_call_id="unicode", content=ref)


@pytest.mark.parametrize("mismatch", ["older", "identity", "reference", "keep_zero"])
def test_turn_local_reuse_requires_the_latest_matching_batch(tmp_path, monkeypatch, mismatch):
    from dataclasses import replace

    store = ToolResultArtifactStore(tmp_path / "a", 50)
    history, originals = _history(store, batches=2, width=1)
    messages = history.get_messages()
    assistant, result = messages[-2:]
    if mismatch == "older":
        assistant, result = messages[2:4]
    elif mismatch == "identity":
        assistant = replace(assistant)
    reference = "different reference" if mismatch == "reference" else result.content
    handoff = TurnLocalFullGroup(assistant, {result.tool_call_id: (reference, "wrong stale full")})
    calls = []
    real_rehydrate = ToolResultArtifactStore.rehydrate

    def rehydrate(store, **kwargs):
        calls.append(kwargs["tool_call_id"])
        return real_rehydrate(store, **kwargs)

    monkeypatch.setattr(ToolResultArtifactStore, "rehydrate", rehydrate)
    budget = _budget(keep=0 if mismatch == "keep_zero" else 2)
    context = ContextBuilder(budget, TokenEstimator(),
                             retention_policy=ToolResultRetentionPolicy(budget, store)).build(
        history, turn_local_full_group=handoff,
    ).context
    assert calls == ([] if mismatch == "keep_zero" else ["b0-0", "b1-0"])
    for message in context.messages:
        if message.role == "tool":
            if mismatch == "keep_zero":
                assert EXTERNALIZED_TOOL_RESULT_MARKER in message.content
            else:
                assert message.content == originals[message.tool_call_id]
