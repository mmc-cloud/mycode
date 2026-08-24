from datetime import UTC, datetime
from pathlib import Path

from mycode.context_budget import MemoryContextStats
from mycode.instructions import (
    InstructionBundle,
    InstructionIssue,
    InstructionSource,
)
from mycode.memory import MemoryEntry
from mycode.memory_context import MemoryRecall, MemoryRecallSource
from mycode.messages import Message
from mycode.subagents.snapshots import (
    FrozenMemoryRecallProvider,
    create_runtime_context_snapshot,
)


LOADED_AT = datetime(2026, 7, 13, 8, 30, tzinfo=UTC)


def test_runtime_snapshot_records_content_free_instruction_fingerprints(
    tmp_path,
) -> None:
    source_path = tmp_path / "AGENTS.md"
    bundle = InstructionBundle(
        sources=(
            InstructionSource(
                path=source_path,
                scope="project",
                content="private project rule",
            ),
        ),
        issues=(
            InstructionIssue(
                path=tmp_path / "nested" / "AGENTS.md",
                code="invalid_utf8",
                message="instruction file must be valid UTF-8",
            ),
        ),
    )

    snapshot = create_runtime_context_snapshot(
        bundle,
        memory_recall=None,
        loaded_at=LOADED_AT,
    )

    metadata = snapshot.metadata.instructions
    assert metadata.loaded_at == LOADED_AT
    assert metadata.total_chars == len("private project rule")
    assert metadata.sources[0].path == source_path
    assert metadata.sources[0].content_bytes == len("private project rule")
    assert len(metadata.sources[0].sha256) == 64
    assert metadata.warnings[0].code == "invalid_utf8"
    assert snapshot.project_instructions.endswith("private project rule")
    assert "private project rule" not in repr(snapshot.metadata)
    assert "private project rule" not in repr(snapshot)


def test_runtime_snapshot_fingerprints_memory_without_exposing_entry_body(
    tmp_path,
) -> None:
    memory_message = Message(role="system", content="private recalled memory")
    recall = MemoryRecall(
        message=memory_message,
        entries=(
            MemoryEntry(
                scope="project",
                kind="fact",
                key="test.command",
                content="private recalled memory",
            ),
        ),
        stats=MemoryContextStats(
            safe_entry_count=1,
            relevant_entry_count=1,
            selected_entry_count=1,
            estimated_tokens=10,
            scopes=("project",),
        ),
        sources=(
            MemoryRecallSource(
                scope="project",
                path=tmp_path / "MEMORY.md",
                content_chars=80,
                sha256="a" * 64,
            ),
        ),
    )

    snapshot = create_runtime_context_snapshot(
        InstructionBundle(),
        memory_recall=recall,
        loaded_at=LOADED_AT,
    )

    metadata = snapshot.metadata.memory
    assert metadata.enabled is True
    assert metadata.sources[0].sha256 == "a" * 64
    assert metadata.selected_entries[0].content_chars == len(
        "private recalled memory"
    )
    assert len(metadata.selected_entries[0].sha256) == 64
    assert "private recalled memory" not in repr(metadata)
    assert FrozenMemoryRecallProvider(recall).recall("changed query") is recall


def test_runtime_snapshot_hash_changes_when_instruction_content_changes() -> None:
    first = create_runtime_context_snapshot(
        InstructionBundle(
            sources=(
                InstructionSource(
                    path=Path(__file__),
                    scope="project",
                    content="version one",
                ),
            )
        ),
        memory_recall=None,
        loaded_at=LOADED_AT,
    )
    second = create_runtime_context_snapshot(
        InstructionBundle(
            sources=(
                InstructionSource(
                    path=Path(__file__),
                    scope="project",
                    content="version two",
                ),
            )
        ),
        memory_recall=None,
        loaded_at=LOADED_AT,
    )

    assert first.metadata.instructions.combined_sha256 != (
        second.metadata.instructions.combined_sha256
    )
    assert first.metadata.combined_sha256 != second.metadata.combined_sha256
