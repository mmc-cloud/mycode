from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path

from mycode.context_budget import MemoryContextStats
from mycode.instructions import InstructionBundle, InstructionIssueCode, InstructionScope
from mycode.memory import MemoryKind, MemoryScope
from mycode.memory_context import MemoryRecall


@dataclass(frozen=True)
class InstructionSourceFingerprint:
    scope: InstructionScope
    path: Path
    content_bytes: int
    content_chars: int
    sha256: str


@dataclass(frozen=True)
class InstructionWarningSnapshot:
    path: Path
    code: InstructionIssueCode
    message: str


@dataclass(frozen=True)
class InstructionSnapshotMetadata:
    loaded_at: datetime
    total_chars: int
    combined_sha256: str
    sources: tuple[InstructionSourceFingerprint, ...] = ()
    warnings: tuple[InstructionWarningSnapshot, ...] = ()


@dataclass(frozen=True)
class MemorySourceFingerprint:
    scope: MemoryScope
    path: Path
    content_bytes: int
    content_chars: int
    sha256: str


@dataclass(frozen=True)
class MemoryEntryFingerprint:
    scope: MemoryScope
    kind: MemoryKind
    content_chars: int
    sha256: str


@dataclass(frozen=True)
class MemorySnapshotMetadata:
    enabled: bool
    loaded_at: datetime
    combined_sha256: str
    stats: MemoryContextStats | None = None
    sources: tuple[MemorySourceFingerprint, ...] = ()
    selected_entries: tuple[MemoryEntryFingerprint, ...] = ()


@dataclass(frozen=True)
class SubAgentSnapshotMetadata:
    loaded_at: datetime
    combined_sha256: str
    instructions: InstructionSnapshotMetadata
    memory: MemorySnapshotMetadata


@dataclass(frozen=True)
class RuntimeContextSnapshot:
    """Immutable run-local context; bodies must not be persisted by default."""

    metadata: SubAgentSnapshotMetadata
    project_instructions: str = field(repr=False, compare=False)
    memory_recall: MemoryRecall | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FrozenMemoryRecallProvider:
    recall_snapshot: MemoryRecall

    def recall(self, user_request: str) -> MemoryRecall:
        return self.recall_snapshot


def create_runtime_context_snapshot(
    instruction_bundle: InstructionBundle,
    *,
    memory_recall: MemoryRecall | None,
    loaded_at: datetime,
) -> RuntimeContextSnapshot:
    project_instructions = instruction_bundle.to_prompt_text()
    instruction_metadata = InstructionSnapshotMetadata(
        loaded_at=loaded_at,
        total_chars=instruction_bundle.total_chars,
        combined_sha256=_sha256_text(project_instructions),
        sources=tuple(
            InstructionSourceFingerprint(
                scope=source.scope,
                path=source.path,
                content_bytes=(
                    source.content_bytes
                    if source.content_bytes > 0
                    else len(source.content.encode("utf-8"))
                ),
                content_chars=len(source.content),
                sha256=source.sha256 or _sha256_text(source.content),
            )
            for source in instruction_bundle.sources
        ),
        warnings=tuple(
            InstructionWarningSnapshot(
                path=issue.path,
                code=issue.code,
                message=issue.message,
            )
            for issue in instruction_bundle.issues
        ),
    )
    memory_metadata = _memory_snapshot_metadata(memory_recall, loaded_at=loaded_at)
    combined_sha256 = _sha256_json(
        {
            "instructions": instruction_metadata.combined_sha256,
            "memory": memory_metadata.combined_sha256,
            "memory_enabled": memory_metadata.enabled,
        }
    )
    return RuntimeContextSnapshot(
        metadata=SubAgentSnapshotMetadata(
            loaded_at=loaded_at,
            combined_sha256=combined_sha256,
            instructions=instruction_metadata,
            memory=memory_metadata,
        ),
        project_instructions=project_instructions,
        memory_recall=memory_recall,
    )


def _memory_snapshot_metadata(
    recall: MemoryRecall | None,
    *,
    loaded_at: datetime,
) -> MemorySnapshotMetadata:
    if recall is None:
        return MemorySnapshotMetadata(
            enabled=False,
            loaded_at=loaded_at,
            combined_sha256=_sha256_text(""),
        )

    selected_entries = tuple(
        MemoryEntryFingerprint(
            scope=entry.scope,
            kind=entry.kind,
            content_chars=len(entry.content),
            sha256=_sha256_json(
                {
                    "scope": entry.scope,
                    "kind": entry.kind,
                    "key": entry.key,
                    "content": entry.content,
                }
            ),
        )
        for entry in recall.entries
    )
    sources = tuple(
        MemorySourceFingerprint(
            scope=source.scope,
            path=source.path,
            content_bytes=source.content_bytes,
            content_chars=source.content_chars,
            sha256=source.sha256,
        )
        for source in recall.sources
    )
    message_content = "" if recall.message is None else recall.message.content
    return MemorySnapshotMetadata(
        enabled=True,
        loaded_at=loaded_at,
        combined_sha256=_sha256_json(
            {
                "message": message_content,
                "sources": [source.sha256 for source in sources],
                "entries": [entry.sha256 for entry in selected_entries],
            }
        ),
        stats=recall.stats,
        sources=sources,
        selected_entries=selected_entries,
    )


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
