"""Request-local tool result projection and atomic precision degradation."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from mycode.artifacts import (
    ArtifactExternalizationFailureHandler,
    EXTERNALIZED_TOOL_RESULT_MARKER,
    ToolResultArtifactStore,
    artifact_externalization_failure_content,
    artifact_failure_reason,
    artifact_reference_info,
)
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.tool_result_format import (
    COMPRESSED_TOOL_RESULT_MARKER,
    _compress_tool_result,
    _group_non_system_messages,
    _group_has_tool_result,
    _tool_names_by_id,
)

if TYPE_CHECKING:
    from mycode.context_budget import ContextBudget

ToolGroup = tuple[Message, ...]


@dataclass(frozen=True)
class TurnLocalFullGroup:
    """One freshly persisted batch, handed off to the next context build only."""

    assistant: Message
    # call id -> (persisted reference, original full content); no disk validation
    # is needed for these originals supplied by the just-completed execution.
    results: dict[str, tuple[str, str]]
    externalized_count: int = 0

    def content_for(self, assistant: Message, result: Message) -> str | None:
        if assistant is not self.assistant:
            return None
        entry = self.results.get(result.tool_call_id or "")
        if entry is None or entry[0] != result.content:
            return None
        return entry[1]


@dataclass(frozen=True)
class ToolResultRetentionStats:
    full_groups: int = 0
    artifact_groups: int = 0
    metadata_groups: int = 0
    budget_downgraded_groups: int = 0
    rehydration_failures: int = 0
    turn_local_full_groups: int = 0
    artifact_rehydrated_groups: int = 0
    artifact_rehydrate_count: int = 0
    artifact_externalized_count: int = 0


@dataclass
class RetentionProjection:
    conversation: Conversation
    # Assistant object identity survives Compact's raw tail; call ids may repeat.
    # Alternatives contain canonical refs, never a second cache of full payloads.
    artifact_groups: dict[int, ToolGroup] = field(default_factory=dict)
    rehydration_failures: int = 0
    turn_local_full_groups: int = 0
    artifact_rehydrated_groups: int = 0
    artifact_rehydrate_count: int = 0
    artifact_externalized_count: int = 0


@dataclass(frozen=True)
class ToolResultRetentionPolicy:
    budget: "ContextBudget"
    artifact_store: ToolResultArtifactStore | None = None
    on_externalization_failure: ArtifactExternalizationFailureHandler | None = None

    def project(
        self, conversation: Conversation, *,
        turn_local_full_group: TurnLocalFullGroup | None = None,
    ) -> RetentionProjection:
        projection = RetentionProjection(
            conversation,
            artifact_externalized_count=(
                0
                if turn_local_full_group is None
                else turn_local_full_group.externalized_count
            ),
        )
        groups = [
            group for group in _group_non_system_messages(tuple(conversation.get_messages()))
            if _group_has_tool_result(group)
        ]
        keep = self.budget.recent_tool_result_groups_to_keep
        recent = {id(g[0]) for g in groups[-keep:]} if keep else set()
        replacements: dict[int, Message] = {}
        for group in groups:
            names = _tool_names_by_id(group)
            canonical_messages = []
            for message in group:
                canonical_message = self._canonical(message, names)
                if (
                    canonical_message.content != message.content
                    and canonical_message.role == "tool"
                    and artifact_reference_info(
                        tool_name=names.get(
                            canonical_message.tool_call_id or "",
                            "unknown",
                        ),
                        tool_call_id=canonical_message.tool_call_id,
                        content=canonical_message.content,
                    )
                    is not None
                ):
                    projection.artifact_externalized_count += 1
                canonical_messages.append(canonical_message)
            canonical = tuple(canonical_messages)
            projected = canonical
            if self.artifact_store is not None and id(group[0]) in recent:
                restored = []
                failures = 0
                turn_local_hits = 0
                rehydrate_attempts = 0
                for message in canonical:
                    if message.role != "tool" or not self._is_reference(message, names):
                        restored.append(message)
                        continue
                    try:
                        content = (
                            turn_local_full_group.content_for(group[0], message)
                            if turn_local_full_group is not None and group is groups[-1]
                            else None
                        )
                        if content is None:
                            rehydrate_attempts += 1
                            projection.artifact_rehydrate_count += 1
                            content = self.artifact_store.rehydrate(
                                tool_name=names[message.tool_call_id],
                                tool_call_id=message.tool_call_id,
                                content=message.content,
                            )
                        else:
                            turn_local_hits += 1
                        restored.append(replace(message, content=content))
                    except Exception:
                        failures += 1
                projection.rehydration_failures += failures
                # Failure also applies atomically: don't mix successfully restored
                # siblings with an unavailable reference in the same batch.
                if not failures:
                    projected = tuple(restored)
                    if projected != canonical and turn_local_hits:
                        projection.turn_local_full_groups += 1
                    if projected != canonical and rehydrate_attempts:
                        projection.artifact_rehydrated_groups += 1
            if projected != canonical:
                projection.artifact_groups[id(group[0])] = canonical
            replacements.update({id(old): new for old, new in zip(group, projected)})
        messages = [replacements.get(id(m), m) for m in conversation.get_messages()]
        if any(a is not b for a, b in zip(messages, conversation.get_messages())):
            projection.conversation = Conversation.from_messages(messages)
        return projection

    def _canonical(self, message: Message, names: dict[str, str]) -> Message:
        if message.role != "tool" or self.artifact_store is None:
            return message
        # Structural recognition is independent of availability. A missing file
        # must not turn its reference into a newly externalized artifact-of-ref.
        if self._is_reference(message, names):
            return message
        name = names.get(message.tool_call_id or "", "unknown")
        try:
            content = self.artifact_store.externalize(
                tool_name=name, tool_call_id=message.tool_call_id, content=message.content,
            )
        except Exception as error:
            if self.on_externalization_failure is not None:
                content = self.on_externalization_failure(
                    name, message.tool_call_id, message.content, error,
                )
            else:
                content = artifact_externalization_failure_content(
                    tool_name=name, tool_call_id=message.tool_call_id,
                    original_content=message.content, reason=artifact_failure_reason(error),
                )
        return message if content == message.content else replace(message, content=content)

    @staticmethod
    def _is_reference(message: Message, names: dict[str, str]) -> bool:
        if artifact_reference_info(
            tool_name=names.get(message.tool_call_id or "", "unknown"),
            tool_call_id=message.tool_call_id, content=message.content,
        ) is not None:
            return True
        # Damaged references also stay references: rehydrate rejects them and
        # the whole batch falls back instead of externalizing the ref text.
        lines = message.content.split("\n", 2)
        return len(lines) > 1 and lines[1] == EXTERNALIZED_TOOL_RESULT_MARKER

    def artifact_candidates(
        self, groups: list[ToolGroup], projection: RetentionProjection,
    ) -> Iterator[tuple[int, ToolGroup]]:
        for index, group in enumerate(groups):
            alternative = projection.artifact_groups.get(id(group[0]))
            if alternative is not None and alternative != group:
                yield index, alternative

    def metadata_candidates(
        self, groups: list[ToolGroup], *, include_recent: bool = True,
        projection: RetentionProjection | None = None,
    ) -> Iterator[tuple[int, ToolGroup]]:
        indexes = [i for i, group in enumerate(groups) if _group_has_tool_result(group)]
        keep = self.budget.recent_tool_result_groups_to_keep
        if not include_recent and keep:
            indexes = indexes[:-keep]
        for index in indexes:
            group = groups[index]
            # A ref may cost more than a small restored Full and be skipped by
            # Budget. Still retain its locator when producing Metadata.
            source = (
                group if projection is None
                else projection.artifact_groups.get(id(group[0]), group)
            )
            names = _tool_names_by_id(group)
            threshold = (
                0 if include_recent else self.budget.tool_result_compression_threshold_chars
            )
            compressed = tuple(
                _compress_tool_result(message, threshold, names) for message in source
            )
            if compressed != group:
                yield index, compressed

    @staticmethod
    def stats(
        groups: list[ToolGroup], projection: RetentionProjection, downgraded: set[int],
    ) -> ToolResultRetentionStats:
        counts = {"full": 0, "artifact": 0, "metadata": 0}
        for group in groups:
            results = [m.content for m in group if m.role == "tool"]
            if not results:
                continue
            if any(COMPRESSED_TOOL_RESULT_MARKER in content for content in results):
                counts["metadata"] += 1
            elif any(EXTERNALIZED_TOOL_RESULT_MARKER in content for content in results):
                counts["artifact"] += 1
            else:
                counts["full"] += 1
        return ToolResultRetentionStats(
            full_groups=counts["full"], artifact_groups=counts["artifact"],
            metadata_groups=counts["metadata"], budget_downgraded_groups=len(downgraded),
            rehydration_failures=projection.rehydration_failures,
            turn_local_full_groups=projection.turn_local_full_groups,
            artifact_rehydrated_groups=projection.artifact_rehydrated_groups,
            artifact_rehydrate_count=projection.artifact_rehydrate_count,
            artifact_externalized_count=projection.artifact_externalized_count,
        )
