from dataclasses import dataclass
from datetime import datetime
import hashlib

from mycode.context_budget import MemoryContextStats
from mycode.session_store import WritableSession
from mycode.subagents.audit import SubAgentToolAudit
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.lifecycle import SubAgentStateTransition
from mycode.subagents.runtime import SubAgentExecution, SubAgentModelContextStats
from mycode.subagents.snapshots import SubAgentSnapshotMetadata


MAX_PERSISTED_SNAPSHOT_HASHES = 12


@dataclass(frozen=True)
class SessionSubAgentObserver:
    session: WritableSession

    def on_state(self, task: SubAgentTask, transition: SubAgentStateTransition) -> None:
        self.session.append_subagent_event(transition.run_id, {
            "type": "state", "state": transition.state, "role": transition.role,
            "reason": transition.reason, "occurred_at": transition.occurred_at.isoformat(),
            "task_sha256": _task_sha256(task),
            "objective_chars": len(task.objective), "context_chars": len(task.context),
            "scope_path_count": len(task.scope_paths),
        })

    def on_snapshot(
        self, task: SubAgentTask, run_id: str,
        snapshot: SubAgentSnapshotMetadata, occurred_at: datetime,
    ) -> None:
        self.session.append_subagent_event(run_id, {
            "type": "snapshot", "occurred_at": occurred_at.isoformat(),
            "snapshot": _snapshot_record(snapshot),
        })

    def on_tool_audit(
        self, task: SubAgentTask, run_id: str,
        audit: SubAgentToolAudit, occurred_at: datetime,
    ) -> None:
        self.session.append_subagent_event(run_id, {
            "type": "tool_audit", "occurred_at": occurred_at.isoformat(),
            "tool_name": audit.tool_name, "arguments_sha256": audit.arguments_sha256,
            "argument_summary": dict(audit.argument_summary), "ok": audit.ok,
            "exit_code": audit.exit_code, "duration_ms": audit.duration_ms,
            "output_chars": audit.output_chars, "truncated": audit.truncated,
            "reason": audit.reason,
        })

    def on_result(
        self, task: SubAgentTask, execution: SubAgentExecution, occurred_at: datetime,
    ) -> None:
        self.session.append_subagent_event(execution.result.run_id, {
            "type": "result", "occurred_at": occurred_at.isoformat(),
            "status": execution.result.status, "stop_reason": execution.result.stop_reason,
            "result": execution.result.model_dump(mode="json", exclude_none=True),
            "context": _context_record(execution.context),
            "token_usage": None if execution.token_usage is None else {
                "prompt_tokens": execution.token_usage.prompt_tokens,
                "completion_tokens": execution.token_usage.completion_tokens,
                "total_tokens": execution.token_usage.total_tokens,
            },
            "tool_call_count": execution.tool_call_count,
            "validation_execution_count": execution.validation_execution_count,
        })


def _task_sha256(task: SubAgentTask) -> str:
    serialized = task.model_dump_json(exclude_none=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _snapshot_record(snapshot: SubAgentSnapshotMetadata) -> dict[str, object]:
    instruction_hashes = [
        source.sha256
        for source in snapshot.instructions.sources[:MAX_PERSISTED_SNAPSHOT_HASHES]
    ]
    memory_source_hashes = [
        source.sha256
        for source in snapshot.memory.sources[:MAX_PERSISTED_SNAPSHOT_HASHES]
    ]
    memory_entry_hashes = [
        entry.sha256
        for entry in snapshot.memory.selected_entries[:MAX_PERSISTED_SNAPSHOT_HASHES]
    ]
    return {
        "combined_sha256": snapshot.combined_sha256,
        "instructions": {
            "combined_sha256": snapshot.instructions.combined_sha256,
            "total_chars": snapshot.instructions.total_chars,
            "source_count": len(snapshot.instructions.sources),
            "warning_count": len(snapshot.instructions.warnings),
            "source_sha256": instruction_hashes,
            "source_hashes_omitted": max(
                0,
                len(snapshot.instructions.sources) - len(instruction_hashes),
            ),
        },
        "memory": {
            "enabled": snapshot.memory.enabled,
            "combined_sha256": snapshot.memory.combined_sha256,
            "source_count": len(snapshot.memory.sources),
            "selected_entry_count": len(snapshot.memory.selected_entries),
            "source_sha256": memory_source_hashes,
            "selected_entry_sha256": memory_entry_hashes,
            "source_hashes_omitted": max(
                0,
                len(snapshot.memory.sources) - len(memory_source_hashes),
            ),
            "entry_hashes_omitted": max(
                0,
                len(snapshot.memory.selected_entries) - len(memory_entry_hashes),
            ),
            "stats": _memory_stats_record(snapshot.memory.stats),
        },
    }


def _context_record(
    context: SubAgentModelContextStats | None,
) -> dict[str, object] | None:
    if context is None:
        return None
    return {
        "estimated_input_tokens": context.estimated_input_tokens,
        "max_input_tokens": context.max_input_tokens,
        "selected_message_count": context.selected_message_count,
        "original_message_count": context.original_message_count,
        "compressed_tool_result_count": context.compressed_tool_result_count,
        "memory_stats": _memory_stats_record(context.memory_stats),
    }


def _memory_stats_record(
    stats: MemoryContextStats | None,
) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "safe_entry_count": stats.safe_entry_count,
        "relevant_entry_count": stats.relevant_entry_count,
        "selected_entry_count": stats.selected_entry_count,
        "included_entry_count": stats.included_entry_count,
        "estimated_tokens": stats.estimated_tokens,
        "irrelevant_entry_count": stats.irrelevant_entry_count,
        "conflict_count": stats.conflict_count,
        "budget_omitted_count": stats.budget_omitted_count,
        "issue_count": stats.issue_count,
        "scopes": list(stats.scopes),
    }
