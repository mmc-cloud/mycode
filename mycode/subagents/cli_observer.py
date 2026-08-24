from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from mycode.cli_presenter import CliDisplayMode
from mycode.subagents.audit import SubAgentToolAudit
from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.lifecycle import SubAgentStateTransition
from mycode.subagents.runtime import SubAgentExecution
from mycode.subagents.snapshots import SubAgentSnapshotMetadata


CLI_SUBAGENT_SUMMARY_CHARS = 160


@dataclass(frozen=True)
class CliSubAgentObserver:
    output: Callable[[str], None]
    mode: CliDisplayMode = "normal"

    def on_state(
        self,
        task: SubAgentTask,
        transition: SubAgentStateTransition,
    ) -> None:
        run = transition.run_id[:8]
        if transition.state == "running" and transition.reason == "run_started":
            if self.mode == "normal":
                self.output(f"activity> {transition.role.capitalize()} started")
                return
            self.output(
                f"subagent> start role={transition.role} run={run} "
                f"objective_chars={len(task.objective)} "
                f"context_chars={len(task.context)} scope_paths={len(task.scope_paths)}"
            )
            return
        if self.mode == "normal":
            if transition.state in {"failed", "interrupted"}:
                self.output(
                    f"activity> {transition.role.capitalize()} "
                    f"{transition.state}: {transition.reason}"
                )
            return
        self.output(
            f"subagent> role={transition.role} run={run} "
            f"state={transition.state} reason={transition.reason}"
        )

    def on_snapshot(
        self,
        task: SubAgentTask,
        run_id: str,
        snapshot: SubAgentSnapshotMetadata,
        occurred_at: datetime,
    ) -> None:
        if self.mode != "debug":
            return
        self.output(
            f"subagent_context> role={task.role} run={run_id[:8]} "
            f"snapshot={snapshot.combined_sha256[:12]} "
            f"instructions={len(snapshot.instructions.sources)} "
            f"memory_entries={len(snapshot.memory.selected_entries)}"
        )

    def on_tool_audit(
        self,
        task: SubAgentTask,
        run_id: str,
        audit: SubAgentToolAudit,
        occurred_at: datetime,
    ) -> None:
        if self.mode == "normal":
            return
        status = "ok" if audit.ok else "error"
        details = [
            f"subagent_tool> role={task.role}",
            f"name={audit.tool_name}",
            f"status={status}",
        ]
        if self.mode == "debug":
            details.insert(1, f"run={run_id[:8]}")
            details.append(f"args={audit.arguments_sha256[:12]}")
            details.append(f"output_chars={audit.output_chars}")
        if audit.exit_code is not None:
            details.append(f"exit_code={audit.exit_code}")
        if audit.duration_ms is not None:
            details.append(f"duration_ms={audit.duration_ms}")
        if audit.truncated:
            details.append("truncated=true")
        if audit.reason is not None:
            details.append(f"reason={audit.reason}")
        self.output(" ".join(details))

    def on_result(
        self,
        task: SubAgentTask,
        execution: SubAgentExecution,
        occurred_at: datetime,
    ) -> None:
        if self.mode == "normal":
            self.output(
                f"activity> {execution.result.role.capitalize()} "
                f"{execution.result.status}: {_summary(execution.result.summary)}"
            )
            return
        usage = execution.token_usage
        token_text = ""
        if usage is not None:
            token_text = f" tokens={usage.total_tokens}"
        run_text = (
            f" run={execution.result.run_id[:8]}" if self.mode == "debug" else ""
        )
        self.output(
            f"subagent_result> role={execution.result.role}{run_text} "
            f"status={execution.result.status} stop={execution.result.stop_reason}"
            f"{token_text} summary={_summary(execution.result.summary)!r}"
        )


def _summary(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= CLI_SUBAGENT_SUMMARY_CHARS:
        return normalized
    return normalized[: CLI_SUBAGENT_SUMMARY_CHARS - 3].rstrip() + "..."
