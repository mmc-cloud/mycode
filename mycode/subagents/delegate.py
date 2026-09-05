from typing import Protocol

from mycode.subagents.contracts import SubAgentTask
from mycode.subagents.limits import MAX_DELEGATION_DEPTH
from mycode.subagents.observability import SubAgentObserver
from mycode.subagents.runtime import SubAgentExecution
from mycode.tools.base import PydanticTool, ToolResult


class SubAgentExecutor(Protocol):
    def execute(
        self,
        task: SubAgentTask,
        *,
        observer: SubAgentObserver | None = None,
    ) -> SubAgentExecution:
        pass


class DelegateTaskTool(PydanticTool[SubAgentTask]):
    name = "delegate_task"
    description = (
        "Delegate one bounded investigation, validation, or review task to an "
        "independent SubAgent and return its structured result. Multiple independent "
        "delegate_task calls in one response may run with bounded parallelism."
    )
    args_model = SubAgentTask
    capability = "control"
    risk = "low"

    def __init__(
        self,
        runtime: SubAgentExecutor,
        *,
        current_depth: int = 0,
        observer: SubAgentObserver | None = None,
    ) -> None:
        if current_depth < 0:
            raise ValueError("current_depth must not be negative.")
        self.runtime = runtime
        self.current_depth = current_depth
        self.observer = observer

    def _run(self, args: SubAgentTask) -> ToolResult:
        if self.current_depth >= MAX_DELEGATION_DEPTH:
            return ToolResult.failure(
                error=(
                    "SubAgent delegation depth limit reached: "
                    f"{MAX_DELEGATION_DEPTH}."
                ),
                metadata={
                    "tool_name": self.name,
                    "reason": "delegation_depth_limit",
                    "current_depth": self.current_depth,
                    "max_depth": MAX_DELEGATION_DEPTH,
                },
            )

        execution = self.runtime.execute(args, observer=self.observer)
        content = execution.result.model_dump_json(exclude_none=True)
        metadata: dict[str, object] = {
            "tool_name": self.name,
            "run_id": execution.result.run_id,
            "role": execution.result.role,
            "child_status": execution.result.status,
            "child_stop_reason": execution.result.stop_reason,
            "conversation_message_count": execution.conversation_message_count,
            "tool_call_count": execution.tool_call_count,
            "validation_execution_count": execution.validation_execution_count,
            "result_chars": len(content),
        }
        if execution.snapshot is not None:
            metadata["snapshot_sha256"] = execution.snapshot.combined_sha256
        if execution.token_usage is not None:
            metadata["token_usage"] = {
                "prompt_tokens": execution.token_usage.prompt_tokens,
                "completion_tokens": execution.token_usage.completion_tokens,
                "total_tokens": execution.token_usage.total_tokens,
            }
        return ToolResult.success(content=content, metadata=metadata)
