from collections.abc import Callable
from dataclasses import dataclass, field
import json

from pydantic import ValidationError

from mycode.agent import AgentModelResponse, AgentToolCall
from mycode.runner import (
    ToolBatchExecution,
    ToolCallExecution,
    append_tool_call_limit_failures,
    execute_tool_batch,
    partition_tool_calls_by_limit,
)
from mycode.subagents.audit import SubAgentToolAudit, build_tool_audit
from mycode.subagents.contracts import (
    BoundedResultArgs,
    SubAgentPayload,
    SubAgentRole,
    ValidationExecution,
)
from mycode.subagents.results import (
    ResultCompressionError,
    finalize_submitted_payload,
)
from mycode.tools.base import ToolResult
from mycode.tools.registry import ToolRegistry


ToolAuditHandler = Callable[[SubAgentToolAudit], None]


@dataclass
class SubAgentToolBatchHandler:
    role: SubAgentRole
    max_final_payload_chars: int
    max_validation_calls: int
    audit_handler: ToolAuditHandler | None = None
    submitted_payload: SubAgentPayload | None = None
    submission_attempted: bool = False
    tool_call_count: int = 0
    validation_call_count: int = 0
    validation_executions: list[ValidationExecution] = field(default_factory=list)
    validation_evidence_errors: list[str] = field(default_factory=list)

    def __call__(
        self,
        registry: ToolRegistry,
        tool_calls: list[AgentToolCall],
    ) -> ToolBatchExecution:
        self.tool_call_count += len(tool_calls)
        submission_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call.name == "submit_result"
        ]
        executable_calls, overflow_executions = partition_tool_calls_by_limit(
            tool_calls
        )
        if len(submission_calls) > 1:
            self.submission_attempted = True
            batch = ToolBatchExecution(
                executions=tuple(
                    ToolCallExecution(
                        tool_call=tool_call,
                        result=_multiple_submission_result(tool_call),
                    )
                    for tool_call in executable_calls
                )
            )
        elif submission_calls:
            self.submission_attempted = True
            batch = self._execute_submission_barrier(
                registry,
                executable_calls,
                submission_calls[0],
            )
        else:
            batch = self._execute_regular_batch(registry, tool_calls)
            self._emit_audits(batch)
            return batch
        batch = append_tool_call_limit_failures(batch, overflow_executions)
        self._emit_audits(batch)
        return batch

    def validate_submission(self, submitted: BoundedResultArgs) -> str | None:
        if self.validation_evidence_errors:
            return (
                "Runtime could not form trustworthy validation evidence: "
                + "; ".join(self.validation_evidence_errors[:3])
            )
        try:
            payload = finalize_submitted_payload(
                self.role,
                submitted,
                validation_executions=self.validation_executions,
                max_chars=self.max_final_payload_chars,
            )
        except (ValidationError, ResultCompressionError, ValueError) as error:
            return _submission_validation_error(error)
        self.submitted_payload = payload
        return None

    def _execute_submission_barrier(
        self,
        registry: ToolRegistry,
        tool_calls: list[AgentToolCall],
        submission_call: AgentToolCall,
    ) -> ToolBatchExecution:
        self.submission_attempted = True
        executions: list[ToolCallExecution] = []
        submission_result: ToolResult | None = None
        submission_is_executable = any(
            tool_call is submission_call for tool_call in tool_calls
        )
        for tool_call in tool_calls:
            if tool_call is submission_call and submission_is_executable:
                submission_result = registry.run_tool(
                    tool_call.name,
                    tool_call.arguments,
                )
                result = submission_result
            else:
                result = ToolResult.failure(
                    error=(
                        "Tool call skipped because submit_result is a control-flow "
                        "barrier and must be handled first."
                    ),
                    metadata={
                        "tool_name": tool_call.name,
                        "reason": "skipped_due_to_submission_barrier",
                    },
                )
            executions.append(ToolCallExecution(tool_call=tool_call, result=result))

        stop_response = None
        if submission_result is not None and submission_result.ok:
            stop_response = AgentModelResponse(
                content="Structured SubAgent result submitted.",
                stop_reason="control_tool",
            )
        return ToolBatchExecution(
            executions=tuple(executions),
            stop_response=stop_response,
        )

    def _execute_regular_batch(
        self,
        registry: ToolRegistry,
        tool_calls: list[AgentToolCall],
    ) -> ToolBatchExecution:
        return execute_tool_batch(
            registry,
            tool_calls,
            serial_executor=lambda tool_call: self._execute_serial_tool(
                registry,
                tool_call,
            ),
        )

    def _execute_serial_tool(
        self,
        registry: ToolRegistry,
        tool_call: AgentToolCall,
    ) -> ToolResult:
        if tool_call.name == "run_validation":
            return self._run_validation(registry, tool_call)
        return registry.run_tool(tool_call.name, tool_call.arguments)

    def _run_validation(
        self,
        registry: ToolRegistry,
        tool_call: AgentToolCall,
    ) -> ToolResult:
        self.validation_call_count += 1
        if self.validation_call_count > self.max_validation_calls:
            return ToolResult.failure(
                error=(
                    "Validation call limit reached for this SubAgent run: "
                    f"{self.max_validation_calls}."
                ),
                metadata={
                    "tool_name": tool_call.name,
                    "reason": "validation_call_limit",
                    "max_validation_calls": self.max_validation_calls,
                },
            )

        result = registry.run_tool(tool_call.name, tool_call.arguments)
        self._record_validation_execution(tool_call, result)
        return result

    def _record_validation_execution(
        self,
        tool_call: AgentToolCall,
        result: ToolResult,
    ) -> None:
        metadata = result.metadata
        if "exit_code" not in metadata:
            return
        try:
            command = metadata.get("command", tool_call.arguments.get("command"))
            cwd = metadata.get("resolved_cwd", tool_call.arguments.get("cwd", "."))
            execution = ValidationExecution.model_validate(
                {
                    "command": command,
                    "cwd": str(cwd),
                    "exit_code": metadata.get("exit_code"),
                    "duration_ms": metadata.get("duration_ms", 0),
                    "timed_out": metadata.get("timed_out", False),
                }
            )
        except ValidationError as error:
            self.validation_evidence_errors.append(_submission_validation_error(error))
            return
        self.validation_executions.append(execution)

    def _emit_audits(self, batch: ToolBatchExecution) -> None:
        if self.audit_handler is None:
            return
        for execution in batch.executions:
            self.audit_handler(
                build_tool_audit(execution.tool_call, execution.result)
            )


def _multiple_submission_result(tool_call: AgentToolCall) -> ToolResult:
    if tool_call.name == "submit_result":
        return ToolResult.failure(
            error="Multiple submit_result calls in one model response are not allowed.",
            metadata={
                "tool_name": tool_call.name,
                "reason": "multiple_submission_barriers",
            },
        )
    return ToolResult.failure(
        error="Tool call skipped because the submission batch is invalid.",
        metadata={
            "tool_name": tool_call.name,
            "reason": "skipped_due_to_submission_batch_error",
        },
    )


def _submission_validation_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = [
            {
                "location": ".".join(str(item) for item in detail["loc"]),
                "message": detail["msg"],
                "type": detail["type"],
            }
            for detail in error.errors(include_input=False)[:5]
        ]
        return "Runtime result validation failed: " + json.dumps(
            details,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return f"Runtime result validation failed: {error}"
