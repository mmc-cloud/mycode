from dataclasses import dataclass, field

from mycode.agent import AgentToolCall
from mycode.runner import (
    ToolBatchExecution,
    ToolCallExecution,
    append_tool_call_limit_failures,
    execute_tool_batch,
    partition_tool_calls_by_limit,
)
from mycode.subagents.concurrency import BoundedDelegationScheduler
from mycode.subagents.limits import (
    DEFAULT_MAX_CONCURRENT_DELEGATIONS,
    DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN,
)
from mycode.tools.base import ToolResult
from mycode.tools.registry import ToolRegistry


@dataclass
class DelegationToolBatchHandler:
    max_delegations_per_run: int = DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN
    max_concurrent_delegations: int = DEFAULT_MAX_CONCURRENT_DELEGATIONS
    delegation_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_delegations_per_run < 1:
            raise ValueError("max_delegations_per_run must be at least 1.")
        if self.max_concurrent_delegations < 1:
            raise ValueError("max_concurrent_delegations must be at least 1.")

    def start_run(self) -> None:
        self.delegation_count = 0

    def __call__(
        self,
        registry: ToolRegistry,
        tool_calls: list[AgentToolCall],
    ) -> ToolBatchExecution:
        delegation_calls = [
            tool_call for tool_call in tool_calls
            if tool_call.name == "delegate_task"
        ]
        executable_calls, overflow_executions = partition_tool_calls_by_limit(
            tool_calls
        )
        if not delegation_calls:
            return execute_tool_batch(registry, tool_calls)
        batch = self._execute_delegation_barrier(
            registry,
            executable_calls,
        )
        return append_tool_call_limit_failures(batch, overflow_executions)

    def _execute_delegation_barrier(
        self,
        registry: ToolRegistry,
        tool_calls: list[AgentToolCall],
    ) -> ToolBatchExecution:
        delegation_results: dict[int, ToolResult] = {}
        executable_delegations: list[tuple[int, AgentToolCall]] = []
        for index, tool_call in enumerate(tool_calls):
            if tool_call.name != "delegate_task":
                continue
            self.delegation_count += 1
            if self.delegation_count > self.max_delegations_per_run:
                delegation_results[index] = _delegation_limit_result(
                    tool_call,
                    max_delegations_per_run=self.max_delegations_per_run,
                )
            else:
                executable_delegations.append((index, tool_call))

        scheduler = BoundedDelegationScheduler(
            max_concurrent=self.max_concurrent_delegations
        )
        scheduled_calls = [
            tool_call for _, tool_call in executable_delegations
        ]
        scheduled_results = scheduler.execute(
            scheduled_calls,
            lambda tool_call: registry.run_tool(
                tool_call.name,
                tool_call.arguments,
            ),
        )
        for (index, _), result in zip(
            executable_delegations,
            scheduled_results,
            strict=True,
        ):
            delegation_results[index] = result

        executions: list[ToolCallExecution] = []
        for index, tool_call in enumerate(tool_calls):
            if tool_call.name == "delegate_task":
                result = delegation_results[index]
            else:
                result = ToolResult.failure(
                    error=(
                        "Tool call skipped because delegate_task is a control-flow "
                        "barrier and its result must be considered first."
                    ),
                    metadata={
                        "tool_name": tool_call.name,
                        "reason": "skipped_due_to_delegation_barrier",
                    },
                )
            executions.append(ToolCallExecution(tool_call=tool_call, result=result))
        return ToolBatchExecution(executions=tuple(executions))


def _delegation_limit_result(
    tool_call: AgentToolCall,
    *,
    max_delegations_per_run: int,
) -> ToolResult:
    return ToolResult.failure(
        error=(
            "Delegation limit reached for this parent Agent run: "
            f"{max_delegations_per_run}."
        ),
        metadata={
            "tool_name": tool_call.name,
            "reason": "delegation_limit",
            "max_delegations_per_run": max_delegations_per_run,
        },
    )
