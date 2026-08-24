from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import TypeVar

from mycode.agent import AgentToolCall
from mycode.tools.base import ToolResult


ResultT = TypeVar("ResultT")
DelegationExecutor = Callable[[AgentToolCall], ToolResult]


@dataclass
class SubAgentInteractionGate:
    """Serialize process-local SubAgent confirmation and observer interactions."""

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def run(self, operation: Callable[[], ResultT]) -> ResultT:
        with self._lock:
            return operation()


@dataclass(frozen=True)
class BoundedDelegationScheduler:
    """Run independent delegation calls in bounded fork-join chunks."""

    max_concurrent: int

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1.")

    def execute(
        self,
        tool_calls: list[AgentToolCall],
        executor: DelegationExecutor,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for start in range(0, len(tool_calls), self.max_concurrent):
            chunk = tool_calls[start : start + self.max_concurrent]
            results.extend(self._execute_chunk(chunk, executor))
        return results

    def _execute_chunk(
        self,
        tool_calls: list[AgentToolCall],
        executor: DelegationExecutor,
    ) -> list[ToolResult]:
        if not tool_calls:
            return []

        try:
            pool = ThreadPoolExecutor(
                max_workers=len(tool_calls),
                thread_name_prefix="mycode-subagent",
            )
        except Exception as error:
            return [
                _worker_failure("delegation_worker_start_error", error)
                for _ in tool_calls
            ]

        futures: list[tuple[int, Future[ToolResult]]] = []
        results: list[ToolResult | None] = [None] * len(tool_calls)
        process_exception: BaseException | None = None
        try:
            for index, tool_call in enumerate(tool_calls):
                try:
                    futures.append((index, pool.submit(executor, tool_call)))
                except Exception as error:
                    results[index] = _worker_failure(
                        "delegation_worker_start_error",
                        error,
                    )
                except BaseException as error:
                    process_exception = error
                    break

            for index, future in futures:
                try:
                    result = future.result()
                except Exception as error:
                    result = _worker_failure(
                        "delegation_worker_error",
                        error,
                    )
                except BaseException as error:
                    if process_exception is None:
                        process_exception = error
                    continue
                if not isinstance(result, ToolResult):
                    result = ToolResult.failure(
                        error="Delegation worker returned an invalid result.",
                        metadata={"reason": "delegation_worker_contract"},
                    )
                results[index] = result
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
        if process_exception is not None:
            raise process_exception
        return [
            result
            if result is not None
            else ToolResult.failure(
                error="Delegation worker did not produce a result.",
                metadata={"reason": "delegation_worker_missing_result"},
            )
            for result in results
        ]


def _worker_failure(reason: str, error: Exception) -> ToolResult:
    return ToolResult.failure(
        error="Delegation execution failed unexpectedly.",
        metadata={
            "reason": reason,
            "exception_type": type(error).__name__,
        },
    )
