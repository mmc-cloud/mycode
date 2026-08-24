from dataclasses import dataclass, field, replace
from collections.abc import Callable, Iterator
import asyncio
import json

from pydantic import ValidationError

from mycode.agent import (
    AgentEvent,
    AgentModelResponse,
    AgentProgressSnapshot,
    AgentToolCall,
    AgentWarning,
)
from mycode.artifacts import (
    ToolResultArtifactStore,
    artifact_externalization_failure_content,
    artifact_failure_reason,
)
from mycode.context_compact import ConversationCompactor
from mycode.error_handling import format_model_error
from mycode.context_budget import (
    ContextBudget,
    ModelContext,
    TokenEstimator,
    TokenUsage,
    build_model_context,
    format_model_context_stats,
    model_context_needs_notice,
)
from mycode.conversation import Conversation
from mycode.llm import LLMClient
from mycode.memory_context import MemoryRecall, MemoryRecallProvider
from mycode.messages import Message
from mycode.reasoning import ReasoningState
from mycode.run_progress import (
    COMPLETION_CORRECTION_EXTRA_TURNS,
    DEFAULT_MAX_TURNS,
    DEFAULT_READONLY_TURN_LIMIT,
    DEFAULT_STAGNANT_TURN_LIMIT,
    MAIN_CONVERGENCE_PROMPT,
    MAIN_CONVERGENCE_REMAINING_TURNS,
    MAX_TURNS_FINALIZATION_PROMPT,
    ReadinessMode,
    RunProgress,
    ToolBehaviorObservation,
    classify_tool_behavior,
    normalize_run_checkpoint,
    resume_guidance,
    tool_result_evidence,
    turn_guidance,
)
from mycode.tools import ToolRegistry, ToolResult


DEFAULT_REPEATED_TOOL_CALL_LIMIT = 3
DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE = 32
DEFAULT_MAX_CONCURRENT_SAFE_TOOLS = 4
_OBSERVABLE_BOUNDED_READ_FIELDS = {
    "read_artifact": "max_chars",
    "read_file": "max_lines",
    "grep": "max_results",
    "glob": "max_results",
}


class ToolBatchContractError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCallExecution:
    tool_call: AgentToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolBatchExecution:
    executions: tuple[ToolCallExecution, ...]
    stop_response: AgentModelResponse | None = None


@dataclass(frozen=True)
class _ReadinessToolTurn:
    batch_override: ToolBatchExecution | None = None


ToolBatchHandler = Callable[
    [ToolRegistry, list[AgentToolCall]],
    ToolBatchExecution,
]
SerialToolExecutor = Callable[[AgentToolCall], ToolResult]


def execute_tool_batch(
    registry: ToolRegistry,
    tool_calls: list[AgentToolCall],
    *,
    serial_executor: SerialToolExecutor | None = None,
) -> ToolBatchExecution:
    return asyncio.run(
        execute_tool_batch_async(
            registry,
            tool_calls,
            serial_executor=serial_executor,
        )
    )


async def execute_tool_batch_async(
    registry: ToolRegistry,
    tool_calls: list[AgentToolCall],
    *,
    serial_executor: SerialToolExecutor | None = None,
) -> ToolBatchExecution:
    executions: list[ToolCallExecution] = []
    concurrent_calls: list[AgentToolCall] = []
    permission_lock = asyncio.Lock()
    executable_calls, overflow_executions = partition_tool_calls_by_limit(tool_calls)

    for tool_call in executable_calls:
        if registry.is_concurrency_safe(tool_call.name):
            concurrent_calls.append(tool_call)
            continue

        executions.extend(
            await _execute_concurrent_calls(
                registry,
                concurrent_calls,
                permission_lock,
            )
        )
        concurrent_calls.clear()
        executions.append(
            _execute_serial_call(
                registry,
                tool_call,
                serial_executor=serial_executor,
            )
        )

    executions.extend(
        await _execute_concurrent_calls(
            registry,
            concurrent_calls,
            permission_lock,
        )
    )

    executions.extend(overflow_executions)
    return ToolBatchExecution(executions=tuple(executions))


def partition_tool_calls_by_limit(
    tool_calls: list[AgentToolCall],
) -> tuple[list[AgentToolCall], tuple[ToolCallExecution, ...]]:
    executable_calls = tool_calls[:DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE]
    overflow_executions = tuple(
        _tool_call_limit_failure(tool_call, index)
        for index, tool_call in enumerate(
            tool_calls[DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE:],
            start=DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
        )
    )
    return executable_calls, overflow_executions


def append_tool_call_limit_failures(
    batch: ToolBatchExecution,
    overflow_executions: tuple[ToolCallExecution, ...],
) -> ToolBatchExecution:
    if not overflow_executions:
        return batch
    return ToolBatchExecution(
        executions=(*batch.executions, *overflow_executions),
        stop_response=batch.stop_response,
    )


async def _execute_concurrent_calls(
    registry: ToolRegistry,
    tool_calls: list[AgentToolCall],
    permission_lock: asyncio.Lock,
) -> list[ToolCallExecution]:
    if not tool_calls:
        return []

    executions: list[ToolCallExecution] = []
    for start in range(0, len(tool_calls), DEFAULT_MAX_CONCURRENT_SAFE_TOOLS):
        chunk = tool_calls[start : start + DEFAULT_MAX_CONCURRENT_SAFE_TOOLS]
        results = await asyncio.gather(
            *(
                registry.run_tool_async(
                    tool_call.name,
                    tool_call.arguments,
                    permission_lock=permission_lock,
                )
                for tool_call in chunk
            ),
            return_exceptions=True,
        )

        for tool_call, result in zip(chunk, results, strict=True):
            if isinstance(result, ToolResult):
                tool_result = result
            elif isinstance(result, Exception):
                tool_result = ToolResult.failure(
                    error=f"Tool execution failed: {result}",
                    metadata={"exception_type": type(result).__name__},
                )
            else:
                raise result

            executions.append(
                ToolCallExecution(
                    tool_call=tool_call,
                    result=tool_result,
                )
            )
    return executions


def _tool_call_limit_failure(
    tool_call: AgentToolCall,
    index: int,
) -> ToolCallExecution:
    return ToolCallExecution(
        tool_call=tool_call,
        result=ToolResult.failure(
            error=(
                "Tool call was not executed because the model response exceeded "
                f"the maximum of {DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE} tool calls."
            ),
            metadata={
                "reason": "tool_call_response_limit",
                "tool_call_index": index,
                "max_tool_calls": DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
            },
        ),
    )


def _execute_serial_call(
    registry: ToolRegistry,
    tool_call: AgentToolCall,
    *,
    serial_executor: SerialToolExecutor | None = None,
) -> ToolCallExecution:
    try:
        result = (
            registry.run_tool(tool_call.name, tool_call.arguments)
            if serial_executor is None
            else serial_executor(tool_call)
        )
    except Exception as error:
        result = ToolResult.failure(
            error=f"Tool execution failed: {error}",
            metadata={"exception_type": type(error).__name__},
        )
    return ToolCallExecution(tool_call=tool_call, result=result)


@dataclass
class AgentRunner:
    llm_client: LLMClient
    tool_registry: ToolRegistry
    conversation: Conversation = field(default_factory=Conversation)
    max_turns: int = DEFAULT_MAX_TURNS
    convergence_remaining_turns: int | None = MAIN_CONVERGENCE_REMAINING_TURNS
    convergence_prompt: str | None = MAIN_CONVERGENCE_PROMPT
    repeated_tool_call_limit: int = DEFAULT_REPEATED_TOOL_CALL_LIMIT
    stagnant_turn_limit: int = DEFAULT_STAGNANT_TURN_LIMIT
    readonly_turn_limit: int | None = DEFAULT_READONLY_TURN_LIMIT
    finalize_on_max_turns: bool = True
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    token_estimator: TokenEstimator = field(default_factory=TokenEstimator)
    last_model_context: ModelContext | None = field(default=None, init=False)
    last_token_usage: TokenUsage | None = field(default=None, init=False)
    run_token_usage: TokenUsage | None = field(default=None, init=False)
    last_reasoning_char_count: int = field(default=0, init=False)
    instruction_sources: tuple[str, ...] = ()
    instruction_warnings: tuple[str, ...] = ()
    memory_context_selector: MemoryRecallProvider | None = None
    last_memory_recall: MemoryRecall | None = field(default=None, init=False)
    tool_batch_handler: ToolBatchHandler = execute_tool_batch
    compactor: ConversationCompactor | None = None
    tool_result_artifact_store: ToolResultArtifactStore | None = None
    last_artifact_error: str | None = field(default=None, init=False)
    artifact_failure_count: int = field(default=0, init=False)
    last_run_progress: RunProgress | None = field(default=None, init=False)
    _pending_artifact_warnings: list[AgentWarning] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if self.repeated_tool_call_limit < 1:
            raise ValueError("repeated_tool_call_limit must be at least 1.")
        if self.stagnant_turn_limit < 1:
            raise ValueError("stagnant_turn_limit must be at least 1.")
        if self.readonly_turn_limit is not None and self.readonly_turn_limit < 1:
            raise ValueError("readonly_turn_limit must be at least 1 or None.")
        if (self.convergence_remaining_turns is None) != (
            self.convergence_prompt is None
        ):
            raise ValueError(
                "convergence_remaining_turns and convergence_prompt must both be set "
                "or both be None."
            )
        if self.convergence_remaining_turns is None:
            return
        if not 0 < self.convergence_remaining_turns < self.max_turns:
            raise ValueError(
                "convergence_remaining_turns must be greater than 0 and less than "
                "max_turns."
            )
        if not self.convergence_prompt or not self.convergence_prompt.strip():
            raise ValueError("convergence_prompt must not be blank.")

    def run(self, user_message: str) -> Iterator[AgentEvent]:
        self._pending_artifact_warnings.clear()
        try:
            self.run_token_usage = None
            _start_tool_batch_run(self.tool_batch_handler)
            continuation_guidance = resume_guidance(self.conversation, user_message)
            self._recall_memory(user_message)
            self.conversation.add_user_message(user_message)

            previous_tool_call_signature: str | None = None
            repeated_tool_call_count = 0
            progress = RunProgress(
                stagnant_turn_limit=self.stagnant_turn_limit,
                readonly_turn_limit=self.readonly_turn_limit,
            )
            self.last_run_progress = progress
            reported_context_state: (
                tuple[
                    int,
                    int,
                    bool,
                    int,
                    int,
                    str | None,
                    bool,
                    str | None,
                    int,
                ]
                | None
            ) = None

            for turn_index in range(
                self.max_turns + COMPLETION_CORRECTION_EXTRA_TURNS
            ):
                if turn_index >= self.max_turns and not progress.completion_correction_used:
                    break
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                reasoning_state: ReasoningState = "absent"
                tool_calls: list[AgentToolCall] = []
                deferred_visible_events: list[AgentEvent] = []
                defer_completion_candidate = (
                    self.tool_registry.get("run_validation") is not None
                    and progress.needs_completion_correction
                )
                readiness_mode = progress.readiness_mode
                tools = _tool_schemas_for_readiness_mode(
                    self.tool_registry,
                    readiness_mode,
                )
                readiness_guidance, readiness_notice = (
                    progress.readiness_guidance_for_turn(turn_index + 1)
                )
                guidance, guidance_notice = turn_guidance(
                    turn_index=turn_index,
                    max_turns=self.max_turns,
                    convergence_remaining_turns=self.convergence_remaining_turns,
                    convergence_prompt=self.convergence_prompt,
                    task_phase_guidance=progress.task_phase_guidance_for_turn(),
                    readiness_guidance=readiness_guidance,
                    readiness_notice=readiness_notice,
                    resume_guidance=continuation_guidance,
                    replan_reason=progress.take_replan_reason(),
                    completion_correction_guidance=(
                        progress.take_completion_correction_guidance()
                    ),
                )
                yield AgentEvent(
                    type="turn",
                    content=guidance_notice,
                    turn_number=turn_index + 1,
                    max_turns=self.max_turns,
                )
                model_context = self._model_context(tools, guidance=guidance)
                for warning in self._drain_artifact_warnings():
                    yield AgentEvent(type="artifact_warning", content=warning.content)

                context_state = _context_state(model_context)
                should_report_context = reported_context_state is None or (
                    model_context_needs_notice(model_context)
                    and context_state != reported_context_state
                )
                if should_report_context:
                    yield _context_event(
                        model_context,
                        previous_token_usage=self.last_token_usage,
                    )
                    reported_context_state = context_state

                if model_context.estimate.over_budget:
                    yield AgentEvent(
                        type="error",
                        error=_context_overflow_message(model_context),
                    )
                    yield AgentEvent(type="stop", stop_reason="context_overflow")
                    return

                yield AgentEvent(type="model_start")

                try:
                    for event in self.llm_client.stream_with_tools(
                        Conversation.from_messages(list(model_context.messages)),
                        tools,
                    ):
                        if event.type == "reasoning_delta":
                            reasoning_parts.append(event.reasoning_content)
                            reasoning_state = "present_nonempty"
                            continue

                        if event.type == "reasoning_state":
                            reasoning_state = event.reasoning_state
                            continue

                        if event.type == "text_delta":
                            content_parts.append(event.content)
                            if defer_completion_candidate:
                                deferred_visible_events.append(event)
                            else:
                                yield event
                            continue

                        if event.type == "tool_call" and event.tool_call is not None:
                            tool_calls.append(event.tool_call)
                            observable_event = replace(
                                event,
                                tool_call=_observable_tool_call(
                                    self.tool_registry,
                                    event.tool_call,
                                ),
                            )
                            if defer_completion_candidate:
                                deferred_visible_events.append(observable_event)
                            else:
                                yield observable_event
                            continue

                        if event.type == "error":
                            self._observe_token_usage(model_context)
                            yield event
                            yield AgentEvent(type="stop", stop_reason="model_error")
                            return
                except Exception as error:
                    yield AgentEvent(
                        type="error",
                        error=format_model_error(
                            error,
                            operation="模型流式请求失败",
                        ),
                    )
                    yield AgentEvent(type="stop", stop_reason="model_error")
                    return

                self._observe_token_usage(model_context)
                content = "".join(content_parts)
                if not tool_calls:
                    if (
                        self.tool_registry.get("run_validation") is not None
                        and progress.intercept_unverified_completion_once()
                    ):
                        yield _progress_event(progress)
                        continue
                    yield from deferred_visible_events
                    progress.observe_final_answer()
                    self.conversation.add_assistant_message(content)
                    yield _progress_event(progress)
                    yield AgentEvent(type="stop", stop_reason="final_answer")
                    return

                yield from deferred_visible_events

                readiness_turn = _prepare_readiness_tool_turn(
                    progress,
                    registry=self.tool_registry,
                    readiness_mode=readiness_mode,
                    tool_calls=tool_calls,
                )
                batch_override = readiness_turn.batch_override

                if batch_override is None:
                    repeated_response = _check_repeated_tool_calls(
                        tool_calls,
                        previous_tool_call_signature=previous_tool_call_signature,
                        repeated_tool_call_count=repeated_tool_call_count,
                        repeated_tool_call_limit=self.repeated_tool_call_limit,
                    )
                    if repeated_response.stop_response is not None:
                        yield AgentEvent(
                            type="stop",
                            content=repeated_response.stop_response.content,
                            stop_reason=repeated_response.stop_response.stop_reason,
                        )
                        return

                    previous_tool_call_signature = (
                        repeated_response.previous_tool_call_signature
                    )
                    repeated_tool_call_count = repeated_response.repeated_tool_call_count
                    if repeated_tool_call_count >= 2:
                        progress.request_replan("模型重复了相同的工具调用")

                self.conversation.add_assistant_tool_calls(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content="".join(reasoning_parts) or None,
                    reasoning_state=reasoning_state,
                )

                batch = (
                    batch_override
                    if batch_override is not None
                    else self.tool_batch_handler(self.tool_registry, tool_calls)
                )
                _validate_tool_batch(tool_calls, batch)
                progress.observe_evidence(_batch_evidence(batch))
                if batch_override is None:
                    _observe_tool_turn_progress(
                        progress,
                        turn_number=turn_index + 1,
                        registry=self.tool_registry,
                        batch=batch,
                    )
                for execution in batch.executions:
                    yield AgentEvent(type="tool_result", tool_result=execution.result)
                    persisted_content = self._tool_result_content(execution)
                    for warning in self._drain_artifact_warnings():
                        yield AgentEvent(type="artifact_warning", content=warning.content)
                    self.conversation.add_tool_result_message(
                        tool_call_id=execution.tool_call.id,
                        content=persisted_content,
                    )
                yield _progress_event(progress)
                if batch.stop_response is not None:
                    yield AgentEvent(
                        type="stop",
                        content=batch.stop_response.content,
                        stop_reason=batch.stop_response.stop_reason,
                    )
                    return

            progress.observe_turn_limit()
            yield _progress_event(progress)
            if self.finalize_on_max_turns:
                yield from self._stream_finalization_after_max_turns()
                return
            yield AgentEvent(
                type="stop",
                content="Agent stopped because it reached the maximum number of turns.",
                stop_reason="max_turns",
            )

        finally:
            self._pending_artifact_warnings.clear()

    def _stream_finalization_after_max_turns(self) -> Iterator[AgentEvent]:
        finalization, model_context = self._max_turns_finalization_context()
        if finalization is None:
            yield AgentEvent(
                type="stop",
                content=(
                    f"本轮已达到 {self.max_turns} 轮上限，最终整理时上下文"
                    "超过模型可用范围，未能生成阶段性结果。"
                ),
                stop_reason="max_turns",
            )
            return

        content_parts: list[str] = []
        yield AgentEvent(type="model_start")
        try:
            for chunk in self.llm_client.stream_complete(finalization):
                if chunk == "":
                    continue
                content_parts.append(chunk)
                yield AgentEvent(type="text_delta", content=chunk)
        except Exception as error:
            yield AgentEvent(
                type="error",
                error=format_model_error(
                    error,
                    operation="最终整理请求失败",
                ),
            )
            yield AgentEvent(
                type="stop",
                content=(
                    f"本轮已达到 {self.max_turns} 轮上限，且未能生成"
                    "阶段性结果。"
                ),
                stop_reason="max_turns",
            )
            return

        self._observe_token_usage(model_context)
        content = "".join(content_parts).strip()
        if content == "":
            yield AgentEvent(
                type="stop",
                content=(
                    f"本轮已达到 {self.max_turns} 轮上限，模型没有返回"
                    "可用的阶段性结果。"
                ),
                stop_reason="max_turns",
            )
            return

        content = normalize_run_checkpoint(content)
        self.conversation.add_assistant_message(content)
        yield AgentEvent(
            type="stop",
            content=(
                f"本轮已达到 {self.max_turns} 轮上限；上面是基于现有信息"
                "整理的阶段性结果。"
            ),
            stop_reason="max_turns",
        )

    def _max_turns_finalization_context(
        self,
    ) -> tuple[Conversation | None, ModelContext]:
        model_context = self._model_context([])
        if model_context.estimate.over_budget:
            return None, model_context
        finalization_candidate = Conversation.from_messages(
            list(model_context.messages)
        )
        finalization_candidate.add_user_message(MAX_TURNS_FINALIZATION_PROMPT)
        finalization_context = build_model_context(
            finalization_candidate,
            self.context_budget,
            tools=[],
            token_estimator=self.token_estimator,
        )
        if finalization_context.estimate.over_budget:
            return None, finalization_context
        return (
            Conversation.from_messages(list(finalization_context.messages)),
            finalization_context,
        )

    def _model_context(
        self,
        tools: list[dict[str, object]],
        *,
        guidance: tuple[str, ...] = (),
    ) -> ModelContext:
        memory_recall = self.last_memory_recall
        canonical_conversation = self._artifact_context_conversation()
        context_conversation = canonical_conversation
        compact_stats = None
        if self.compactor is not None:
            prepared = self.compactor.prepare(
                context_conversation,
                self.context_budget,
                tools=tools,
                token_estimator=self.token_estimator,
                memory_message=(
                    None if memory_recall is None else memory_recall.message
                ),
            )
            context_conversation = prepared.conversation
            compact_stats = prepared.stats
            self._accumulate_run_token_usage(prepared.attempt_token_usage)

        context = build_model_context(
            context_conversation,
            self.context_budget,
            tools=tools,
            token_estimator=self.token_estimator,
            memory_message=(
                None if memory_recall is None else memory_recall.message
            ),
            memory_stats=(
                None if memory_recall is None else memory_recall.stats
            ),
        )
        if (
            context.estimate.over_budget
            and compact_stats is not None
            and compact_stats.boundary_id is not None
        ):
            canonical_fallback = build_model_context(
                canonical_conversation,
                self.context_budget,
                tools=tools,
                token_estimator=self.token_estimator,
                memory_message=(
                    None if memory_recall is None else memory_recall.message
                ),
                memory_stats=(
                    None if memory_recall is None else memory_recall.stats
                ),
            )
            if not canonical_fallback.estimate.over_budget:
                context = canonical_fallback
                compact_stats = replace(
                    compact_stats,
                    status="canonical_fallback",
                    summary_visible=False,
                )
        if compact_stats is not None:
            context = replace(context, compact_stats=compact_stats)
        if guidance:
            context = _add_model_guidance(
                context,
                guidance=guidance,
                budget=self.context_budget,
                tools=tools,
                token_estimator=self.token_estimator,
            )
        self.last_model_context = context
        return context

    def _artifact_context_conversation(self) -> Conversation:
        if self.tool_result_artifact_store is None:
            return self.conversation
        try:
            return self.tool_result_artifact_store.externalize_conversation(
                self.conversation,
                on_failure=self._historical_artifact_failure_content,
            )
        except Exception as error:
            reason = self._record_artifact_failure(
                error,
                phase="history_projection",
                tool_name="unknown",
            )
            return self._redact_oversized_tool_results(reason)

    def _tool_result_content(self, execution: ToolCallExecution) -> str:
        content = format_tool_result(execution.result)
        store = self.tool_result_artifact_store
        if store is None:
            return content
        try:
            externalized = store.externalize(
                tool_name=execution.tool_call.name,
                tool_call_id=execution.tool_call.id,
                content=content,
            )
        except Exception as error:
            reason = self._record_artifact_failure(
                error,
                phase="tool_result",
                tool_name=execution.tool_call.name,
            )
            return artifact_externalization_failure_content(
                tool_name=execution.tool_call.name,
                tool_call_id=execution.tool_call.id,
                original_content=content,
                reason=reason,
            )
        return externalized

    def _historical_artifact_failure_content(
        self,
        tool_name: str,
        tool_call_id: str,
        content: str,
        error: Exception,
    ) -> str:
        reason = self._record_artifact_failure(
            error,
            phase="history_projection",
            tool_name=tool_name,
        )
        return artifact_externalization_failure_content(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            original_content=content,
            reason=reason,
        )

    def _record_artifact_failure(
        self,
        error: Exception,
        *,
        phase: str,
        tool_name: str,
    ) -> str:
        reason = artifact_failure_reason(error)
        self.artifact_failure_count += 1
        self.last_artifact_error = reason
        safe_tool_name = _safe_event_tool_name(tool_name)
        self._pending_artifact_warnings.append(
            AgentWarning(
                type="artifact_externalization",
                content=(
                    f"phase={phase}, reason={reason}, tool={safe_tool_name}, "
                    f"failures={self.artifact_failure_count}; "
                    "tool body was not persisted"
                ),
            )
        )
        return reason

    def _drain_artifact_warnings(self) -> tuple[AgentWarning, ...]:
        warnings = tuple(self._pending_artifact_warnings)
        self._pending_artifact_warnings.clear()
        return warnings

    def _redact_oversized_tool_results(self, reason: str) -> Conversation:
        store = self.tool_result_artifact_store
        if store is None:
            return self.conversation
        tool_names: dict[str, str] = {}
        messages = []
        for message in self.conversation.get_messages():
            if message.role == "assistant":
                for tool_call in message.tool_calls:
                    tool_names[tool_call.id] = tool_call.name
            if (
                message.role != "tool"
                or message.tool_call_id is None
                or len(message.content) <= store.threshold_chars
            ):
                messages.append(message)
                continue
            tool_name = tool_names.get(message.tool_call_id, "unknown")
            messages.append(
                Message(
                    role="tool",
                    content=artifact_externalization_failure_content(
                        tool_name=tool_name,
                        tool_call_id=message.tool_call_id,
                        original_content=message.content,
                        reason=reason,
                    ),
                    tool_call_id=message.tool_call_id,
                )
            )
        return Conversation.from_messages(messages)

    def _recall_memory(self, user_message: str) -> None:
        if self.memory_context_selector is None:
            self.last_memory_recall = None
            return
        self.last_memory_recall = self.memory_context_selector.recall(user_message)

    def _observe_token_usage(self, context: ModelContext) -> None:
        usage = getattr(self.llm_client, "last_token_usage", None)
        self.last_token_usage = usage
        self.last_reasoning_char_count = getattr(
            self.llm_client,
            "last_reasoning_char_count",
            0,
        )
        self._accumulate_run_token_usage(usage)
        self.token_estimator.observe(context.estimate, usage)

    def _accumulate_run_token_usage(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        previous = self.run_token_usage
        self.run_token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens + (
                0 if previous is None else previous.prompt_tokens
            ),
            completion_tokens=usage.completion_tokens + (
                0 if previous is None else previous.completion_tokens
            ),
            total_tokens=usage.total_tokens + (
                0 if previous is None else previous.total_tokens
            ),
        )


@dataclass(frozen=True)
class _RepeatedToolCallCheck:
    previous_tool_call_signature: str | None
    repeated_tool_call_count: int
    stop_response: AgentModelResponse | None = None


def format_tool_result(result: ToolResult) -> str:
    status = "OK" if result.ok else "ERROR"
    body = result.content if result.ok else result.error or ""
    metadata = json.dumps(result.metadata, ensure_ascii=False, sort_keys=True, default=str)

    return f"{status}\n{body}\n\nMETADATA\n{metadata}"


def _context_event(
    context: ModelContext,
    *,
    previous_token_usage: TokenUsage | None,
) -> AgentEvent:
    return AgentEvent(
        type="context",
        content=format_model_context_stats(
            context,
            previous_prompt_tokens=(
                previous_token_usage.prompt_tokens
                if previous_token_usage is not None
                else None
            ),
        ),
    )


def _progress_event(progress: RunProgress) -> AgentEvent:
    return AgentEvent(
        type="progress",
        progress=AgentProgressSnapshot(
            task_phase=progress.task_phase,
            behaviors=tuple(sorted(progress.last_tool_behaviors)),
            transition_reason=progress.last_progress_reason,
            ready_investigation_turn_count=progress.ready_investigation_turn_count,
            done_extra_tool_turn_count=progress.done_extra_tool_turn_count,
        ),
    )


def _context_state(
    context: ModelContext,
) -> tuple[int, int, bool, int, int, str | None, bool, str | None, int]:
    memory = context.memory_stats
    compact = context.compact_stats
    return (
        context.trimmed_message_count,
        context.compressed_tool_result_count,
        context.estimate.over_budget,
        0 if memory is None else memory.included_entry_count,
        0 if memory is None else memory.issue_count,
        None if compact is None else compact.boundary_id,
        False if compact is None else compact.summary_visible,
        None if compact is None else compact.status,
        0 if compact is None else compact.consecutive_failure_count,
    )


def _context_overflow_message(context: ModelContext) -> str:
    return (
        "Model request was not sent because the estimated input context exceeds "
        f"the configured budget ({context.estimate.estimated_input_tokens}/"
        f"{context.estimate.max_input_tokens} tokens)."
    )


def _add_model_guidance(
    context: ModelContext,
    *,
    guidance: tuple[str, ...],
    budget: ContextBudget,
    tools: list[dict[str, object]],
    token_estimator: TokenEstimator,
) -> ModelContext:
    guided_conversation = Conversation.from_messages(
        [
            *context.messages,
            Message(role="system", content="\n\n".join(guidance)),
        ]
    )
    guided_context = build_model_context(
        guided_conversation,
        budget,
        tools=tools,
        token_estimator=token_estimator,
    )
    return replace(
        guided_context,
        original_message_count=context.original_message_count + 1,
        memory_stats=context.memory_stats,
        compact_stats=context.compact_stats,
    )


def _batch_evidence(batch: ToolBatchExecution) -> set[str]:
    return {
        tool_result_evidence(
            tool_name=execution.tool_call.name,
            ok=execution.result.ok,
            content=execution.result.content,
            error=execution.result.error,
        )
        for execution in batch.executions
    }


def _tool_schemas_for_readiness_mode(
    registry: ToolRegistry,
    readiness_mode: ReadinessMode,
) -> list[dict[str, object]]:
    if readiness_mode in {"open", "soft_convergence"}:
        return registry.get_schemas()

    schemas: list[dict[str, object]] = []
    for tool in registry.list_tools():
        capability = tool.get_permission_profile().capability
        if readiness_mode == "ready_action" and capability not in {
            "write",
            "command",
        }:
            continue
        schemas.append(tool.get_schema())
    return schemas


def _observable_tool_call(
    registry: ToolRegistry,
    tool_call: AgentToolCall,
) -> AgentToolCall:
    field_name = _OBSERVABLE_BOUNDED_READ_FIELDS.get(tool_call.name)
    if field_name is None or field_name not in tool_call.arguments:
        return tool_call

    tool = registry.get(tool_call.name)
    if tool is None:
        return tool_call
    try:
        args = tool.parse_arguments(tool_call.arguments)
    except ValidationError:
        return tool_call
    arguments = dict(tool_call.arguments)
    arguments[field_name] = args.model_dump()[field_name]
    return replace(tool_call, arguments=arguments)


def _prepare_readiness_tool_turn(
    progress: RunProgress,
    *,
    registry: ToolRegistry,
    readiness_mode: ReadinessMode,
    tool_calls: list[AgentToolCall],
) -> _ReadinessToolTurn:
    if readiness_mode == "open":
        return _ReadinessToolTurn()

    allowed_names = {
        str(schema["name"])
        for schema in _tool_schemas_for_readiness_mode(registry, readiness_mode)
    }
    if all(tool_call.name in allowed_names for tool_call in tool_calls):
        return _ReadinessToolTurn()

    if readiness_mode != "soft_convergence":
        progress.observe_restricted_tool_turn(
            attempted_investigation=any(
                (tool := registry.get(tool_call.name)) is not None
                and tool.get_permission_profile().capability == "read"
                for tool_call in tool_calls
            )
        )
    return _ReadinessToolTurn(
        batch_override=_rejected_tool_batch(
            tool_calls,
            error="Tool call is not available in the active readiness tool view.",
            reason="readiness_tool_not_allowed",
        )
    )


def _rejected_tool_batch(
    tool_calls: list[AgentToolCall],
    *,
    error: str,
    reason: str,
) -> ToolBatchExecution:
    return ToolBatchExecution(
        executions=tuple(
            ToolCallExecution(
                tool_call=tool_call,
                result=ToolResult.failure(
                    error=error,
                    metadata={
                        "tool_name": tool_call.name,
                        "reason": reason,
                    },
                ),
            )
            for tool_call in tool_calls
        )
    )


def _observe_tool_turn_progress(
    progress: RunProgress,
    *,
    turn_number: int,
    registry: ToolRegistry,
    batch: ToolBatchExecution,
) -> None:
    progress.observe_tool_turn(
        turn_number=turn_number,
        observations=_batch_tool_behavior_observations(registry, batch),
    )


def _batch_tool_behavior_observations(
    registry: ToolRegistry,
    batch: ToolBatchExecution,
) -> tuple[ToolBehaviorObservation, ...]:
    return tuple(
        ToolBehaviorObservation(
            behavior=classify_tool_behavior(
                tool_name=execution.tool_call.name,
                capability=(
                    tool.get_permission_profile().capability
                    if (tool := registry.get(execution.tool_call.name)) is not None
                    else None
                ),
                arguments=execution.tool_call.arguments,
                ok=execution.result.ok,
                metadata=execution.result.metadata,
            ),
            succeeded=execution.result.ok,
            tool_name=execution.tool_call.name,
        )
        for execution in batch.executions
    )


def _check_repeated_tool_calls(
    tool_calls: list[AgentToolCall],
    *,
    previous_tool_call_signature: str | None,
    repeated_tool_call_count: int,
    repeated_tool_call_limit: int,
) -> _RepeatedToolCallCheck:
    for tool_call in tool_calls:
        signature = _tool_call_signature(tool_call)
        if signature == previous_tool_call_signature:
            repeated_tool_call_count += 1
        else:
            previous_tool_call_signature = signature
            repeated_tool_call_count = 1

        if repeated_tool_call_count >= repeated_tool_call_limit:
            return _RepeatedToolCallCheck(
                previous_tool_call_signature=previous_tool_call_signature,
                repeated_tool_call_count=repeated_tool_call_count,
                stop_response=AgentModelResponse(
                    content=(
                        "Agent stopped because the model repeated the same tool call "
                        "too many times."
                    ),
                    stop_reason="repeated_tool_call",
                ),
            )

    return _RepeatedToolCallCheck(
        previous_tool_call_signature=previous_tool_call_signature,
        repeated_tool_call_count=repeated_tool_call_count,
    )


def _tool_call_signature(tool_call: AgentToolCall) -> str:
    arguments = json.dumps(
        tool_call.arguments,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return f"{tool_call.name}:{arguments}"


def _validate_tool_batch(
    tool_calls: list[AgentToolCall],
    batch: ToolBatchExecution,
) -> None:
    executed_calls = tuple(execution.tool_call for execution in batch.executions)
    if executed_calls != tuple(tool_calls):
        raise ToolBatchContractError(
            "Tool batch handler must return exactly one ordered result for every "
            "model tool call."
        )
    if (
        batch.stop_response is not None
        and batch.stop_response.stop_reason == "tool_calls"
    ):
        raise ToolBatchContractError(
            "Tool batch stop response must use a terminal stop reason."
        )


def _start_tool_batch_run(handler: ToolBatchHandler) -> None:
    start_run = getattr(handler, "start_run", None)
    if callable(start_run):
        start_run()


def _safe_event_tool_name(tool_name: str) -> str:
    normalized = "".join(
        character
        for character in tool_name[:64]
        if character.isascii()
        and (character.isalnum() or character in {"_", "-"})
    )
    return normalized or "unknown"
