from dataclasses import dataclass, field, replace
from collections.abc import Callable, Generator, Iterator, Sequence
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
    artifact_reference_info,
)
from mycode.context_compact import ConversationCompactor
from mycode.tool_result_retention import ToolResultRetentionPolicy, TurnLocalFullGroup
from mycode.context_builder import ContextBuilder
from mycode.error_handling import format_model_error
from mycode.context_budget import (
    ContextBudget,
    ModelContext,
    TokenEstimator,
    TokenUsage,
    format_model_context_stats,
    model_context_needs_notice,
)
from mycode.conversation import Conversation
from mycode.llm import LLMClient
from mycode.memory_context import MemoryRecall, MemoryRecallProvider
from mycode.messages import Message
from mycode.observability import ObservationSink, emit_observation
from mycode.skills import ActiveSkillState
from mycode.reasoning import ReasoningState
from mycode.run_progress import (
    DEFAULT_MAX_TURNS,
    MAIN_NEAR_LIMIT_PROMPT,
    MAIN_NEAR_LIMIT_REMAINING_TURNS,
    MAX_TURNS_FINALIZATION_PROMPT,
    PolicyDecision,
    RuntimeObservation,
    RuntimePolicy,
    RuntimeState,
    decide_runtime_policy,
    observe_tool_result,
    normalize_run_checkpoint,
    resume_guidance,
)
from mycode.tools import ToolRegistry, ToolResult, Workspace


DEFAULT_REPEATED_TOOL_CALL_LIMIT = 3
DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE = 32
DEFAULT_MAX_CONCURRENT_SAFE_TOOLS = 4
EMPTY_RESPONSE_RETRY_PROMPT = (
    "Your previous response contained neither tool calls nor a final answer. "
    "Continue the task or provide a final response."
)
EMPTY_RESPONSE_ERROR = (
    "模型响应错误（empty_response）：连续两次响应都没有工具调用或非空最终回答。"
)
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
    near_limit_remaining_turns: int | None = MAIN_NEAR_LIMIT_REMAINING_TURNS
    near_limit_prompt: str | None = MAIN_NEAR_LIMIT_PROMPT
    repeated_tool_call_limit: int = DEFAULT_REPEATED_TOOL_CALL_LIMIT
    finalize_on_max_turns: bool = True
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    token_estimator: TokenEstimator = field(default_factory=TokenEstimator)
    last_model_context: ModelContext | None = field(default=None, init=False)
    last_token_usage: TokenUsage | None = field(default=None, init=False)
    run_token_usage: TokenUsage | None = field(default=None, init=False)
    last_reasoning_char_count: int = field(default=0, init=False)
    instruction_sources: tuple[str, ...] = ()
    instruction_warnings: tuple[str, ...] = ()
    skill_warnings: tuple[str, ...] = ()
    active_skill_state: ActiveSkillState | None = None
    memory_context_selector: MemoryRecallProvider | None = None
    last_memory_recall: MemoryRecall | None = field(default=None, init=False)
    tool_batch_handler: ToolBatchHandler = execute_tool_batch
    compactor: ConversationCompactor | None = None
    tool_result_artifact_store: ToolResultArtifactStore | None = None
    observability_sink: ObservationSink | None = field(default=None, repr=False)
    observability_scope: str = "main"
    observability_run_id: str | None = None
    last_artifact_error: str | None = field(default=None, init=False)
    artifact_failure_count: int = field(default=0, init=False)
    last_runtime_state: RuntimeState | None = field(default=None, init=False)
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
        if (self.near_limit_remaining_turns is None) != (
            self.near_limit_prompt is None
        ):
            raise ValueError(
                "near_limit_remaining_turns and near_limit_prompt must both be set "
                "or both be None."
            )
        if self.near_limit_remaining_turns is None:
            return
        if not 0 < self.near_limit_remaining_turns < self.max_turns:
            raise ValueError(
                "near_limit_remaining_turns must be greater than 0 and less than "
                "max_turns."
            )
        if not self.near_limit_prompt or not self.near_limit_prompt.strip():
            raise ValueError("near_limit_prompt must not be blank.")

    def run(self, user_message: str) -> Iterator[AgentEvent]:
        pending_full_group: TurnLocalFullGroup | None = None
        self._pending_artifact_warnings.clear()
        try:
            if self.active_skill_state is not None:
                self.active_skill_state.clear()
            self.run_token_usage = None
            _start_tool_batch_run(self.tool_batch_handler)
            continuation_guidance = resume_guidance(self.conversation, user_message)
            self._recall_memory(user_message)
            self.conversation.add_user_message(user_message)

            previous_tool_call_signature: str | None = None
            repeated_tool_call_count = 0
            pending_runtime_decision = PolicyDecision(
                policy=RuntimePolicy.NO_INTERVENTION
            )
            near_limit_guidance_sent = False
            runtime_state = RuntimeState()
            self.last_runtime_state = runtime_state
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
                    int,
                    int,
                ]
                | None
            ) = None

            for turn_index in range(self.max_turns):
                guidance = pending_runtime_decision.guidance
                notice = pending_runtime_decision.notice
                pending_runtime_decision = PolicyDecision(
                    policy=RuntimePolicy.NO_INTERVENTION
                )
                if not guidance and turn_index == 0 and continuation_guidance:
                    guidance = (continuation_guidance,)
                    notice = "已载入上次检查点，将直接衔接剩余工作"
                remaining_turns = self.max_turns - turn_index
                if (
                    self.near_limit_remaining_turns is not None
                    and self.near_limit_prompt is not None
                    and remaining_turns == self.near_limit_remaining_turns
                    and not near_limit_guidance_sent
                ):
                    guidance = (*guidance, self.near_limit_prompt)
                    near_limit_guidance_sent = True
                    near_limit_notice = (
                        f"距离运行轮次上限还有 {self.near_limit_remaining_turns} 轮，"
                        "已发送一次接近上限提醒"
                    )
                    notice = (
                        near_limit_notice
                        if not notice
                        else f"{notice}；{near_limit_notice}"
                    )
                tools = self.tool_registry.get_schemas()
                yield AgentEvent(
                    type="turn",
                    content=notice,
                    turn_number=turn_index + 1,
                    max_turns=self.max_turns,
                )
                turn_local_full_group = pending_full_group
                pending_full_group = None
                for response_attempt in range(2):
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    reasoning_state: ReasoningState = "absent"
                    tool_calls: list[AgentToolCall] = []
                    request_guidance = guidance
                    if response_attempt == 1:
                        request_guidance = (
                            *request_guidance,
                            EMPTY_RESPONSE_RETRY_PROMPT,
                        )
                    model_context = self._model_context(
                        tools,
                        guidance=request_guidance,
                        turn_local_full_group=turn_local_full_group,
                        observability_turn=turn_index + 1,
                    )
                    for warning in self._drain_artifact_warnings():
                        yield AgentEvent(
                            type="artifact_warning",
                            content=warning.content,
                        )

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

                    self._emit_context_snapshot(
                        model_context,
                        turn=turn_index + 1,
                        call_kind="agent_tools",
                    )

                    if model_context.estimate.over_budget:
                        yield AgentEvent(
                            type="error",
                            error=_context_overflow_message(model_context),
                        )
                        yield AgentEvent(
                            type="stop",
                            stop_reason="context_overflow",
                        )
                        return

                    yield AgentEvent(type="model_start")

                    try:
                        for event in self.llm_client.stream_with_tools(
                            Conversation.from_messages(
                                list(model_context.messages)
                            ),
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
                                yield event
                                continue

                            if (
                                event.type == "tool_call"
                                and event.tool_call is not None
                            ):
                                tool_calls.append(event.tool_call)
                                observable_event = replace(
                                    event,
                                    tool_call=_observable_tool_call(
                                        self.tool_registry,
                                        event.tool_call,
                                    ),
                                )
                                yield observable_event
                                continue

                            if event.type == "error":
                                self._observe_token_usage(model_context)
                                self._emit_model_response(
                                    turn=turn_index + 1,
                                    call_kind="agent_tools",
                                    content="".join(content_parts),
                                    tool_calls=tool_calls,
                                )
                                yield event
                                yield AgentEvent(
                                    type="stop",
                                    stop_reason="model_error",
                                )
                                return
                    except Exception as error:
                        self._emit_model_response(
                            turn=turn_index + 1,
                            call_kind="agent_tools",
                            content="".join(content_parts),
                            tool_calls=tool_calls,
                            fallback_error_type=type(error).__name__,
                        )
                        yield AgentEvent(
                            type="error",
                            error=format_model_error(
                                error,
                                operation="模型流式请求失败",
                            ),
                        )
                        yield AgentEvent(
                            type="stop",
                            stop_reason="model_error",
                        )
                        return

                    self._observe_token_usage(model_context)
                    content = "".join(content_parts)
                    empty_response = not tool_calls and not content.strip()
                    terminal_empty_response = (
                        empty_response and response_attempt == 1
                    )
                    self._emit_model_response(
                        turn=turn_index + 1,
                        call_kind="agent_tools",
                        content=content,
                        tool_calls=tool_calls,
                        error_type_override=(
                            "empty_response"
                            if terminal_empty_response
                            else None
                        ),
                    )
                    if not empty_response:
                        break
                    if response_attempt == 0:
                        continue
                    yield AgentEvent(
                        type="error",
                        error=EMPTY_RESPONSE_ERROR,
                    )
                    yield AgentEvent(
                        type="stop",
                        stop_reason="model_error",
                    )
                    return

                turn_local_full_group = None
                if not tool_calls:
                    runtime_state.last_reason = "final_answer"
                    self.conversation.add_assistant_message(content)
                    yield _progress_event(runtime_state)
                    yield AgentEvent(type="stop", stop_reason="final_answer")
                    return

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
                self.conversation.add_assistant_tool_calls(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content="".join(reasoning_parts) or None,
                    reasoning_state=reasoning_state,
                )

                batch = self.tool_batch_handler(self.tool_registry, tool_calls)
                _validate_tool_batch(tool_calls, batch)
                _observe_tool_turn_progress(
                    runtime_state,
                    registry=self.tool_registry,
                    batch=batch,
                )
                pending_runtime_decision = decide_runtime_policy(runtime_state)
                pending_full_group = yield from self._persist_tool_results(batch)
                yield _progress_event(runtime_state)
                if batch.stop_response is not None:
                    pending_full_group = None
                    yield AgentEvent(
                        type="stop",
                        content=batch.stop_response.content,
                        stop_reason=batch.stop_response.stop_reason,
                    )
                    return
                del batch

            yield _progress_event(runtime_state)
            if self.finalize_on_max_turns:
                finalization_events = self._stream_finalization_after_max_turns(
                    turn_local_full_group=pending_full_group,
                )
                pending_full_group = None
                yield from finalization_events
                return
            pending_full_group = None
            yield AgentEvent(
                type="stop",
                content="Agent stopped because it reached the maximum number of turns.",
                stop_reason="max_turns",
            )

        finally:
            pending_full_group = None
            self._pending_artifact_warnings.clear()
            if self.active_skill_state is not None:
                self.active_skill_state.clear()

    def _stream_finalization_after_max_turns(
        self, *, turn_local_full_group: TurnLocalFullGroup | None = None,
    ) -> Iterator[AgentEvent]:
        try:
            finalization, model_context = self._max_turns_finalization_context(
                turn_local_full_group=turn_local_full_group,
            )
        finally:
            turn_local_full_group = None
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
        self._emit_context_snapshot(
            model_context,
            turn=self.max_turns + 1,
            call_kind="max_turns_finalization",
        )
        yield AgentEvent(type="model_start")
        try:
            for chunk in self.llm_client.stream_complete(finalization):
                if chunk == "":
                    continue
                content_parts.append(chunk)
                yield AgentEvent(type="text_delta", content=chunk)
        except Exception as error:
            self._emit_model_response(
                turn=self.max_turns + 1,
                call_kind="max_turns_finalization",
                content="".join(content_parts),
                tool_calls=(),
                fallback_error_type=type(error).__name__,
            )
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
        self._emit_model_response(
            turn=self.max_turns + 1,
            call_kind="max_turns_finalization",
            content="".join(content_parts),
            tool_calls=(),
        )
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
        self, *, turn_local_full_group: TurnLocalFullGroup | None = None,
    ) -> tuple[Conversation | None, ModelContext]:
        finalization_context = self._model_context(
            [],
            turn_local_full_group=turn_local_full_group,
            observability_turn=self.max_turns + 1,
            request_messages=(
                Message(role="user", content=MAX_TURNS_FINALIZATION_PROMPT),
            ),
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
        request_messages: tuple[Message, ...] = (),
        turn_local_full_group: TurnLocalFullGroup | None = None,
        observability_turn: int | None = None,
    ) -> ModelContext:
        memory_recall = self.last_memory_recall
        result = ContextBuilder(
            budget=self.context_budget,
            token_estimator=self.token_estimator,
            compactor=self.compactor,
            retention_policy=ToolResultRetentionPolicy(
                self.context_budget, self.tool_result_artifact_store,
                self._historical_artifact_failure_content,
            ),
        ).build(
            self.conversation,
            tools=tools,
            memory_message=None if memory_recall is None else memory_recall.message,
            memory_stats=None if memory_recall is None else memory_recall.stats,
            guidance=guidance,
            request_messages=request_messages,
            persistent_system_messages=(
                ()
                if self.active_skill_state is None
                else tuple(
                    Message(role="system", content=content)
                    for content in self.active_skill_state.to_system_contexts()
                )
            ),
            turn_local_full_group=turn_local_full_group,
            observability_turn=observability_turn,
        )
        self._accumulate_run_token_usage(result.compact_attempt_token_usage)
        self.last_model_context = result.context
        return result.context

    def _persist_tool_results(
        self, batch: ToolBatchExecution,
    ) -> Generator[AgentEvent, None, TurnLocalFullGroup | None]:
        """Persist immediately, handing off only this batch's successfully stored Fulls."""
        assistant = self.conversation.get_messages()[-1]
        results: dict[str, tuple[str, str]] = {}
        externalized_count = 0
        for execution in batch.executions:
            yield AgentEvent(type="tool_result", tool_result=execution.result)
            content = format_tool_result(execution.result)
            persisted_content = self._tool_result_content(execution, content=content)
            for warning in self._drain_artifact_warnings():
                yield AgentEvent(type="artifact_warning", content=warning.content)
            self.conversation.add_tool_result_message(
                tool_call_id=execution.tool_call.id,
                content=persisted_content,
            )
            artifact_reference = (
                artifact_reference_info(
                    tool_name=execution.tool_call.name,
                    tool_call_id=execution.tool_call.id,
                    content=persisted_content,
                ) is not None
            )
            if persisted_content != content and artifact_reference:
                externalized_count += 1
            if (
                self.context_budget.recent_tool_result_groups_to_keep > 0
                and persisted_content != content
                and artifact_reference
            ):
                results[execution.tool_call.id] = (persisted_content, content)
        if not results and externalized_count == 0:
            return None
        return TurnLocalFullGroup(
            assistant,
            results,
            externalized_count=externalized_count,
        )

    def _tool_result_content(self, execution: ToolCallExecution, *, content: str) -> str:
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

    def _emit_model_response(
        self,
        *,
        turn: int | None,
        call_kind: str,
        content: str,
        tool_calls: Sequence[AgentToolCall],
        fallback_error_type: str | None = None,
        error_type_override: str | None = None,
    ) -> None:
        observation = getattr(self.llm_client, "last_model_response", None)
        if not isinstance(observation, dict):
            usage = getattr(self.llm_client, "last_token_usage", None)
            observation = {
                "model": getattr(self.llm_client, "model", None),
                "request_id": None,
                "provider_request_id": None,
                "finish_reason": None,
                "stop_reason": None,
                "content_chars": len(content),
                "content_non_whitespace_chars": sum(
                    not character.isspace() for character in content
                ),
                "tool_call_count": len(tool_calls),
                "tool_names": [call.name for call in tool_calls],
                "reasoning_field_present": False,
                "reasoning_chars": getattr(
                    self.llm_client,
                    "last_reasoning_char_count",
                    0,
                ),
                "prompt_tokens": None if usage is None else usage.prompt_tokens,
                "completion_tokens": (
                    None if usage is None else usage.completion_tokens
                ),
                "total_tokens": None if usage is None else usage.total_tokens,
                "latency_ms": None,
                "first_token_latency_ms": None,
                "stream_chunk_count": None,
                "retry_count": None,
                "error_type": error_type_override or fallback_error_type,
                "http_status": None,
                "empty_response": not content.strip() and not tool_calls,
            }
        elif error_type_override is not None:
            observation = {
                **observation,
                "error_type": error_type_override,
            }
        emit_observation(
            self.observability_sink,
            "model_response",
            {
                "run_scope": self.observability_scope,
                "run_id": self.observability_run_id,
                "turn": turn,
                "call_kind": call_kind,
                **observation,
            },
        )

    def _emit_context_snapshot(
        self,
        context: ModelContext,
        *,
        turn: int | None,
        call_kind: str,
    ) -> None:
        compact = context.compact_stats
        retention = context.retention_stats
        active_skill_names = (
            []
            if self.active_skill_state is None
            else [skill.name for skill in self.active_skill_state.get_active()]
        )
        emit_observation(
            self.observability_sink,
            "context_snapshot",
            {
                "run_scope": self.observability_scope,
                "run_id": self.observability_run_id,
                "turn": turn,
                "call_kind": call_kind,
                "estimated_input_tokens": context.estimate.estimated_input_tokens,
                "max_input_tokens": context.estimate.max_input_tokens,
                "selected_message_count": context.selected_message_count,
                "source_message_count": context.source_message_count,
                "over_budget": context.estimate.over_budget,
                "active_skill_count": len(active_skill_names),
                "active_skill_names": active_skill_names,
                "turn_local_full_hit": (
                    0 if retention is None else retention.turn_local_full_groups
                ),
                "artifact_rehydrate_count": (
                    0 if retention is None else retention.artifact_rehydrate_count
                ),
                "artifact_externalized_count": (
                    0 if retention is None else retention.artifact_externalized_count
                ),
                "compact_triggered": (
                    compact is not None and compact.status == "compacted"
                ),
                "compact_status": None if compact is None else compact.status,
                "retention": {
                    "full_groups": 0 if retention is None else retention.full_groups,
                    "turn_local_full_groups": (
                        0 if retention is None else retention.turn_local_full_groups
                    ),
                    "artifact_rehydrated_groups": (
                        0
                        if retention is None
                        else retention.artifact_rehydrated_groups
                    ),
                    "artifact_groups": (
                        0 if retention is None else retention.artifact_groups
                    ),
                    "metadata_groups": (
                        0 if retention is None else retention.metadata_groups
                    ),
                    "budget_downgraded_groups": (
                        0
                        if retention is None
                        else retention.budget_downgraded_groups
                    ),
                    "rehydration_failures": (
                        0 if retention is None else retention.rehydration_failures
                    ),
                },
            },
        )

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


def _progress_event(runtime_state: RuntimeState) -> AgentEvent:
    return AgentEvent(
        type="progress",
        progress=AgentProgressSnapshot(
            stagnation_turns=runtime_state.stagnation_turns,
            same_tool_repeat=runtime_state.same_tool_repeat,
            same_result_repeat=runtime_state.same_result_repeat,
            resource_repeat=runtime_state.resource_repeat,
            convergence_guided=runtime_state.convergence_guided,
            reason=runtime_state.last_reason,
        ),
    )


def _context_state(
    context: ModelContext,
) -> tuple[int, int, bool, int, int, str | None, bool, str | None, int, int, int]:
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
        0 if context.retention_stats is None else context.retention_stats.budget_downgraded_groups,
        0 if context.retention_stats is None else context.retention_stats.rehydration_failures,
    )


def _context_overflow_message(context: ModelContext) -> str:
    return (
        "Model request was not sent because the estimated input context exceeds "
        f"the configured budget ({context.estimate.estimated_input_tokens}/"
        f"{context.estimate.max_input_tokens} tokens)."
    )


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


def _observe_tool_turn_progress(
    runtime_state: RuntimeState,
    *,
    registry: ToolRegistry,
    batch: ToolBatchExecution,
) -> None:
    runtime_state.observe_tool_turn(
        _batch_tool_observations(registry, batch),
    )


def _batch_tool_observations(
    registry: ToolRegistry,
    batch: ToolBatchExecution,
) -> tuple[RuntimeObservation, ...]:
    observations: list[RuntimeObservation] = []
    for execution in batch.executions:
        tool = registry.get(execution.tool_call.name)
        workspace = None if tool is None else getattr(tool, "workspace", None)
        observations.append(
            observe_tool_result(
                tool_name=execution.tool_call.name,
                capability=(
                    tool.get_permission_profile().capability
                    if tool is not None
                    else None
                ),
                arguments=execution.tool_call.arguments,
                ok=execution.result.ok,
                content=execution.result.content,
                error=execution.result.error,
                metadata=execution.result.metadata,
                workspace=workspace if isinstance(workspace, Workspace) else None,
            )
        )
    return tuple(observations)


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
