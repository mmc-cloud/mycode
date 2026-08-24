from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from mycode.agent import AgentEvent, AgentModelResponse
from mycode.context_budget import ContextBudget, MemoryContextStats, TokenEstimator, TokenUsage
from mycode.conversation import Conversation
from mycode.instructions import InstructionBundle, load_instruction_bundle
from mycode.llm import LLMClient
from mycode.memory import MemoryStore
from mycode.memory_context import MemoryContextSelector, MemoryRecallPolicy
from mycode.messages import Message
from mycode.permissions import Confirmer, RejectingConfirmer
from mycode.runner import (
    DEFAULT_REPEATED_TOOL_CALL_LIMIT,
    AgentRunner,
)
from mycode.subagents.contracts import (
    SubAgentEnvelopeStatus,
    SubAgentResult,
    SubAgentStopReason,
    SubAgentTask,
)
from mycode.subagents.concurrency import SubAgentInteractionGate
from mycode.subagents.lifecycle import (
    RunTracker,
    StateTransitionHandler,
    SubAgentStateTransition,
    TrackingConfirmer,
)
from mycode.subagents.limits import MAX_VALIDATION_EXECUTIONS
from mycode.subagents.observability import (
    SubAgentObserver,
    SynchronizedSubAgentObserver,
)
from mycode.subagents.profiles import create_subagent_tool_registry, get_agent_profile
from mycode.subagents.prompts import build_subagent_system_prompt
from mycode.subagents.results import (
    DEFAULT_MAX_FINAL_PAYLOAD_CHARS,
)
from mycode.subagents.snapshots import (
    FrozenMemoryRecallProvider,
    SubAgentSnapshotMetadata,
    create_runtime_context_snapshot,
)
from mycode.subagents.tool_batch import SubAgentToolBatchHandler
from mycode.tools.workspace import Workspace


DEFAULT_MAX_VALIDATION_CALLS = 20
DEFAULT_SUBAGENT_MAX_TURNS = 20
DEFAULT_SUBAGENT_CONVERGENCE_REMAINING_TURNS = 3
MAX_RUNTIME_ERROR_CHARS = 2000

InstructionLoader = Callable[[Path, Path], InstructionBundle]
LLMClientFactory = Callable[[], LLMClient]


@dataclass(frozen=True)
class SubAgentModelContextStats:
    estimated_input_tokens: int
    max_input_tokens: int
    selected_message_count: int
    original_message_count: int
    compressed_tool_result_count: int
    memory_stats: MemoryContextStats | None


@dataclass(frozen=True)
class SubAgentExecution:
    result: SubAgentResult
    transitions: tuple[SubAgentStateTransition, ...]
    snapshot: SubAgentSnapshotMetadata | None
    context: SubAgentModelContextStats | None
    token_usage: TokenUsage | None
    conversation_message_count: int
    tool_call_count: int
    validation_execution_count: int


@dataclass(frozen=True)
class SubAgentRuntime:
    workspace: Workspace
    llm_client_factory: LLMClientFactory
    confirmer: Confirmer | None = None
    memory_store: MemoryStore | None = None
    memory_recall_policy: MemoryRecallPolicy = field(default_factory=MemoryRecallPolicy)
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    convergence_remaining_turns: int | None = (
        DEFAULT_SUBAGENT_CONVERGENCE_REMAINING_TURNS
    )
    repeated_tool_call_limit: int = DEFAULT_REPEATED_TOOL_CALL_LIMIT
    max_final_payload_chars: int = DEFAULT_MAX_FINAL_PAYLOAD_CHARS
    max_validation_calls: int = DEFAULT_MAX_VALIDATION_CALLS
    working_directory: Path | None = None
    instruction_loader: InstructionLoader = field(
        default=lambda root, working: load_instruction_bundle(
            root,
            working_directory=working,
        )
    )
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    run_id_factory: Callable[[], str] = field(default=lambda: uuid4().hex)
    interaction_gate: SubAgentInteractionGate = field(
        default_factory=SubAgentInteractionGate,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if self.convergence_remaining_turns is not None and not (
            0 < self.convergence_remaining_turns < self.max_turns
        ):
            raise ValueError(
                "convergence_remaining_turns must be greater than 0 and less than "
                "max_turns."
            )
        if self.repeated_tool_call_limit < 1:
            raise ValueError("repeated_tool_call_limit must be at least 1.")
        if self.max_final_payload_chars < 1:
            raise ValueError("max_final_payload_chars must be at least 1.")
        if self.max_validation_calls < 1:
            raise ValueError("max_validation_calls must be at least 1.")
        if self.max_validation_calls > MAX_VALIDATION_EXECUTIONS:
            raise ValueError(
                "max_validation_calls must not exceed "
                f"{MAX_VALIDATION_EXECUTIONS}."
            )
        if (
            self.memory_store is not None
            and self.memory_store.project.workspace_root != self.workspace.root
        ):
            raise ValueError(
                "memory_store project must match the SubAgent workspace."
            )
        working_directory = self._resolved_working_directory()
        if not working_directory.is_relative_to(self.workspace.root):
            raise ValueError("working_directory must be inside the workspace.")

    def execute(
        self,
        task: SubAgentTask,
        *,
        on_state_transition: StateTransitionHandler | None = None,
        observer: SubAgentObserver | None = None,
    ) -> SubAgentExecution:
        run_observer = (
            None
            if observer is None
            else SynchronizedSubAgentObserver(
                observer,
                self.interaction_gate,
            )
        )
        run_state_transition = (
            None
            if on_state_transition is None
            else lambda transition: self.interaction_gate.run(
                lambda: on_state_transition(transition)
            )
        )
        run_id = self.interaction_gate.run(self.run_id_factory)
        tracker = RunTracker(
            run_id=run_id,
            role=task.role,
            clock=self._now,
            handler=_state_transition_handler(
                task,
                on_state_transition=run_state_transition,
                observer=run_observer,
            ),
        )
        snapshot_metadata: SubAgentSnapshotMetadata | None = None
        runner: AgentRunner | None = None
        batch_handler: SubAgentToolBatchHandler | None = None

        try:
            tracker.transition("running", "run_started")
            instruction_bundle = self.instruction_loader(
                self.workspace.root,
                self._resolved_working_directory(),
            )
            memory_recall = None
            if self.memory_store is not None:
                memory_recall = MemoryContextSelector(
                    self.memory_store,
                    policy=self.memory_recall_policy,
                    token_estimator=TokenEstimator(),
                ).recall(_memory_query(task))
            runtime_snapshot = create_runtime_context_snapshot(
                instruction_bundle,
                memory_recall=memory_recall,
                loaded_at=self._now(),
            )
            snapshot_metadata = runtime_snapshot.metadata
            if run_observer is not None:
                run_observer.on_snapshot(
                    task,
                    run_id,
                    snapshot_metadata,
                    snapshot_metadata.loaded_at,
                )

            profile = get_agent_profile(task.role)
            batch_handler = SubAgentToolBatchHandler(
                role=task.role,
                max_final_payload_chars=self.max_final_payload_chars,
                max_validation_calls=self.max_validation_calls,
                audit_handler=(
                    None
                    if run_observer is None
                    else lambda audit: run_observer.on_tool_audit(
                        task,
                        run_id,
                        audit,
                        self._now(),
                    )
                ),
            )
            tracking_confirmer = TrackingConfirmer(
                delegate=(
                    RejectingConfirmer() if self.confirmer is None else self.confirmer
                ),
                tracker=tracker,
                interaction_gate=self.interaction_gate,
            )
            registry = create_subagent_tool_registry(
                profile,
                self.workspace,
                confirmer=tracking_confirmer,
                result_validator=batch_handler.validate_submission,
            )
            conversation = Conversation.from_messages(
                [
                    Message(
                        role="system",
                        content=build_subagent_system_prompt(
                            profile,
                            registry,
                            project_instructions=runtime_snapshot.project_instructions,
                        ),
                    )
                ]
            )
            runner = AgentRunner(
                llm_client=self.interaction_gate.run(self.llm_client_factory),
                tool_registry=registry,
                conversation=conversation,
                max_turns=self.max_turns,
                convergence_remaining_turns=self.convergence_remaining_turns,
                convergence_prompt=(
                    None
                    if self.convergence_remaining_turns is None
                    else profile.convergence_prompt
                ),
                repeated_tool_call_limit=self.repeated_tool_call_limit,
                readonly_turn_limit=None,
                finalize_on_max_turns=False,
                context_budget=self.context_budget,
                token_estimator=TokenEstimator(),
                memory_context_selector=(
                    None
                    if runtime_snapshot.memory_recall is None
                    else FrozenMemoryRecallProvider(runtime_snapshot.memory_recall)
                ),
                tool_batch_handler=batch_handler,
            )
            response = _collect_agent_events(runner.run(_task_message(task)))
            result = _result_from_response(
                run_id=run_id,
                task=task,
                response=response,
                batch_handler=batch_handler,
            )
            tracker.transition(result.status, result.stop_reason)
        except (KeyboardInterrupt, SystemExit):
            tracker.transition_once("interrupted", "interrupted")
            raise
        except Exception as error:
            result = _failed_result(
                run_id=run_id,
                task=task,
                status="failed",
                stop_reason="runtime_error",
                summary="SubAgent runtime failed before producing a valid result.",
                error=_safe_error(error),
            )
            tracker.transition_once("failed", "runtime_error")

        execution = SubAgentExecution(
            result=result,
            transitions=tuple(tracker.transitions),
            snapshot=snapshot_metadata,
            context=_context_stats(runner),
            token_usage=None if runner is None else runner.run_token_usage,
            conversation_message_count=(
                0 if runner is None else len(runner.conversation.get_messages())
            ),
            tool_call_count=0 if batch_handler is None else batch_handler.tool_call_count,
            validation_execution_count=(
                0
                if batch_handler is None
                else len(batch_handler.validation_executions)
            ),
        )
        if run_observer is not None:
            run_observer.on_result(task, execution, self._now())
        return execution

    def _resolved_working_directory(self) -> Path:
        if self.working_directory is None:
            return self.workspace.root
        path = self.working_directory
        if not path.is_absolute():
            path = self.workspace.root / path
        return path.resolve(strict=False)

    def _now(self) -> datetime:
        return self.interaction_gate.run(self.clock)


def _state_transition_handler(
    task: SubAgentTask,
    *,
    on_state_transition: StateTransitionHandler | None,
    observer: SubAgentObserver | None,
) -> StateTransitionHandler | None:
    if on_state_transition is None and observer is None:
        return None

    def handle(transition: SubAgentStateTransition) -> None:
        if on_state_transition is not None:
            on_state_transition(transition)
        if observer is not None:
            observer.on_state(task, transition)

    return handle


def _collect_agent_events(events: Iterator[AgentEvent]) -> AgentModelResponse:
    current_turn_content: list[str] = []
    current_error: str | None = None
    final_content = ""
    stop_reason = None

    for event in events:
        if event.type in {"turn", "model_start"}:
            current_turn_content = []
            current_error = None
        elif event.type == "text_delta":
            current_turn_content.append(event.content)
        elif event.type == "error":
            if event.error is not None:
                current_error = event.error
        elif event.type == "stop":
            stop_reason = event.stop_reason or "model_error"
            final_content = "".join(current_turn_content)
            if final_content:
                continue
            if stop_reason in {"model_error", "context_overflow"} and current_error:
                final_content = current_error
            elif event.content:
                final_content = event.content

    return AgentModelResponse(
        content=final_content,
        stop_reason=stop_reason or "model_error",
    )


def _result_from_response(
    *,
    run_id: str,
    task: SubAgentTask,
    response: AgentModelResponse,
    batch_handler: SubAgentToolBatchHandler,
) -> SubAgentResult:
    if response.stop_reason == "control_tool" and batch_handler.submitted_payload is not None:
        return SubAgentResult(
            run_id=run_id,
            role=task.role,
            status="completed",
            stop_reason="submitted",
            summary=batch_handler.submitted_payload.summary,
            payload=batch_handler.submitted_payload,
        )

    if response.stop_reason == "max_turns":
        stop_reason: SubAgentStopReason = (
            "invalid_result" if batch_handler.submission_attempted else "max_turns"
        )
    elif response.stop_reason in {
        "model_error",
        "context_overflow",
        "repeated_tool_call",
    }:
        stop_reason = response.stop_reason
    else:
        stop_reason = "invalid_result"

    summary = {
        "max_turns": "SubAgent reached its maximum number of turns.",
        "invalid_result": "SubAgent did not submit a valid structured result.",
        "model_error": "SubAgent model request failed.",
        "context_overflow": "SubAgent context exceeded its configured budget.",
        "repeated_tool_call": "SubAgent repeated the same tool call too many times.",
    }[stop_reason]
    return _failed_result(
        run_id=run_id,
        task=task,
        status="failed",
        stop_reason=stop_reason,
        summary=summary,
        error=(
            _omitted_model_content(response.content, fallback=summary)
            if stop_reason == "invalid_result"
            else _safe_text(response.content, fallback=summary)
        ),
    )


def _failed_result(
    *,
    run_id: str,
    task: SubAgentTask,
    status: SubAgentEnvelopeStatus,
    stop_reason: SubAgentStopReason,
    summary: str,
    error: str,
) -> SubAgentResult:
    return SubAgentResult(
        run_id=run_id,
        role=task.role,
        status=status,
        stop_reason=stop_reason,
        summary=summary,
        error=error,
    )


def _context_stats(runner: AgentRunner | None) -> SubAgentModelContextStats | None:
    if runner is None or runner.last_model_context is None:
        return None
    context = runner.last_model_context
    return SubAgentModelContextStats(
        estimated_input_tokens=context.estimate.estimated_input_tokens,
        max_input_tokens=context.estimate.max_input_tokens,
        selected_message_count=context.selected_message_count,
        original_message_count=context.original_message_count,
        compressed_tool_result_count=context.compressed_tool_result_count,
        memory_stats=context.memory_stats,
    )


def _task_message(task: SubAgentTask) -> str:
    return (
        "Complete this delegated task and submit the role-specific structured result.\n\n"
        "<delegated_task_json>\n"
        + json.dumps(task.model_dump(), ensure_ascii=False, indent=2)
        + "\n</delegated_task_json>"
    )


def _memory_query(task: SubAgentTask) -> str:
    return "\n".join(
        part
        for part in (
            task.objective,
            task.context,
            " ".join(task.scope_paths),
        )
        if part
    )


def _safe_error(error: Exception) -> str:
    return _safe_text(
        f"{type(error).__name__}: {error}",
        fallback=f"{type(error).__name__} with no safe message.",
    )


def _omitted_model_content(content: str, *, fallback: str) -> str:
    normalized = content.strip()
    if not normalized:
        return fallback
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (
        f"{fallback} Raw assistant content was not returned to the parent "
        f"({len(normalized)} characters, sha256={digest})."
    )


def _safe_text(content: str, *, fallback: str) -> str:
    normalized = content.strip()
    if not normalized:
        return fallback
    if len(normalized) <= MAX_RUNTIME_ERROR_CHARS:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (
        f"{fallback} Original message omitted because it exceeded the runtime "
        f"error limit ({len(normalized)} characters, sha256={digest})."
    )
