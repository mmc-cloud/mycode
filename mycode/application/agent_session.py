from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from mycode.agent.events import AgentEvent
from mycode.agent.outcome import AgentRunOutcome
from mycode.agent.runner import AgentRunner
from mycode.application.events import RuntimeEvent
from mycode.application.runtime import build_agent_runner, run_agent_turn
from mycode.context.budget import ModelContext
from mycode.error_handling import error_summary
from mycode.application.sessions import (
    ActiveProjectSession,
    SessionStartRequest,
    start_project_session,
)
from mycode.config import LLMConfig
from mycode.mcp import MCPConfig, MCPManager
from mycode.observability import ObservationSink
from mycode.permissions import Confirmer
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity
from mycode.subagents.observability import CompositeSubAgentObserver, SubAgentObserver
from mycode.subagents.persistence import SessionSubAgentObserver


RuntimeEventHandler = Callable[[RuntimeEvent], None]
CompactResultStatus = Literal["compacted", "skipped", "failed"]


@dataclass(frozen=True)
class ContextStatus:
    """Content-free, serializable inspection of the current context."""

    estimated_input_tokens: int
    context_window_tokens: int
    max_input_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    estimate_source: str
    last_provider_prompt_tokens: int | None
    source_message_count: int
    model_visible_message_count: int
    memory_entry_count: int
    memory_estimated_tokens: int
    compact_status: str
    compact_covered_message_count: int
    compressed_tool_result_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "estimated": True,
            "estimated_input_tokens": self.estimated_input_tokens,
            "context_window_tokens": self.context_window_tokens,
            "max_input_tokens": self.max_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "estimate_source": self.estimate_source,
            "last_provider_prompt_tokens": self.last_provider_prompt_tokens,
            "source_message_count": self.source_message_count,
            "model_visible_message_count": self.model_visible_message_count,
            "memory_entry_count": self.memory_entry_count,
            "memory_estimated_tokens": self.memory_estimated_tokens,
            "compact_status": self.compact_status,
            "compact_covered_message_count": self.compact_covered_message_count,
            "compressed_tool_result_count": self.compressed_tool_result_count,
        }


@dataclass(frozen=True)
class CompactResult:
    status: CompactResultStatus
    reason: str | None
    before: ContextStatus | None
    after: ContextStatus | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "before": None if self.before is None else self.before.to_dict(),
            "after": None if self.after is None else self.after.to_dict(),
        }


def _context_status(
    context: ModelContext,
    *,
    context_window_tokens: int,
    reserved_output_tokens: int,
    safety_margin_tokens: int,
    last_provider_prompt_tokens: int | None,
) -> ContextStatus:
    memory = context.memory_stats
    compact = context.compact_stats
    if compact is None or compact.boundary_id is None:
        compact_status = "none"
        compact_covered_message_count = 0
    elif compact.summary_visible:
        compact_status = "active"
        compact_covered_message_count = compact.compacted_message_count
    else:
        compact_status = compact.status
        compact_covered_message_count = compact.compacted_message_count
    return ContextStatus(
        estimated_input_tokens=context.estimate.estimated_input_tokens,
        context_window_tokens=context_window_tokens,
        max_input_tokens=context.estimate.max_input_tokens,
        reserved_output_tokens=reserved_output_tokens,
        safety_margin_tokens=safety_margin_tokens,
        estimate_source=context.estimate.token_estimate_source,
        last_provider_prompt_tokens=last_provider_prompt_tokens,
        source_message_count=context.source_message_count,
        model_visible_message_count=context.selected_message_count,
        memory_entry_count=(0 if memory is None else memory.included_entry_count),
        memory_estimated_tokens=(0 if memory is None else memory.estimated_tokens),
        compact_status=compact_status,
        compact_covered_message_count=compact_covered_message_count,
        compressed_tool_result_count=context.compressed_tool_result_count,
    )


@dataclass
class AgentApplicationSession:
    runner: AgentRunner
    active_project_session: ActiveProjectSession
    mcp_manager: MCPManager
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    @property
    def compact_state_recovered(self) -> bool:
        return self.active_project_session.compact_state_recovered

    @property
    def mcp_statuses(self):
        return self.mcp_manager.statuses

    def get_context_status(self) -> ContextStatus:
        """Recompute current Context statistics without invoking an LLM."""
        self._ensure_active()
        context = self.runner.inspect_context()
        return _context_status(
            context,
            context_window_tokens=self.runner.context_budget.context_window_tokens,
            reserved_output_tokens=self.runner.context_budget.reserved_output_tokens,
            safety_margin_tokens=self.runner.context_budget.safety_margin_tokens,
            last_provider_prompt_tokens=(
                None
                if self.runner.last_token_usage is None
                else self.runner.last_token_usage.prompt_tokens
            ),
        )

    def compact_context(self) -> CompactResult:
        """Run one manual Compact and return a safe structured outcome."""
        self._ensure_active()
        before: ContextStatus | None = None
        try:
            before = self.get_context_status()
            context = self.runner.compact_context()
            compact = context.compact_stats
            if compact is None:
                raise RuntimeError("Compact returned no statistics.")
            operation_status = compact.status
            if operation_status == "compacted":
                result_status: CompactResultStatus = "compacted"
                reason = None
            elif operation_status in {
                "insufficient_history",
                "cooldown",
                "circuit_open",
            }:
                result_status = "skipped"
                reason = operation_status
            elif operation_status in {"failed", "invalid_boundary"}:
                result_status = "failed"
                compactor = getattr(self.runner, "compactor", None)
                reason = (
                    None
                    if compactor is None
                    else compactor.state.last_failure_reason
                ) or operation_status
            else:
                raise RuntimeError(
                    f"Unexpected manual Compact status: {operation_status}."
                )
            after = _context_status(
                context,
                context_window_tokens=self.runner.context_budget.context_window_tokens,
                reserved_output_tokens=self.runner.context_budget.reserved_output_tokens,
                safety_margin_tokens=self.runner.context_budget.safety_margin_tokens,
                last_provider_prompt_tokens=(
                    None
                    if self.runner.last_token_usage is None
                    else self.runner.last_token_usage.prompt_tokens
                ),
            )
            return CompactResult(
                status=result_status,
                reason=reason,
                before=before,
                after=after,
            )
        except Exception as error:  # noqa: BLE001 - application command boundary
            return CompactResult(
                status="failed",
                reason=error_summary(error),
                before=before,
                after=(
                    self._safe_context_status(before)
                    if before is not None
                    else None
                ),
            )

    def _ensure_active(self) -> None:
        if self._cleaned_up:
            raise RuntimeError(
                "AgentApplicationSession is already closed or interrupted."
            )

    def _safe_context_status(self, fallback: ContextStatus) -> ContextStatus:
        try:
            return self.get_context_status()
        except Exception:
            return fallback

    def startup_events(self) -> Iterator[RuntimeEvent]:
        yield RuntimeEvent(
            type="runtime_ready",
            session_id=self.active_project_session.record.id,
            session_title=self.active_project_session.record.title,
            session_created=self.active_project_session.created,
            compact_state_recovered=self.compact_state_recovered,
            instruction_sources=tuple(
                getattr(self.runner, "instruction_sources", ())
            ),
            instruction_warnings=tuple(
                getattr(self.runner, "instruction_warnings", ())
            ),
            skill_warnings=tuple(getattr(self.runner, "skill_warnings", ())),
        )
        for status in self.mcp_statuses:
            yield RuntimeEvent(type="mcp_status", mcp_status=status)

    def run_turn(
        self,
        content: str,
        *,
        turn_id: str | None = None,
        event_handler: RuntimeEventHandler | None = None,
    ) -> AgentRunOutcome:
        if self._cleaned_up:
            raise RuntimeError(
                "AgentApplicationSession is already closed or interrupted."
            )

        def handle_agent_event(agent_event: AgentEvent) -> None:
            if event_handler is not None:
                event_handler(
                    RuntimeEvent(
                        type="agent",
                        turn_id=turn_id,
                        agent_event=agent_event,
                    )
                )

        outcome = run_agent_turn(
            self.runner,
            content,
            event_handler=handle_agent_event,
        )
        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    type="turn_finished",
                    turn_id=turn_id,
                    outcome=outcome,
                )
            )
        return outcome

    def close(self) -> None:
        self._cleanup("closed")

    def interrupt(self) -> None:
        """Finalize this session as interrupted and release its resources.

        This does not asynchronously cancel a currently running Agent turn.
        """
        self._cleanup("interrupted")

    def _cleanup(self, status: Literal["closed", "interrupted"]) -> None:
        if self._cleaned_up:
            return
        try:
            if status == "closed":
                self.active_project_session.close()
            else:
                self.active_project_session.interrupt()
        finally:
            try:
                self.mcp_manager.close()
            finally:
                self._cleaned_up = True


def start_agent_application_session(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    request: SessionStartRequest,
    mcp_config: MCPConfig | None = None,
    confirmer: Confirmer | None = None,
    external_observer: SubAgentObserver | None = None,
    llm_config: LLMConfig | None = None,
    observability_sink: ObservationSink | None = None,
) -> AgentApplicationSession:
    active_session = start_project_session(store, project, request=request)
    mcp_manager: MCPManager | None = None
    try:
        conversation_history = active_session.load_history()
        compact_state = active_session.load_compact_state()
        mcp_manager = MCPManager(
            MCPConfig() if mcp_config is None else mcp_config,
            observability_sink=observability_sink,
        )
        mcp_manager.start()
        session_observer = SessionSubAgentObserver(session=active_session.writer)
        subagent_observer: SubAgentObserver = session_observer
        if external_observer is not None:
            subagent_observer = CompositeSubAgentObserver(
                observers=(session_observer, external_observer),
            )
        runner = build_agent_runner(
            workspace_path=project.workspace_root,
            confirmer=confirmer,
            conversation_history=conversation_history,
            on_message_added=active_session.persist_message,
            compact_state=compact_state,
            on_compact_state_changed=active_session.persist_compact_state,
            artifact_directory=active_session.artifact_directory,
            subagent_observer=subagent_observer,
            llm_config=llm_config,
            llm_session_id=active_session.record.id,
            observability_sink=observability_sink,
        )
        for tool in mcp_manager.tools:
            runner.tool_registry.register(tool)
        return AgentApplicationSession(
            runner=runner,
            active_project_session=active_session,
            mcp_manager=mcp_manager,
        )
    except BaseException:
        if mcp_manager is not None:
            try:
                mcp_manager.close()
            except BaseException:
                pass
        try:
            active_session.interrupt()
        except BaseException:
            pass
        raise
