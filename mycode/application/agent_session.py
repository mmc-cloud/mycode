from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from mycode.agent.events import AgentEvent
from mycode.agent.outcome import AgentRunOutcome
from mycode.agent.runner import AgentRunner
from mycode.application.events import RuntimeEvent
from mycode.application.runtime import build_agent_runner, run_agent_turn
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
