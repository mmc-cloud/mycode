"""Project-scoped application startup ownership.

This module prepares stable process/project dependencies once and creates
session-scoped application sessions on demand.  It deliberately has no
knowledge of CLI, TUI, JSONL, or any other presentation protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mycode.application.agent_session import (
    AgentApplicationSession,
    start_agent_application_session,
)
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig, load_llm_config
from mycode.mcp.config import MCPConfig, MCPConfigError, load_mcp_config_layers
from mycode.mcp.trust import MCPTrustConfirmer, resolve_project_mcp_trust
from mycode.observability import ObservationSink
from mycode.permissions import Confirmer
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity
from mycode.subagents.observability import SubAgentObserver
from mycode.tools.workspace import Workspace


@dataclass(frozen=True)
class ApplicationEnvironment:
    """Stable project/process dependencies shared by application sessions."""

    workspace: Workspace
    project: ProjectIdentity
    session_store: SessionStore


@dataclass(frozen=True)
class ApplicationStartupWarning:
    """A presentation-neutral warning produced while preparing startup."""

    code: str
    message: str


@dataclass
class ApplicationSessionFactory:
    """Project-scoped dependencies and a factory for candidate sessions."""

    environment: ApplicationEnvironment
    llm_config: LLMConfig
    resolved_mcp_config: MCPConfig
    confirmer: Confirmer | None = None
    external_observer: SubAgentObserver | None = None
    observability_sink: ObservationSink | None = None

    def open_session(self, request: SessionStartRequest) -> AgentApplicationSession:
        """Create one independent AgentApplicationSession for ``request``."""
        return start_agent_application_session(
            self.environment.session_store,
            self.environment.project,
            request=request,
            mcp_config=self.resolved_mcp_config,
            confirmer=self.confirmer,
            external_observer=self.external_observer,
            llm_config=self.llm_config,
            observability_sink=self.observability_sink,
        )


def create_application_environment(
    workspace_path: Path | None = None,
    *,
    session_store: SessionStore | None = None,
) -> ApplicationEnvironment:
    """Create the stable project/process environment for one frontend."""
    workspace_root = Path.cwd() if workspace_path is None else workspace_path
    workspace = Workspace(workspace_root)
    return ApplicationEnvironment(
        workspace=workspace,
        project=ProjectIdentity.from_workspace(workspace.root),
        session_store=SessionStore() if session_store is None else session_store,
    )


def prepare_application_session_factory(
    environment: ApplicationEnvironment,
    *,
    llm_config: LLMConfig | None = None,
    mcp_config: MCPConfig | None = None,
    mcp_trust_confirmer: MCPTrustConfirmer | None = None,
    trust_file: str | Path | None = None,
    confirmer: Confirmer | None = None,
    external_observer: SubAgentObserver | None = None,
    observability_sink: ObservationSink | None = None,
    warning_handler: Callable[[ApplicationStartupWarning], None] | None = None,
) -> ApplicationSessionFactory:
    """Resolve process-scoped dependencies and return a session factory.

    Explicit configs are treated as already-resolved snapshots.  The default
    MCP path loads and resolves trust exactly once here; sessions created by
    the returned factory reuse that snapshot while still getting their own
    MCPManager in ``start_agent_application_session``.
    """
    effective_llm_config = resolve_application_llm_config(
        environment,
        llm_config=llm_config,
    )

    if mcp_config is not None:
        effective_mcp_config = mcp_config
    else:
        try:
            loaded = load_mcp_config_layers(
                workspace_root=environment.workspace.root
            )
        except MCPConfigError as error:
            if warning_handler is not None:
                warning_handler(
                    ApplicationStartupWarning(
                        code="mcp_config_error",
                        message=str(error),
                    )
                )
            effective_mcp_config = MCPConfig()
        else:
            if (
                mcp_trust_confirmer is None
                and getattr(loaded, "project").mcp_servers
            ):
                raise ValueError(
                    "mcp_trust_confirmer is required when MCP config is not supplied."
                )
            effective_mcp_config = resolve_project_mcp_trust(
                loaded,
                environment.project,
                confirmer=mcp_trust_confirmer,
                trust_file=trust_file,
            ).config

    return ApplicationSessionFactory(
        environment=environment,
        llm_config=effective_llm_config,
        resolved_mcp_config=effective_mcp_config,
        confirmer=confirmer,
        external_observer=external_observer,
        observability_sink=observability_sink,
    )


def resolve_application_llm_config(
    environment: ApplicationEnvironment,
    *,
    llm_config: LLMConfig | None = None,
) -> LLMConfig:
    """Resolve one application-scoped LLM configuration snapshot."""
    if llm_config is not None:
        return llm_config
    return load_llm_config(workspace_root=environment.workspace.root)


__all__ = [
    "ApplicationEnvironment",
    "ApplicationSessionFactory",
    "ApplicationStartupWarning",
    "create_application_environment",
    "prepare_application_session_factory",
    "resolve_application_llm_config",
]
