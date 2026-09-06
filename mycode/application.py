"""Application-layer use cases shared by interactive and automated frontends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mycode.agent import AgentEvent, AgentStopReason
from mycode.artifacts import (
    ArtifactWriteGuard,
    ReadArtifactTool,
    ToolResultArtifactStore,
)
from mycode.config import LLMConfig, load_llm_config
from mycode.context_budget import ContextBudget
from mycode.context_compact import CompactState, ConversationCompactor
from mycode.conversation import Conversation
from mycode.instructions import load_instruction_bundle
from mycode.llm import OpenAICompatibleLLMClient
from mycode.memory import MemoryStore
from mycode.memory_context import MemoryContextSelector, MemoryRecallPolicy
from mycode.messages import Message
from mycode.observability import ObservationSink
from mycode.permissions import Confirmer
from mycode.project import ProjectIdentity
from mycode.prompts import build_agent_system_prompt
from mycode.run_outcome import AgentRunOutcome
from mycode.runner import AgentRunner
from mycode.skills import ActiveSkillState, SkillRegistry
from mycode.subagents.delegate import DelegateTaskTool
from mycode.subagents.delegation import DelegationToolBatchHandler
from mycode.subagents.observability import SubAgentObserver
from mycode.subagents.runtime import SubAgentRuntime
from mycode.tools import (
    LoadSkillTool,
    ReadSkillResourceTool,
    RunSkillScriptTool,
    Workspace,
    create_default_tool_registry,
)


AgentEventHandler = Callable[[AgentEvent], None]


def build_agent_runner(
    workspace_path: Path | None = None,
    *,
    confirmer: Confirmer | None = None,
    conversation_history: Conversation | None = None,
    on_message_added: Callable[[Message], None] | None = None,
    compact_state: CompactState | None = None,
    on_compact_state_changed: Callable[[CompactState], None] | None = None,
    artifact_directory: Path | None = None,
    artifact_write_guard: ArtifactWriteGuard | None = None,
    memory_store: MemoryStore | None = None,
    subagent_observer: SubAgentObserver | None = None,
    llm_config: LLMConfig | None = None,
    observability_sink: ObservationSink | None = None,
) -> AgentRunner:
    """Assemble one Agent runtime without coupling it to a presentation layer."""
    if (artifact_directory is None) != (artifact_write_guard is None):
        raise ValueError(
            "artifact_directory and artifact_write_guard must be provided together."
        )
    workspace_root = Path.cwd() if workspace_path is None else workspace_path
    workspace = Workspace(workspace_root)
    project = ProjectIdentity.from_workspace(workspace.root)
    effective_memory_store = (
        MemoryStore(project) if memory_store is None else memory_store
    )
    instruction_bundle = load_instruction_bundle(workspace.root)
    skill_registry = SkillRegistry.discover(workspace.root)
    active_skill_state = ActiveSkillState()
    config = (
        load_llm_config(workspace_root=workspace.root)
        if llm_config is None
        else llm_config
    )
    client = OpenAICompatibleLLMClient(config=config)
    summary_client = OpenAICompatibleLLMClient(
        config=config,
        model=config.compact_model,
        thinking_enabled=False,
    )
    context_budget = context_budget_from_config(config)
    memory_recall_policy = MemoryRecallPolicy(
        max_tokens=config.memory_context_tokens,
    )
    subagent_runtime = SubAgentRuntime(
        workspace=workspace,
        llm_client_factory=lambda: OpenAICompatibleLLMClient(
            config=config,
            model=config.subagent_model,
        ),
        confirmer=confirmer,
        memory_store=effective_memory_store,
        memory_recall_policy=memory_recall_policy,
        context_budget=context_budget,
        observability_sink=observability_sink,
    )
    history_messages = (
        [] if conversation_history is None else conversation_history.get_messages()
    )
    if any(message.role == "system" for message in history_messages):
        raise ValueError("conversation_history must not contain system messages.")
    conversation = Conversation.from_messages(
        [
            Message(
                role="system",
                content=build_agent_system_prompt(
                    instruction_bundle.to_prompt_text(),
                    memory_enabled=True,
                    delegation_enabled=True,
                    skill_catalog=skill_registry.get_catalog(),
                ),
            ),
            *history_messages,
        ],
        on_message_added=on_message_added,
    )
    artifact_store = (
        None
        if artifact_directory is None
        else ToolResultArtifactStore(
            root=artifact_directory,
            threshold_chars=context_budget.tool_result_compression_threshold_chars,
            write_guard=artifact_write_guard,
        )
    )
    extra_tools = []
    if artifact_store is not None:
        extra_tools.append(ReadArtifactTool(artifact_store.root))
    extra_tools.append(
        DelegateTaskTool(
            subagent_runtime,
            observer=subagent_observer,
        )
    )
    if skill_registry.list_skills():
        extra_tools.extend(
            [
                LoadSkillTool(skill_registry, active_skill_state),
                ReadSkillResourceTool(skill_registry, active_skill_state),
                RunSkillScriptTool(workspace, skill_registry, active_skill_state),
            ]
        )
    tool_registry = create_default_tool_registry(
        workspace,
        confirmer=confirmer,
        memory_store=effective_memory_store,
        extra_tools=extra_tools,
    )

    return AgentRunner(
        llm_client=client,
        tool_registry=tool_registry,
        conversation=conversation,
        context_budget=context_budget,
        instruction_sources=tuple(
            source.label for source in instruction_bundle.sources
        ),
        instruction_warnings=tuple(
            issue.display for issue in instruction_bundle.issues
        ),
        skill_warnings=tuple(warning.display for warning in skill_registry.warnings),
        active_skill_state=active_skill_state,
        memory_context_selector=MemoryContextSelector(
            effective_memory_store,
            policy=memory_recall_policy,
        ),
        tool_batch_handler=DelegationToolBatchHandler(),
        compactor=ConversationCompactor(
            llm_client=summary_client,
            state=CompactState() if compact_state is None else compact_state,
            on_state_changed=on_compact_state_changed,
            observability_sink=observability_sink,
            observability_scope="compact",
        ),
        tool_result_artifact_store=artifact_store,
        observability_sink=observability_sink,
        observability_scope="main",
    )


def run_agent_turn(
    runner: AgentRunner,
    content: str,
    *,
    event_handler: AgentEventHandler | None = None,
) -> AgentRunOutcome:
    """Run one user request, optionally forwarding events to its caller."""
    stop_reasons: list[AgentStopReason] = []
    for event in runner.run(content):
        if event.type == "stop" and event.stop_reason is not None:
            stop_reasons.append(event.stop_reason)
        if event_handler is not None:
            event_handler(event)

    if len(stop_reasons) != 1:
        return AgentRunOutcome.from_stop_reason(None)
    return AgentRunOutcome.from_stop_reason(stop_reasons[0])


def context_budget_from_config(config: LLMConfig) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=config.context_window_tokens,
        reserved_output_tokens=config.reserved_output_tokens,
        safety_margin_tokens=config.context_safety_margin_tokens,
    )
