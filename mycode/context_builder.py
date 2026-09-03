"""One-way model context construction shared by Agent and Chat."""

from dataclasses import dataclass, replace

from mycode.context_budget import (
    ContextBudget,
    MemoryContextStats,
    ModelContext,
    TokenEstimator,
    TokenUsage,
    budget_model_context,
)
from mycode.context_compact import ConversationCompactor
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.tool_result_retention import ToolResultRetentionPolicy, TurnLocalFullGroup


@dataclass(frozen=True)
class ContextBuildResult:
    context: ModelContext
    compact_attempt_token_usage: TokenUsage | None = None


@dataclass
class ContextBuilder:
    budget: ContextBudget
    token_estimator: TokenEstimator
    compactor: ConversationCompactor | None = None
    retention_policy: ToolResultRetentionPolicy | None = None

    def build(
        self,
        conversation: Conversation,
        *,
        tools: list[dict[str, object]] | None = None,
        memory_message: Message | None = None,
        memory_stats: MemoryContextStats | None = None,
        guidance: tuple[str, ...] = (),
        request_messages: tuple[Message, ...] = (),
        persistent_system_messages: tuple[Message, ...] = (),
        turn_local_full_group: TurnLocalFullGroup | None = None,
        observability_turn: int | None = None,
    ) -> ContextBuildResult:
        """Project tool retention, Compact, assemble this request, then budget once.

        Persistent system messages are request-scoped but visible to Compact's
        trigger. Guidance and request messages remain invisible to Compact. An
        over-budget result is returned to the caller, which must not send it.
        """
        projection = None
        if self.retention_policy is not None:
            projection = self.retention_policy.project(
                conversation, turn_local_full_group=turn_local_full_group,
            )
            conversation = projection.conversation
        if any(message.role != "system" for message in persistent_system_messages):
            raise ValueError("persistent_system_messages must use the system role.")
        if persistent_system_messages:
            conversation = Conversation.from_messages(
                [*conversation.get_messages(), *persistent_system_messages]
            )
        compact_stats = None
        compact_usage = None
        if self.compactor is not None:
            prepared = self.compactor.prepare(
                conversation,
                self.budget,
                tools=tools,
                token_estimator=self.token_estimator,
                memory_message=memory_message,
                observability_turn=observability_turn,
            )
            conversation = prepared.conversation
            compact_stats = prepared.stats
            compact_usage = prepared.attempt_token_usage

        messages = list(conversation.get_messages())
        if memory_message is not None:
            messages.append(memory_message)
        if guidance:
            messages.append(Message(role="system", content="\n\n".join(guidance)))
        messages.extend(request_messages)
        context = budget_model_context(
            tuple(messages),
            self.budget,
            tools=tools,
            token_estimator=self.token_estimator,
            memory_message=memory_message,
            memory_stats=memory_stats,
            retention_policy=self.retention_policy,
            retention_projection=projection,
        )
        return ContextBuildResult(
            context=replace(context, compact_stats=compact_stats),
            compact_attempt_token_usage=compact_usage,
        )
