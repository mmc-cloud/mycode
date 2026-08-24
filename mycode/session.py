from dataclasses import dataclass, field, replace
from collections.abc import Iterator

from mycode.context_compact import ConversationCompactor
from mycode.context_budget import (
    ContextBudget,
    ContextBudgetExceededError,
    ModelContext,
    TokenEstimator,
    TokenUsage,
    build_model_context,
)
from mycode.conversation import Conversation
from mycode.llm import LLMClient
from mycode.messages import Message


@dataclass
class ChatSession:
    llm_client: LLMClient
    conversation: Conversation = field(default_factory=Conversation)
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    token_estimator: TokenEstimator = field(default_factory=TokenEstimator)
    last_model_context: ModelContext | None = field(default=None, init=False)
    last_token_usage: TokenUsage | None = field(default=None, init=False)
    last_reasoning_char_count: int = field(default=0, init=False)
    last_compact_token_usage: TokenUsage | None = field(default=None, init=False)
    compactor: ConversationCompactor | None = None

    def send_user_message(self, content: str) -> Message:
        self.conversation.add_user_message(content)

        context = self._model_context()
        reply = self.llm_client.complete(
            Conversation.from_messages(list(context.messages))
        )
        self._observe_token_usage(context)

        self.conversation.add_message(reply)

        return reply

    def stream_user_message(self, content: str) -> Iterator[str]:
        self.conversation.add_user_message(content)

        context = self._model_context()
        model_conversation = Conversation.from_messages(list(context.messages))
        return self._stream_model_reply(context, model_conversation)

    def _stream_model_reply(
        self,
        context: ModelContext,
        model_conversation: Conversation,
    ) -> Iterator[str]:
        reply_parts: list[str] = []

        for chunk in self.llm_client.stream_complete(model_conversation):
            reply_parts.append(chunk)
            yield chunk

        self._observe_token_usage(context)
        self.conversation.add_assistant_message("".join(reply_parts))

    def _model_context(self) -> ModelContext:
        canonical_conversation = self.conversation
        context_conversation = canonical_conversation
        compact_stats = None
        if self.compactor is not None:
            prepared = self.compactor.prepare(
                self.conversation,
                self.context_budget,
                token_estimator=self.token_estimator,
            )
            context_conversation = prepared.conversation
            compact_stats = prepared.stats
            self.last_compact_token_usage = prepared.attempt_token_usage

        context = build_model_context(
            context_conversation,
            self.context_budget,
            token_estimator=self.token_estimator,
        )
        if (
            context.estimate.over_budget
            and compact_stats is not None
            and compact_stats.boundary_id is not None
        ):
            canonical_fallback = build_model_context(
                canonical_conversation,
                self.context_budget,
                token_estimator=self.token_estimator,
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
        self.last_model_context = context
        if context.estimate.over_budget:
            raise ContextBudgetExceededError(context)

        return context

    def _observe_token_usage(self, context: ModelContext) -> None:
        usage = getattr(self.llm_client, "last_token_usage", None)
        self.last_token_usage = usage
        self.last_reasoning_char_count = getattr(
            self.llm_client,
            "last_reasoning_char_count",
            0,
        )
        self.token_estimator.observe(context.estimate, usage)
