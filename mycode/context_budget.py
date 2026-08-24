from dataclasses import dataclass, field, replace
import json
import math

from mycode.conversation import Conversation
from mycode.messages import Message


DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_RESERVED_OUTPUT_TOKENS = 8192
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 4096
DEFAULT_TOOL_RESULT_COMPRESSION_THRESHOLD_CHARS = 4000
DEFAULT_RECENT_TOOL_RESULT_GROUPS_TO_KEEP = 1
DEFAULT_ASCII_TOKENS_PER_CHAR = 0.5
DEFAULT_MIXED_TOKENS_PER_CHAR = 0.8
DEFAULT_NON_ASCII_TOKENS_PER_CHAR = 1.2
DEFAULT_CALIBRATION_SAFETY_FACTOR = 1.1
DEFAULT_MAX_CALIBRATION_SAMPLES = 20
TOOL_RESULT_METADATA_MARKER = "\n\nMETADATA\n"
COMPRESSED_TOOL_RESULT_MARKER = "[tool result compressed]"
LARGE_TOOL_METADATA_KEYS = {"stdout", "stderr"}
MAX_METADATA_STRING_CHARS = 200
MAX_TOOL_RESULT_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class ContextBudget:
    """Model context budget measured in tokens."""

    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS
    safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS
    tool_result_compression_threshold_chars: int = (
        DEFAULT_TOOL_RESULT_COMPRESSION_THRESHOLD_CHARS
    )
    recent_tool_result_groups_to_keep: int = DEFAULT_RECENT_TOOL_RESULT_GROUPS_TO_KEEP

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1.")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be at least 0.")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be at least 0.")
        if self.max_input_tokens < 1:
            raise ValueError(
                "reserved_output_tokens and safety_margin_tokens must leave at "
                "least 1 input token."
            )
        if self.tool_result_compression_threshold_chars < 1:
            raise ValueError(
                "tool_result_compression_threshold_chars must be at least 1."
            )
        if self.recent_tool_result_groups_to_keep < 0:
            raise ValueError("recent_tool_result_groups_to_keep must be at least 0.")

    @property
    def max_input_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TokenEstimator:
    """Estimate input tokens and calibrate from provider-reported prompt usage."""

    ascii_tokens_per_char: float = DEFAULT_ASCII_TOKENS_PER_CHAR
    mixed_tokens_per_char: float = DEFAULT_MIXED_TOKENS_PER_CHAR
    non_ascii_tokens_per_char: float = DEFAULT_NON_ASCII_TOKENS_PER_CHAR
    calibration_safety_factor: float = DEFAULT_CALIBRATION_SAFETY_FACTOR
    max_samples_per_profile: int = DEFAULT_MAX_CALIBRATION_SAMPLES
    _samples: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("ascii_tokens_per_char", self.ascii_tokens_per_char),
            ("mixed_tokens_per_char", self.mixed_tokens_per_char),
            ("non_ascii_tokens_per_char", self.non_ascii_tokens_per_char),
            ("calibration_safety_factor", self.calibration_safety_factor),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
        if self.max_samples_per_profile < 1:
            raise ValueError("max_samples_per_profile must be at least 1.")

    def estimate(self, *, total_chars: int, non_ascii_chars: int) -> "TokenEstimate":
        profile = _text_profile(total_chars, non_ascii_chars)
        samples = self._samples.get(profile, [])
        if samples:
            coefficient = _percentile(samples, 0.9) * self.calibration_safety_factor
            source = "calibrated"
        else:
            coefficient = self._default_coefficient(profile)
            source = "default"

        return TokenEstimate(
            estimated_tokens=math.ceil(total_chars * coefficient),
            tokens_per_char=coefficient,
            profile=profile,
            source=source,
        )

    def observe(self, estimate: "ConversationEstimate", usage: TokenUsage | None) -> None:
        if usage is None or usage.prompt_tokens < 1 or estimate.total_chars < 1:
            return

        samples = self._samples.setdefault(estimate.token_profile, [])
        samples.append(usage.prompt_tokens / estimate.total_chars)
        del samples[: -self.max_samples_per_profile]

    def _default_coefficient(self, profile: str) -> float:
        if profile == "ascii":
            return self.ascii_tokens_per_char
        if profile == "mixed":
            return self.mixed_tokens_per_char
        return self.non_ascii_tokens_per_char


@dataclass(frozen=True)
class TokenEstimate:
    estimated_tokens: int
    tokens_per_char: float
    profile: str
    source: str


@dataclass(frozen=True)
class MessageEstimate:
    """Per-message estimate in Python characters, not UTF-8 bytes or model tokens."""

    role: str
    content_chars: int
    reasoning_chars: int = 0
    tool_call_chars: int = 0
    tool_call_id_chars: int = 0
    serialization_overhead_chars: int = 0

    @property
    def total_chars(self) -> int:
        return (
            len(self.role)
            + self.content_chars
            + self.reasoning_chars
            + self.tool_call_chars
            + self.tool_call_id_chars
            + self.serialization_overhead_chars
        )


@dataclass(frozen=True)
class ConversationEstimate:
    message_count: int
    message_chars: int
    estimated_input_tokens: int
    max_input_tokens: int
    tokens_per_char: float
    token_profile: str
    token_estimate_source: str
    message_estimates: tuple[MessageEstimate, ...]
    tool_schema_chars: int = 0
    message_list_overhead_chars: int = 0

    @property
    def total_chars(self) -> int:
        return self.message_chars + self.tool_schema_chars

    @property
    def over_budget(self) -> bool:
        return self.estimated_input_tokens > self.max_input_tokens


@dataclass(frozen=True)
class ModelContext:
    messages: tuple[Message, ...]
    estimate: ConversationEstimate
    original_message_count: int
    compressed_tool_result_count: int = 0
    memory_stats: "MemoryContextStats | None" = None
    compact_stats: "CompactContextStats | None" = None

    @property
    def selected_message_count(self) -> int:
        return len(self.messages)

    @property
    def source_message_count(self) -> int:
        """Canonical message count represented by this model-visible view."""
        if (
            self.compact_stats is None
            or self.compact_stats.compacted_message_count == 0
            or not self.compact_stats.summary_visible
        ):
            return self.original_message_count

        # The visible compact-summary message replaces the covered canonical
        # messages, so add the covered messages and remove that one summary.
        return (
            self.original_message_count
            + self.compact_stats.compacted_message_count
            - 1
        )

    @property
    def trimmed_message_count(self) -> int:
        return self.original_message_count - self.selected_message_count

    @property
    def trimmed(self) -> bool:
        return self.trimmed_message_count > 0


@dataclass(frozen=True)
class MemoryContextStats:
    """Content-free observability for one long-term-memory recall."""

    safe_entry_count: int = 0
    relevant_entry_count: int = 0
    selected_entry_count: int = 0
    included_entry_count: int = 0
    estimated_tokens: int = 0
    irrelevant_entry_count: int = 0
    conflict_count: int = 0
    budget_omitted_count: int = 0
    issue_count: int = 0
    scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        numeric_values = (
            self.safe_entry_count,
            self.relevant_entry_count,
            self.selected_entry_count,
            self.included_entry_count,
            self.estimated_tokens,
            self.irrelevant_entry_count,
            self.conflict_count,
            self.budget_omitted_count,
            self.issue_count,
        )
        if any(value < 0 for value in numeric_values):
            raise ValueError("Memory context statistics must not be negative.")
        if self.included_entry_count > self.selected_entry_count:
            raise ValueError(
                "included_entry_count must not exceed selected_entry_count."
            )

    def with_included_entries(self, count: int) -> "MemoryContextStats":
        return replace(self, included_entry_count=count)


@dataclass(frozen=True)
class CompactContextStats:
    """Content-free observability for one conversation Compact decision."""

    status: str
    boundary_id: str | None = None
    compacted_message_count: int = 0
    covered_turn_count: int = 0
    consecutive_failure_count: int = 0
    retry_after_message_count: int = 0
    circuit_open: bool = False
    summary_visible: bool = False

    def __post_init__(self) -> None:
        numeric_values = (
            self.compacted_message_count,
            self.covered_turn_count,
            self.consecutive_failure_count,
            self.retry_after_message_count,
        )
        if any(value < 0 for value in numeric_values):
            raise ValueError("Compact context statistics must not be negative.")
        if self.boundary_id is None and (
            self.compacted_message_count > 0 or self.covered_turn_count > 0
        ):
            raise ValueError(
                "Compact counts require a persisted or in-memory boundary id."
            )
        if self.summary_visible and self.boundary_id is None:
            raise ValueError(
                "A visible Compact summary requires a boundary id."
            )


class ContextBudgetExceededError(RuntimeError):
    def __init__(self, context: ModelContext) -> None:
        self.context = context
        super().__init__(
            "Model context exceeds the configured input budget: "
            f"tokens={context.estimate.estimated_input_tokens}/"
            f"{context.estimate.max_input_tokens}."
        )


def estimate_message(message: Message) -> MessageEstimate:
    model_dict = message.to_model_dict()
    content_chars = len(message.content)
    reasoning_chars = len(message.reasoning_content or "")
    tool_call_chars = _estimate_tool_calls(message)
    tool_call_id_chars = len(message.tool_call_id or "")
    component_chars = (
        len(message.role)
        + content_chars
        + reasoning_chars
        + tool_call_chars
        + tool_call_id_chars
    )
    serialized_chars = len(json.dumps(model_dict, ensure_ascii=False))

    return MessageEstimate(
        role=message.role,
        content_chars=content_chars,
        reasoning_chars=reasoning_chars,
        tool_call_chars=tool_call_chars,
        tool_call_id_chars=tool_call_id_chars,
        serialization_overhead_chars=serialized_chars - component_chars,
    )


def estimate_conversation(
    conversation: Conversation,
    budget: ContextBudget | None = None,
    tools: list[dict[str, object]] | None = None,
    token_estimator: TokenEstimator | None = None,
) -> ConversationEstimate:
    active_budget = ContextBudget() if budget is None else budget
    active_estimator = TokenEstimator() if token_estimator is None else token_estimator
    message_estimates = tuple(
        estimate_message(message) for message in conversation.get_messages()
    )
    serialized_message_text = json.dumps(
        conversation.to_model_messages(), ensure_ascii=False
    )
    serialized_messages = len(serialized_message_text)
    estimated_message_chars = sum(
        estimate.total_chars for estimate in message_estimates
    )
    message_list_overhead_chars = serialized_messages - estimated_message_chars

    serialized_tool_text = _serialize_tool_schemas(tools)
    total_text = serialized_message_text + serialized_tool_text
    token_estimate = active_estimator.estimate(
        total_chars=len(total_text),
        non_ascii_chars=_count_non_ascii(total_text),
    )

    return ConversationEstimate(
        message_count=len(message_estimates),
        message_chars=serialized_messages,
        estimated_input_tokens=token_estimate.estimated_tokens,
        max_input_tokens=active_budget.max_input_tokens,
        tokens_per_char=token_estimate.tokens_per_char,
        token_profile=token_estimate.profile,
        token_estimate_source=token_estimate.source,
        message_estimates=message_estimates,
        tool_schema_chars=len(serialized_tool_text),
        message_list_overhead_chars=message_list_overhead_chars,
    )


def build_model_context(
    conversation: Conversation,
    budget: ContextBudget | None = None,
    tools: list[dict[str, object]] | None = None,
    token_estimator: TokenEstimator | None = None,
    *,
    memory_message: Message | None = None,
    memory_stats: MemoryContextStats | None = None,
) -> ModelContext:
    active_budget = ContextBudget() if budget is None else budget
    active_estimator = TokenEstimator() if token_estimator is None else token_estimator
    conversation_messages = tuple(conversation.get_messages())
    if memory_message is not None and memory_message.role != "system":
        raise ValueError("memory_message must use the system role.")
    if memory_message is None and memory_stats is not None:
        if memory_stats.selected_entry_count > 0:
            raise ValueError(
                "memory_stats cannot report selected entries without a memory_message."
            )

    candidate_messages = conversation_messages
    if memory_message is not None:
        candidate_messages = (*conversation_messages, memory_message)

    context = _build_model_context_from_messages(
        candidate_messages,
        active_budget,
        tools=tools,
        token_estimator=active_estimator,
    )
    if memory_message is not None and context.estimate.over_budget:
        context = _build_model_context_from_messages(
            conversation_messages,
            active_budget,
            tools=tools,
            token_estimator=active_estimator,
        )
        if memory_stats is not None:
            memory_stats = memory_stats.with_included_entries(0)
    elif memory_stats is not None:
        memory_stats = memory_stats.with_included_entries(
            memory_stats.selected_entry_count
        )

    return replace(context, memory_stats=memory_stats)


def _build_model_context_from_messages(
    original_messages: tuple[Message, ...],
    budget: ContextBudget,
    *,
    tools: list[dict[str, object]] | None,
    token_estimator: TokenEstimator,
) -> ModelContext:
    system_messages = tuple(
        message for message in original_messages if message.role == "system"
    )
    non_system_groups = _group_non_system_messages(original_messages)
    valid_messages = system_messages + _flatten_groups(non_system_groups)
    full_estimate = estimate_conversation(
        Conversation.from_messages(list(valid_messages)),
        budget,
        tools=tools,
        token_estimator=token_estimator,
    )
    if not full_estimate.over_budget:
        return ModelContext(
            messages=valid_messages,
            estimate=full_estimate,
            original_message_count=len(original_messages),
        )

    non_system_groups = _compress_old_tool_results(
        non_system_groups,
        budget,
    )
    compressed_messages = system_messages + _flatten_groups(non_system_groups)
    compressed_estimate = estimate_conversation(
        Conversation.from_messages(list(compressed_messages)),
        budget,
        tools=tools,
        token_estimator=token_estimator,
    )
    if not compressed_estimate.over_budget:
        return ModelContext(
            messages=compressed_messages,
            estimate=compressed_estimate,
            original_message_count=len(original_messages),
            compressed_tool_result_count=_count_compressed_tool_results(
                compressed_messages
            ),
        )

    selected_groups: list[tuple[Message, ...]] = []

    for group in reversed(non_system_groups):
        candidate_groups = [group, *selected_groups]
        candidate_messages = system_messages + _flatten_groups(candidate_groups)
        candidate_estimate = estimate_conversation(
            Conversation.from_messages(list(candidate_messages)),
            budget,
            tools=tools,
            token_estimator=token_estimator,
        )
        if candidate_estimate.over_budget and selected_groups:
            break

        selected_groups.insert(0, group)

    selected_messages = system_messages + _flatten_groups(selected_groups)
    selected_estimate = estimate_conversation(
        Conversation.from_messages(list(selected_messages)),
        budget,
        tools=tools,
        token_estimator=token_estimator,
    )

    return ModelContext(
        messages=selected_messages,
        estimate=selected_estimate,
        original_message_count=len(original_messages),
        compressed_tool_result_count=_count_compressed_tool_results(selected_messages),
    )


def model_context_needs_notice(context: ModelContext) -> bool:
    memory_needs_notice = context.memory_stats is not None and (
        context.memory_stats.included_entry_count
        != context.memory_stats.selected_entry_count
        or context.memory_stats.budget_omitted_count > 0
        or context.memory_stats.conflict_count > 0
        or context.memory_stats.issue_count > 0
    )
    return (
        context.trimmed
        or context.compressed_tool_result_count > 0
        or (
            context.compact_stats is not None
            and (
                context.compact_stats.compacted_message_count > 0
                or context.compact_stats.status
                in {
                    "failed",
                    "cooldown",
                    "circuit_open",
                    "invalid_boundary",
                    "canonical_fallback",
                }
            )
        )
        or context.estimate.over_budget
        or memory_needs_notice
    )


def format_model_context_stats(
    context: ModelContext,
    *,
    previous_prompt_tokens: int | None = None,
) -> str:
    estimated_tokens = context.estimate.estimated_input_tokens
    max_input_tokens = context.estimate.max_input_tokens
    used_percent = estimated_tokens / max_input_tokens * 100
    left_percent = max(0.0, 100.0 - used_percent)
    source = (
        "calibrated estimate"
        if context.estimate.token_estimate_source == "calibrated"
        else "conservative estimate, calibration pending"
    )
    summary = (
        f"~{estimated_tokens:,} / {max_input_tokens:,} tokens "
        f"({used_percent:.1f}% used, {left_percent:.1f}% left, {source}), "
        f"messages={context.selected_message_count}/{context.source_message_count}"
    )
    if context.compact_stats is not None:
        compact = context.compact_stats
        if compact.boundary_id is not None:
            compact_label = (
                "compact"
                if compact.summary_visible
                else "compact_boundary"
            )
            summary += (
                f", {compact_label}={compact.compacted_message_count} messages/"
                f"{compact.covered_turn_count} turns"
                f"@{compact.boundary_id[:8]}"
            )
            if not compact.summary_visible:
                summary += ", compact_summary_visible=false"
        if compact.status in {
            "failed",
            "cooldown",
            "circuit_open",
            "invalid_boundary",
            "canonical_fallback",
        }:
            summary += (
                f", compact_status={compact.status}, "
                f"compact_failures={compact.consecutive_failure_count}, "
                f"compact_retry_after_messages="
                f"{compact.retry_after_message_count}"
            )
    if context.memory_stats is not None:
        memory = context.memory_stats
        summary += (
            f", memory={memory.included_entry_count} injected/"
            f"{memory.selected_entry_count} selected/"
            f"{memory.relevant_entry_count} relevant/"
            f"{memory.safe_entry_count} safe "
            f"(~{memory.estimated_tokens:,} tokens)"
        )
        if memory.scopes:
            summary += f", memory_scopes={'+'.join(memory.scopes)}"
        if memory.irrelevant_entry_count > 0:
            summary += f", memory_irrelevant={memory.irrelevant_entry_count}"
        if memory.conflict_count > 0:
            summary += f", memory_conflicts={memory.conflict_count}"
        if memory.budget_omitted_count > 0:
            summary += f", memory_budget_omitted={memory.budget_omitted_count}"
        if memory.issue_count > 0:
            summary += f", memory_warnings={memory.issue_count}"
    if previous_prompt_tokens is not None:
        summary += f", previous_actual_input={previous_prompt_tokens:,} tokens"
    if not model_context_needs_notice(context):
        return summary

    return (
        f"{summary}, trimmed={context.trimmed_message_count}, "
        f"compressed={context.compressed_tool_result_count}, "
        f"over_budget={context.estimate.over_budget}"
    )


def _estimate_tool_calls(message: Message) -> int:
    model_tool_calls = message.to_model_dict().get("tool_calls")
    if not isinstance(model_tool_calls, list):
        return 0

    return len(json.dumps(model_tool_calls, ensure_ascii=False))


def _serialize_tool_schemas(tools: list[dict[str, object]] | None) -> str:
    if not tools:
        return ""

    openai_tools = [{"type": "function", "function": dict(tool)} for tool in tools]
    return json.dumps(openai_tools, ensure_ascii=False)


def _count_non_ascii(content: str) -> int:
    return sum(1 for character in content if ord(character) > 127)


def _text_profile(total_chars: int, non_ascii_chars: int) -> str:
    if total_chars < 1 or non_ascii_chars / total_chars < 0.1:
        return "ascii"
    if non_ascii_chars / total_chars < 0.5:
        return "mixed"
    return "non_ascii"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _group_non_system_messages(
    messages: tuple[Message, ...],
) -> list[tuple[Message, ...]]:
    groups: list[tuple[Message, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role in {"system", "tool"}:
            index += 1
            continue

        if message.role == "assistant" and message.tool_calls:
            expected_tool_call_ids = [
                tool_call.id for tool_call in message.tool_calls
            ]
            tool_results: list[Message] = []
            next_index = index + 1
            while (
                next_index < len(messages)
                and messages[next_index].role == "tool"
            ):
                tool_results.append(messages[next_index])
                next_index += 1

            result_ids = [message.tool_call_id for message in tool_results]
            chain_is_complete = (
                len(expected_tool_call_ids) == len(set(expected_tool_call_ids))
                and len(result_ids) == len(expected_tool_call_ids)
                and len(result_ids) == len(set(result_ids))
                and set(result_ids) == set(expected_tool_call_ids)
            )
            if chain_is_complete:
                groups.append((message, *tool_results))

            index = next_index
            continue

        groups.append((message,))
        index += 1

    return groups


def _flatten_groups(groups: list[tuple[Message, ...]]) -> tuple[Message, ...]:
    return tuple(message for group in groups for message in group)


def _compress_old_tool_results(
    groups: list[tuple[Message, ...]],
    budget: ContextBudget,
) -> list[tuple[Message, ...]]:
    tool_group_indexes = [
        index for index, group in enumerate(groups) if _group_has_tool_result(group)
    ]
    keep_indexes = set(tool_group_indexes[-budget.recent_tool_result_groups_to_keep :])
    if budget.recent_tool_result_groups_to_keep == 0:
        keep_indexes = set()

    compressed_groups: list[tuple[Message, ...]] = []
    for index, group in enumerate(groups):
        if index in keep_indexes:
            compressed_groups.append(group)
            continue

        tool_names_by_id = _tool_names_by_id(group)
        compressed_groups.append(
            tuple(
                _compress_tool_result(message, budget, tool_names_by_id)
                for message in group
            )
        )

    return compressed_groups


def _group_has_tool_result(group: tuple[Message, ...]) -> bool:
    return any(message.role == "tool" for message in group)


def _tool_names_by_id(group: tuple[Message, ...]) -> dict[str, str]:
    tool_names: dict[str, str] = {}
    for message in group:
        if message.role != "assistant":
            continue

        for tool_call in message.tool_calls:
            tool_names[tool_call.id] = tool_call.name

    return tool_names


def _compress_tool_result(
    message: Message,
    budget: ContextBudget,
    tool_names_by_id: dict[str, str],
) -> Message:
    if message.role != "tool":
        return message

    if len(message.content) <= budget.tool_result_compression_threshold_chars:
        return message

    parsed = parse_tool_result_content(message.content)
    tool_name = tool_names_by_id.get(message.tool_call_id or "", "unknown")
    metadata = safe_tool_metadata(parsed.metadata)
    if parsed.result_preview:
        preview_key = "error_preview" if parsed.status == "ERROR" else "result_preview"
        metadata[preview_key] = parsed.result_preview
    metadata.update(
        {
            "context_compressed": True,
            "original_chars": len(message.content),
            "tool_name": tool_name,
            "tool_call_id": message.tool_call_id,
        }
    )
    compressed_content = (
        f"{parsed.status}\n"
        f"{COMPRESSED_TOOL_RESULT_MARKER}\n"
        f"tool_name: {tool_name}\n"
        f"original_chars: {len(message.content)}\n"
        f"{TOOL_RESULT_METADATA_MARKER}"
        f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)}"
    )

    return Message(
        role="tool",
        content=compressed_content,
        tool_call_id=message.tool_call_id,
    )


@dataclass(frozen=True)
class ParsedToolResultContent:
    status: str
    metadata: dict[str, object]
    result_preview: str


def parse_tool_result_content(content: str) -> ParsedToolResultContent:
    body, _separator, metadata_text = content.partition(TOOL_RESULT_METADATA_MARKER)
    body_lines = body.splitlines()
    first_line = body_lines[0] if body_lines else "UNKNOWN"
    status = first_line if first_line in {"OK", "ERROR"} else "UNKNOWN"
    result_body = "\n".join(body_lines[1:]) if status != "UNKNOWN" else body
    result_preview = _truncate_tool_result_preview(result_body.strip())

    try:
        metadata = json.loads(metadata_text) if metadata_text else {}
    except json.JSONDecodeError:
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}

    return ParsedToolResultContent(
        status=status,
        metadata=metadata,
        result_preview=result_preview,
    )


def _truncate_tool_result_preview(content: str) -> str:
    if len(content) <= MAX_TOOL_RESULT_PREVIEW_CHARS:
        return content

    return f"{content[:MAX_TOOL_RESULT_PREVIEW_CHARS]}..."


def safe_tool_metadata(metadata: dict[str, object]) -> dict[str, object]:
    safe_metadata: dict[str, object] = {}
    for key, value in metadata.items():
        if key in LARGE_TOOL_METADATA_KEYS:
            safe_metadata[f"{key}_omitted"] = True
            continue

        safe_metadata[key] = _safe_metadata_value(value)

    return safe_metadata


def _safe_metadata_value(value: object) -> object:
    if isinstance(value, str) and len(value) > MAX_METADATA_STRING_CHARS:
        return f"{value[:MAX_METADATA_STRING_CHARS]}..."

    if isinstance(value, list):
        return [_safe_metadata_value(item) for item in value[:20]]

    if isinstance(value, dict):
        return {
            str(key): _safe_metadata_value(item)
            for key, item in list(value.items())[:20]
        }

    return value


def _count_compressed_tool_results(messages: tuple[Message, ...]) -> int:
    return sum(
        1
        for message in messages
        if message.role == "tool" and COMPRESSED_TOOL_RESULT_MARKER in message.content
    )
