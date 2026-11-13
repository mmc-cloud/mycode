"""Shared text formatting for interactive command output."""

from collections.abc import Iterable

from mycode.application.agent_session import CompactResult, ContextStatus
from mycode.persistence.session_store import SessionRecord
from mycode.presentation.commands import COMMAND_SPECS


def format_command_help() -> tuple[str, ...]:
    lines = ["available commands:"]
    for spec in COMMAND_SPECS:
        aliases = ""
        if spec.aliases:
            aliases = " (alias: " + ", ".join(
                f"/{alias}" for alias in spec.aliases
            ) + ")"
        lines.append(f"{spec.usage}{aliases} - {spec.description}")
    return tuple(lines)


def format_session_list(
    sessions: Iterable[SessionRecord],
    *,
    current_session_id: str,
) -> tuple[str, ...]:
    records = tuple(sessions)
    if not records:
        return ("sessions in current project:", "(no sessions)")
    return (
        "sessions in current project:",
        *(
            f"{'*' if session.id == current_session_id else ' '} "
            f"{session.id}  {session.title}"
            for session in records
        ),
    )


def format_context_status(status: ContextStatus) -> tuple[str, ...]:
    provider = (
        "unavailable"
        if status.last_provider_prompt_tokens is None
        else f"{status.last_provider_prompt_tokens:,} tokens"
    )
    memory = (
        "memory: none"
        if status.memory_entry_count == 0
        else (
            "memory: "
            f"{status.memory_entry_count} entries / "
            f"~{status.memory_estimated_tokens:,} tokens"
        )
    )
    return (
        "Context",
        f"estimated input: ~{status.estimated_input_tokens:,} tokens",
        f"context occupancy: ~{status.estimated_input_tokens:,} / "
        f"{status.context_window_tokens:,} tokens ({status.context_window_percent:.1f}%)",
        f"input budget: ~{status.estimated_input_tokens:,} / "
        f"{status.max_input_tokens:,} tokens ({status.input_budget_percent:.1f}%)",
        f"context window: {status.context_window_tokens:,}",
        f"window source: {status.context_window_source} (" + (
            f"API limit={status.context_window_server_limit})" if status.context_window_server_limit is not None else "unverified)"),
        f"reserved output: {status.reserved_output_tokens:,}",
        f"safety margin: {status.safety_margin_tokens:,}",
        f"estimate source: {status.estimate_source}",
        f"last provider prompt: {provider}",
        "messages: "
        f"{status.source_message_count} source / "
        f"{status.model_visible_message_count} model-visible",
        memory,
        "compact: "
        f"{status.compact_status} / "
        f"{status.compact_covered_message_count} messages covered",
        "tool results: "
        f"{status.compressed_tool_result_count} compressed",
    )


def format_context_summary(status: ContextStatus) -> str:
    def short(tokens):
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}".rstrip("0").rstrip(".") + "K"
        return str(tokens)
    return f"~{short(status.estimated_input_tokens)} / {short(status.context_window_tokens)} ({status.context_window_percent:.1f}%)"


def format_compact_result(result: CompactResult) -> str:
    if result.status == "compacted" and result.before and result.after:
        return (
            "compacted: ~"
            f"{result.before.estimated_input_tokens:,} → ~"
            f"{result.after.estimated_input_tokens:,} tokens"
        )
    reason = "unknown" if result.reason is None else result.reason
    return f"compact {result.status}: {reason}"
