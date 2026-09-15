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
    percent = (
        0.0
        if status.max_input_tokens < 1
        else status.estimated_input_tokens / status.max_input_tokens * 100
    )
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
        "estimated input: "
        f"{status.estimated_input_tokens:,} / "
        f"{status.max_input_tokens:,} tokens ({percent:.1f}%)",
        f"context window: {status.context_window_tokens:,}",
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


def format_compact_result(result: CompactResult) -> str:
    if result.status == "compacted" and result.before and result.after:
        return (
            "compacted: ~"
            f"{result.before.estimated_input_tokens:,} → ~"
            f"{result.after.estimated_input_tokens:,} tokens"
        )
    reason = "unknown" if result.reason is None else result.reason
    return f"compact {result.status}: {reason}"
