"""Shared, presentation-neutral formatting for Agent event diagnostics."""

from __future__ import annotations

import subprocess


def summarize_event_content(content: str | None, max_chars: int = 160) -> str:
    """Collapse event content to one bounded diagnostic line."""

    if content is None or content == "":
        return ""
    summary = " ".join(content.split())
    if len(summary) <= max_chars:
        return summary
    return f"{summary[: max_chars - 3]}..."


def summarize_tool_arguments(name: str, arguments: dict[str, object]) -> str:
    """Format useful tool diagnostics while replacing large text bodies with sizes."""

    if name == "read_file":
        return _format_arguments(
            {
                "path": arguments.get("path"),
                "start_line": arguments.get("start_line"),
                "max_lines": arguments.get("max_lines"),
            }
        )
    if name == "read_artifact":
        return _format_arguments(
            {
                "artifact_path": arguments.get("artifact_path"),
                "offset_chars": arguments.get("offset_chars"),
                "max_chars": arguments.get("max_chars"),
            }
        )
    if name == "glob":
        return _format_arguments(
            {
                "pattern": arguments.get("pattern"),
                "max_results": arguments.get("max_results"),
            }
        )
    if name == "grep":
        query = arguments.get("query")
        return _format_arguments(
            {
                "query_chars": len(query) if isinstance(query, str) else None,
                "path_pattern": arguments.get("path_pattern"),
                "case_sensitive": arguments.get("case_sensitive"),
                "max_results": arguments.get("max_results"),
            }
        )
    if name == "delegate_task":
        objective = arguments.get("objective")
        context = arguments.get("context")
        scope_paths = arguments.get("scope_paths")
        return _format_arguments(
            {
                "role": arguments.get("role"),
                "objective_chars": (
                    len(objective) if isinstance(objective, str) else None
                ),
                "context_chars": len(context) if isinstance(context, str) else None,
                "scope_path_count": (
                    len(scope_paths) if isinstance(scope_paths, list) else None
                ),
            }
        )
    if name == "write_file":
        content = arguments.get("content")
        return _format_arguments(
            {
                "path": arguments.get("path"),
                "content_chars": len(content) if isinstance(content, str) else None,
            }
        )
    if name == "edit_file":
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        return _format_arguments(
            {
                "path": arguments.get("path"),
                "old_text_chars": (
                    len(old_text) if isinstance(old_text, str) else None
                ),
                "new_text_chars": (
                    len(new_text) if isinstance(new_text, str) else None
                ),
            }
        )
    if name in {"run_command", "run_validation"}:
        command = arguments.get("command")
        command_display = (
            subprocess.list2cmdline(command)
            if isinstance(command, list)
            and all(isinstance(part, str) for part in command)
            else command
        )
        return _format_arguments(
            {
                "command": command_display,
                "cwd": arguments.get("cwd"),
                "timeout_seconds": arguments.get("timeout_seconds"),
                "max_output_chars": arguments.get("max_output_chars"),
            }
        )
    if name == "inspect_changes":
        paths = arguments.get("paths")
        return _format_arguments(
            {
                "action": arguments.get("action"),
                "path_count": len(paths) if isinstance(paths, list) else None,
                "staged": arguments.get("staged"),
                "base_ref": arguments.get("base_ref"),
                "max_output_chars": arguments.get("max_output_chars"),
            }
        )
    if name == "list_memories":
        return _format_arguments({"scope": arguments.get("scope")})
    if name == "save_memory":
        content = arguments.get("content")
        return _format_arguments(
            {
                "scope": arguments.get("scope"),
                "kind": arguments.get("kind"),
                "key": arguments.get("key"),
                "content_chars": len(content) if isinstance(content, str) else None,
            }
        )
    if name == "delete_memory":
        return _format_arguments(
            {"scope": arguments.get("scope"), "key": arguments.get("key")}
        )
    return _format_arguments(
        {
            "argument_keys": sorted(arguments),
            "argument_count": len(arguments),
        }
    )


def _format_arguments(arguments: dict[str, object]) -> str:
    return ", ".join(
        f"{key}={value!r}" for key, value in arguments.items() if value is not None
    )
