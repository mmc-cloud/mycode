from dataclasses import dataclass
import hashlib
import json
import math
from typing import TypeAlias

from mycode.agent import AgentToolCall
from mycode.tools.base import ToolResult


MAX_AUDIT_REASON_CHARS = 100
KNOWN_SUBAGENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "run_validation",
        "inspect_changes",
        "submit_result",
    }
)
AuditScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class SubAgentToolAudit:
    tool_name: str
    arguments_sha256: str
    argument_summary: dict[str, AuditScalar]
    ok: bool
    exit_code: int | None
    duration_ms: int | None
    output_chars: int
    truncated: bool
    reason: str | None


def build_tool_audit(
    tool_call: AgentToolCall,
    result: ToolResult,
) -> SubAgentToolAudit:
    arguments_json = _canonical_json(tool_call.arguments)
    body = result.content if result.ok else result.error or ""
    metadata = result.metadata
    tool_name = (
        tool_call.name
        if tool_call.name in KNOWN_SUBAGENT_TOOL_NAMES
        else "unknown"
    )
    return SubAgentToolAudit(
        tool_name=tool_name,
        arguments_sha256=hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
        argument_summary=_argument_summary(
            tool_name,
            tool_call.arguments,
            requested_tool_name=tool_call.name,
        ),
        ok=result.ok,
        exit_code=_optional_int(metadata.get("exit_code")),
        duration_ms=_optional_non_negative_int(metadata.get("duration_ms")),
        output_chars=len(body),
        truncated=_optional_bool(metadata.get("truncated")) is True,
        reason=_safe_reason(metadata.get("reason")),
    )


def _argument_summary(
    name: str,
    arguments: dict[str, object],
    *,
    requested_tool_name: str,
) -> dict[str, AuditScalar]:
    if name == "unknown":
        return {
            "requested_tool_name_sha256": _value_sha256(requested_tool_name),
            "requested_tool_name_chars": len(requested_tool_name),
            "argument_key_count": len(arguments),
            "argument_keys_sha256": _value_sha256(sorted(str(key) for key in arguments)),
            "argument_chars": len(_canonical_json(arguments)),
        }
    if name == "read_file":
        return {
            "path_sha256": _value_sha256(arguments.get("path")),
            "start_line": _optional_int(arguments.get("start_line")),
            "max_lines": _optional_int(arguments.get("max_lines")),
        }
    if name == "glob":
        return {
            "pattern_sha256": _value_sha256(arguments.get("pattern")),
            "pattern_chars": _string_chars(arguments.get("pattern")),
            "max_results": _optional_int(arguments.get("max_results")),
        }
    if name == "grep":
        return {
            "query_sha256": _value_sha256(arguments.get("query")),
            "query_chars": _string_chars(arguments.get("query")),
            "path_pattern_sha256": _value_sha256(arguments.get("path_pattern")),
            "case_sensitive": _optional_bool(arguments.get("case_sensitive")),
            "max_results": _optional_int(arguments.get("max_results")),
        }
    if name == "run_validation":
        command = arguments.get("command")
        return {
            "command_sha256": _value_sha256(command),
            "command_parts": len(command) if isinstance(command, list) else None,
            "cwd_sha256": _value_sha256(arguments.get("cwd")),
            "timeout_seconds": _optional_number(arguments.get("timeout_seconds")),
            "max_output_chars": _optional_int(arguments.get("max_output_chars")),
        }
    if name == "inspect_changes":
        paths = arguments.get("paths")
        return {
            "action": _short_string(arguments.get("action")),
            "path_count": len(paths) if isinstance(paths, list) else None,
            "paths_sha256": _value_sha256(paths),
            "staged": _optional_bool(arguments.get("staged")),
            "base_ref_sha256": _value_sha256(arguments.get("base_ref")),
            "max_output_chars": _optional_int(arguments.get("max_output_chars")),
        }
    if name == "submit_result":
        findings = arguments.get("findings")
        uncertainties = arguments.get("uncertainties")
        return {
            "status": _short_string(
                arguments.get("status", arguments.get("recommendation"))
            ),
            "summary_chars": _string_chars(arguments.get("summary")),
            "finding_count": len(findings) if isinstance(findings, list) else None,
            "uncertainty_count": (
                len(uncertainties) if isinstance(uncertainties, list) else None
            ),
            "argument_chars": len(_canonical_json(arguments)),
        }
    return {
        "argument_key_count": len(arguments),
        "argument_keys_sha256": _value_sha256(sorted(str(key) for key in arguments)),
        "argument_chars": len(_canonical_json(arguments)),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return {"type": type(value).__name__}


def _value_sha256(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _string_chars(value: object) -> int | None:
    return len(value) if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_non_negative_int(value: object) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _optional_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if isinstance(value, int) or math.isfinite(value) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _short_string(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > MAX_AUDIT_REASON_CHARS:
        return None
    return value


def _safe_reason(value: object) -> str | None:
    reason = _short_string(value)
    if not reason:
        return None
    if not all(
        character in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in reason
    ):
        return None
    return reason
