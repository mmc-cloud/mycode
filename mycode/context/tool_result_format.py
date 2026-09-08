"""Shared tool protocol grouping and compact result formatting (no storage policy)."""

from dataclasses import dataclass
import json

from mycode.messages import Message


TOOL_RESULT_METADATA_MARKER = "\n\nMETADATA\n"
COMPRESSED_TOOL_RESULT_MARKER = "[tool result compressed]"
LARGE_TOOL_METADATA_KEYS = {"stdout", "stderr"}
MAX_METADATA_STRING_CHARS = 200
MAX_TOOL_RESULT_PREVIEW_CHARS = 200


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
    threshold_chars: int,
    tool_names_by_id: dict[str, str],
) -> Message:
    if message.role != "tool":
        return message

    if len(message.content) <= threshold_chars:
        return message

    parsed = parse_tool_result_content(message.content)
    if parsed.metadata.get("context_compressed") is True:
        return message
    tool_name = tool_names_by_id.get(message.tool_call_id or "", "unknown")
    metadata = safe_tool_metadata(parsed.metadata)
    original_chars = len(message.content)
    if parsed.metadata.get("context_externalized") is True:
        original_chars = parsed.metadata.get("original_chars", original_chars)
        metadata.pop("context_externalized", None)
    for key in ("artifact_path", "artifact_sha256"):
        if key in parsed.metadata:
            metadata[key] = parsed.metadata[key]
    if parsed.result_preview and not parsed.metadata.get("context_externalized"):
        preview_key = "error_preview" if parsed.status == "ERROR" else "result_preview"
        metadata[preview_key] = parsed.result_preview
    metadata.update(
        {
            "context_compressed": True,
            "original_chars": original_chars,
            "tool_name": tool_name,
            "tool_call_id": message.tool_call_id,
        }
    )
    compressed_content = (
        f"{parsed.status}\n"
        f"{COMPRESSED_TOOL_RESULT_MARKER}\n"
        f"tool_name: {tool_name}\n"
        f"original_chars: {original_chars}\n"
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
