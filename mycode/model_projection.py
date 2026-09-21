"""Deterministic, provider-neutral projections of canonical model messages."""

from collections.abc import Iterable

from mycode.messages import Message


def project_model_message(
    message: Message,
    *,
    include_reasoning: bool = True,
) -> dict[str, object]:
    projection: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        projection["tool_calls"] = [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": dict(tool_call.arguments),
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        projection["tool_call_id"] = message.tool_call_id
    if include_reasoning and message.reasoning_state != "absent":
        projection["reasoning_content"] = message.reasoning_content
    return projection


def project_model_messages(
    messages: Iterable[Message],
    *,
    include_reasoning: bool = True,
) -> list[dict[str, object]]:
    return [
        project_model_message(message, include_reasoning=include_reasoning)
        for message in messages
    ]
