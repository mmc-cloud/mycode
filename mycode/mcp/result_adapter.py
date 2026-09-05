import json

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from mycode.tools.base import ToolResult


def adapt_mcp_result(result: CallToolResult) -> ToolResult:
    parts: list[str] = []
    blocks: list[dict[str, object]] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
            blocks.append({"type": "text"})
        elif isinstance(block, ResourceLink):
            parts.append(f"Resource: {block.uri}" + (f" ({block.description})" if block.description else ""))
            blocks.append(_without_data(block.model_dump(by_alias=True, exclude_none=True)))
        elif isinstance(block, EmbeddedResource):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                parts.append(resource.text)
            elif isinstance(resource, BlobResourceContents):
                parts.append(f"[embedded binary resource omitted: {resource.uri}]")
            blocks.append(_without_data(block.model_dump(by_alias=True, exclude_none=True)))
        elif isinstance(block, (ImageContent, AudioContent)):
            parts.append(f"[{block.type} content omitted: {block.mime_type}]")
            blocks.append(_without_data(block.model_dump(by_alias=True, exclude_none=True)))
        else:  # pragma: no cover - forward-compatible SDK content type
            parts.append(f"[unsupported MCP content omitted: {type(block).__name__}]")
            blocks.append({"type": type(block).__name__})

    metadata: dict[str, object] = {"mcp_content_blocks": blocks}
    if result.structured_content is not None:
        metadata["structured_content"] = result.structured_content
        if not parts:
            parts.append(json.dumps(result.structured_content, ensure_ascii=False, default=str))
    content = "\n".join(part for part in parts if part != "")
    if result.is_error:
        return ToolResult.failure(content or "MCP tool returned an error", metadata)
    return ToolResult.success(content, metadata)


def _without_data(value: dict[str, object]) -> dict[str, object]:
    value.pop("data", None)
    resource = value.get("resource")
    if isinstance(resource, dict):
        resource.pop("blob", None)
        resource.pop("text", None)
    return value
