from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from mycode.mcp.result_adapter import adapt_mcp_result


def test_normalizes_text_structured_resources_and_multimodal_fallback() -> None:
    result = adapt_mcp_result(CallToolResult(
        content=[
            TextContent(text="hello"),
            ResourceLink(name="doc", uri="file:///doc", description="Docs"),
            EmbeddedResource(resource=TextResourceContents(uri="file:///embedded", text="embedded")),
            ImageContent(data="not-retained", mimeType="image/png"),
        ],
        structuredContent={"answer": 42},
    ))
    assert result.ok is True
    assert "hello" in result.content
    assert "file:///doc" in result.content
    assert "embedded" in result.content
    assert "image content omitted" in result.content
    assert result.metadata["structured_content"] == {"answer": 42}
    assert "not-retained" not in repr(result.metadata)


def test_is_error_becomes_tool_result_failure() -> None:
    result = adapt_mcp_result(CallToolResult(
        content=[TextContent(text="remote failed")], isError=True
    ))
    assert result.ok is False
    assert result.error == "remote failed"
