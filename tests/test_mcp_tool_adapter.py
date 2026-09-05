import asyncio

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from mycode.mcp.tool_adapter import (
    MAX_REGISTRY_NAME_LENGTH,
    MCPToolAdapter,
    build_registry_name,
)
from mycode.permissions import DefaultPermissionChecker, PermissionDecision
from mycode.tools import SyncTool, ToolArgumentValidationError, ToolRegistry


async def fake_call(alias, name, arguments):
    return CallToolResult(content=[TextContent(text=arguments["value"])])


def make_tool(annotations=None) -> Tool:
    return Tool(name="echo", description="Echo.", inputSchema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}, outputSchema={"type": "string"}, annotations=annotations)


def test_preserves_schema_namespace_and_remote_metadata() -> None:
    tool = MCPToolAdapter("local", make_tool(), fake_call)
    assert tool.name == "mcp__local__echo"
    assert tool.registry_name == "mcp__local__echo"
    assert tool.server_alias == "local"
    assert tool.remote_name == "echo"
    assert tool.get_schema()["parameters"] == make_tool().input_schema
    assert tool.output_schema == {"type": "string"}


@pytest.mark.parametrize(
    ("remote_name", "safe_fragment"),
    [
        ("admin.tools.list", "admin-tools-list"),
        ("admin tools/list", "admin-tools-list"),
    ],
)
def test_registry_name_sanitizes_invalid_characters_with_stable_hash(
    remote_name: str, safe_fragment: str
) -> None:
    first = build_registry_name("github", remote_name)
    second = build_registry_name("github", remote_name)

    assert first == second
    assert first.startswith(f"mcp__github__{safe_fragment}__")
    assert all(character.isalnum() or character in "_-" for character in first)


def test_registry_name_is_bounded_and_collision_resistant() -> None:
    long_name = "remote." + "x" * 100
    first_collision = build_registry_name("local", "a.b")
    second_collision = build_registry_name("local", "a/b")

    assert len(build_registry_name("local", long_name)) <= MAX_REGISTRY_NAME_LENGTH
    assert first_collision != second_collision
    assert build_registry_name("first", "echo") != build_registry_name("second", "echo")


def test_call_uses_original_remote_name_not_registry_name() -> None:
    calls = []

    async def recording_call(alias, name, arguments):
        calls.append((alias, name, arguments))
        return CallToolResult(content=[TextContent(text="ok")])

    remote = make_tool()
    remote.name = "admin.tools/list"
    tool = MCPToolAdapter("local", remote, recording_call)

    result = asyncio.run(
        tool.run_authorized_async(
            {"value": "hello"}, PermissionDecision.allow()
        )
    )

    assert result.ok is True
    assert calls == [("local", "admin.tools/list", {"value": "hello"})]


def test_validates_full_json_schema_before_permission() -> None:
    tool = MCPToolAdapter("local", make_tool(), fake_call)
    try:
        tool.parse_arguments({"value": 1, "extra": True})
    except ToolArgumentValidationError as error:
        assert error.errors
    else:
        raise AssertionError("validation should fail")


def test_schema_without_dialect_uses_draft_2020_12() -> None:
    remote = Tool(
        name="dependent",
        inputSchema={
            "type": "object",
            "dependentRequired": {"a": ["b"]},
        },
    )
    tool = MCPToolAdapter("local", remote, fake_call)

    with pytest.raises(ToolArgumentValidationError):
        tool.parse_arguments({"a": 1})


def test_schema_with_draft7_uri_uses_declared_dialect() -> None:
    remote = Tool(
        name="draft7",
        inputSchema={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "dependencies": {"a": ["b"]},
        },
    )
    tool = MCPToolAdapter("local", remote, fake_call)

    with pytest.raises(ToolArgumentValidationError):
        tool.parse_arguments({"a": 1})


def test_annotations_map_conservatively_to_existing_permission_profile() -> None:
    readonly = MCPToolAdapter("x", make_tool(ToolAnnotations(readOnlyHint=True, openWorldHint=False)), fake_call)
    external = MCPToolAdapter("x", make_tool(ToolAnnotations(readOnlyHint=True, openWorldHint=True)), fake_call)
    destructive = MCPToolAdapter("x", make_tool(ToolAnnotations(readOnlyHint=True, destructiveHint=True)), fake_call)
    unknown = MCPToolAdapter("x", make_tool(), fake_call)
    assert readonly.get_permission_profile().capability == "read"
    assert readonly.get_permission_profile().risk == "low"
    assert external.get_permission_profile().capability == "read"
    assert external.get_permission_profile().risk == "low"
    assert destructive.get_permission_profile().capability == "write"
    assert destructive.get_permission_profile().risk == "high"
    assert unknown.get_permission_profile().capability == "write"
    assert unknown.get_permission_profile().risk == "medium"

    request = unknown.build_permission_request({"value": "x"})
    decision = DefaultPermissionChecker().check(
        request, unknown.get_permission_profile()
    )
    assert decision.status == "ask"


def test_native_async_execution_normalizes_result() -> None:
    tool = MCPToolAdapter("local", make_tool(), fake_call)
    result = asyncio.run(tool.run_authorized_async({"value": "hello"}, PermissionDecision.allow()))
    assert result.ok is True
    assert result.content == "hello"
    assert result.metadata["server_alias"] == "local"


def test_native_async_adapter_has_no_fake_sync_execution_methods() -> None:
    assert not issubclass(MCPToolAdapter, SyncTool)
    assert "_run" not in MCPToolAdapter.__dict__
    assert "run_authorized" not in MCPToolAdapter.__dict__


def test_sync_registry_path_rejects_mcp_tool_without_calling_provider() -> None:
    called = False

    async def should_not_call(alias, name, arguments):
        nonlocal called
        called = True
        return CallToolResult(content=[TextContent(text="unexpected")])

    registry = ToolRegistry.from_tools(
        [MCPToolAdapter("local", make_tool(), should_not_call)]
    )

    result = registry.run_tool("mcp__local__echo", {"value": "hello"})

    assert result.error == "Tool requires async execution path"
    assert result.metadata["reason"] == "async_execution_required"
    assert called is False
