import asyncio

import pytest

from mycode.permissions import PermissionDecision
from mycode.tools import (
    BaseTool,
    PydanticTool,
    SyncTool,
    ToolArgs,
    ToolPermissionProfileError,
    ToolResult,
)


def test_tool_result_success_sets_content_and_metadata() -> None:
    result = ToolResult.success(
        content="file content",
        metadata={"path": "mycode/cli.py"},
    )

    assert result.ok is True
    assert result.content == "file content"
    assert result.error is None
    assert result.metadata == {"path": "mycode/cli.py"}


def test_tool_result_failure_sets_error_and_metadata() -> None:
    result = ToolResult.failure(
        error="File not found",
        metadata={"path": "missing.py"},
    )

    assert result.ok is False
    assert result.content == ""
    assert result.error == "File not found"
    assert result.metadata == {"path": "missing.py"}


def test_tool_result_metadata_defaults_are_independent() -> None:
    first = ToolResult.success("first")
    second = ToolResult.success("second")

    assert first.metadata is not second.metadata
    assert second.metadata == {}


def test_tool_result_copies_metadata() -> None:
    metadata: dict[str, object] = {"path": "mycode/cli.py"}

    result = ToolResult.success("content", metadata=metadata)
    metadata["path"] = "changed.py"

    assert result.metadata == {"path": "mycode/cli.py"}


def test_base_tool_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseTool()


def test_base_tool_does_not_require_sync_run_contract() -> None:
    tool = AsyncDictTool()

    assert "_run" not in BaseTool.__abstractmethods__
    result = asyncio.run(
        tool.run_authorized_async(
            {"text": "hello"}, PermissionDecision.allow()
        )
    )
    assert result == ToolResult.success("hello")


def test_sync_tool_supports_non_pydantic_schema_and_async_adapter() -> None:
    tool = DictTool()

    assert tool.get_schema()["parameters"] == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    assert tool.run({"text": "hello"}) == ToolResult.success("hello")
    assert asyncio.run(
        tool.run_authorized_async(
            {"text": "hello"}, PermissionDecision.allow()
        )
    ) == ToolResult.success("hello")
    assert not hasattr(tool, "args_model")


def test_pydantic_tool_get_schema_uses_pydantic_args_model() -> None:
    tool = FakeTool()

    schema = tool.get_schema()

    assert schema == {
        "name": "fake_tool",
        "description": "Fake tool for tests.",
        "parameters": {
            "additionalProperties": False,
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                },
            },
            "required": ["text"],
        },
    }


def test_base_tool_schema_does_not_include_permission_profile() -> None:
    tool = FakeTool()

    schema = tool.get_schema()

    assert "capability" not in schema
    assert "risk" not in schema


def test_base_tool_returns_permission_profile() -> None:
    tool = FakeTool()

    profile = tool.get_permission_profile()

    assert profile.capability == "read"
    assert profile.risk == "low"


def test_base_tool_builds_default_permission_request() -> None:
    tool = FakeTool()

    request = tool.build_permission_request(FakeArgs(text="hello"))

    assert request.tool_name == "fake_tool"
    assert request.capability == "read"
    assert request.action == "fake_tool"
    assert request.target is None
    assert request.arguments == {"text": "hello"}


def test_base_tool_requires_capability_declaration() -> None:
    tool = MissingCapabilityTool()

    with pytest.raises(
        ToolPermissionProfileError,
        match="Tool must declare capability: missing_capability",
    ):
        tool.get_permission_profile()


def test_base_tool_requires_risk_declaration() -> None:
    tool = MissingRiskTool()

    with pytest.raises(
        ToolPermissionProfileError,
        match="Tool must declare risk: missing_risk",
    ):
        tool.get_permission_profile()


def test_parse_arguments_returns_typed_args() -> None:
    tool = FakeTool()

    args = tool.parse_arguments({"text": "hello"})

    assert args == FakeArgs(text="hello")


def test_concrete_tool_run_returns_tool_result() -> None:
    tool = FakeTool()

    result = tool.run({"text": "hello"})

    assert result == ToolResult.success(
        content="hello",
        metadata={"text_length": 5},
    )


def test_concrete_tool_run_returns_failure_for_invalid_arguments() -> None:
    tool = FakeTool()

    result = tool.run({"text": 123})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert result.metadata["validation_errors"] != []
    assert tool.run_count == 0


def test_concrete_tool_run_returns_failure_when_execution_raises() -> None:
    tool = BrokenTool()

    result = tool.run({"text": "hello"})

    assert result == ToolResult.failure(
        error="Tool execution failed: boom",
        metadata={"exception_type": "RuntimeError"},
    )


def test_tool_args_rejects_unknown_fields() -> None:
    tool = FakeTool()

    result = tool.run({"text": "hello", "unknown": "value"})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert tool.run_count == 0


class FakeArgs(ToolArgs):
    text: str


class DictTool(SyncTool[dict[str, object]]):
    name = "dict_tool"
    description = "Non-Pydantic tool for contract tests."
    capability = "read"
    risk = "low"

    @property
    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def parse_arguments(
        self, arguments: dict[str, object]
    ) -> dict[str, object]:
        if not isinstance(arguments.get("text"), str):
            raise TypeError("text must be a string")
        return dict(arguments)

    def arguments_to_dict(
        self, args: dict[str, object]
    ) -> dict[str, object]:
        return dict(args)

    def _run(self, args: dict[str, object]) -> ToolResult:
        return ToolResult.success(str(args["text"]))


class AsyncDictTool(BaseTool[dict[str, object]]):
    name = "async_dict_tool"
    description = "Native async tool for contract tests."
    capability = "read"
    risk = "low"

    @property
    def input_schema(self) -> dict[str, object]:
        return DictTool().input_schema

    def parse_arguments(
        self, arguments: dict[str, object]
    ) -> dict[str, object]:
        return DictTool().parse_arguments(arguments)

    def arguments_to_dict(
        self, args: dict[str, object]
    ) -> dict[str, object]:
        return dict(args)

    async def run_authorized_async(
        self,
        args: dict[str, object],
        decision: PermissionDecision,
    ) -> ToolResult:
        return ToolResult.success(str(args["text"]))


class FakeTool(PydanticTool[FakeArgs]):
    name = "fake_tool"
    description = "Fake tool for tests."
    args_model = FakeArgs
    capability = "read"
    risk = "low"

    def __init__(self) -> None:
        self.run_count = 0

    def _run(self, args: FakeArgs) -> ToolResult:
        self.run_count += 1
        return ToolResult.success(
            content=args.text,
            metadata={"text_length": len(args.text)},
        )


class BrokenTool(PydanticTool[FakeArgs]):
    name = "broken_tool"
    description = "Broken tool for tests."
    args_model = FakeArgs
    capability = "read"
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        raise RuntimeError("boom")


class MissingCapabilityTool(PydanticTool[FakeArgs]):
    name = "missing_capability"
    description = "Tool missing capability for tests."
    args_model = FakeArgs
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(args.text)


class MissingRiskTool(PydanticTool[FakeArgs]):
    name = "missing_risk"
    description = "Tool missing risk for tests."
    args_model = FakeArgs
    capability = "read"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(args.text)
