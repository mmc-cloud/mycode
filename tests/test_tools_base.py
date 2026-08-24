import pytest

from mycode.tools import BaseTool, ToolArgs, ToolPermissionProfileError, ToolResult


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


def test_base_tool_get_schema_uses_pydantic_args_model() -> None:
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


class FakeTool(BaseTool[FakeArgs]):
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


class BrokenTool(BaseTool[FakeArgs]):
    name = "broken_tool"
    description = "Broken tool for tests."
    args_model = FakeArgs
    capability = "read"
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        raise RuntimeError("boom")


class MissingCapabilityTool(BaseTool[FakeArgs]):
    name = "missing_capability"
    description = "Tool missing capability for tests."
    args_model = FakeArgs
    risk = "low"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(args.text)


class MissingRiskTool(BaseTool[FakeArgs]):
    name = "missing_risk"
    description = "Tool missing risk for tests."
    args_model = FakeArgs
    capability = "read"

    def _run(self, args: FakeArgs) -> ToolResult:
        return ToolResult.success(args.text)
