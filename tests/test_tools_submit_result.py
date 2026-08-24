from mycode.subagents.contracts import ExplorerResult
from mycode.tools import ToolRegistry
from mycode.tools.submit_result import SubmitResultTool


def _valid_result() -> dict[str, object]:
    return {
        "status": "no_match",
        "summary": "No matching implementation was found.",
        "searched_scope": ["mycode"],
        "findings": [],
        "uncertainties": [],
    }


def test_submit_result_schema_uses_role_specific_model() -> None:
    tool = SubmitResultTool(role="explorer", result_model=ExplorerResult)

    schema = tool.get_schema()

    assert schema["name"] == "submit_result"
    assert set(schema["parameters"]["required"]) >= {
        "status",
        "summary",
        "searched_scope",
    }
    assert "findings" in schema["parameters"]["properties"]


def test_submit_result_uses_control_capability_without_confirmation() -> None:
    tool = SubmitResultTool(role="explorer", result_model=ExplorerResult)
    registry = ToolRegistry.from_tools([tool])

    result = registry.run_tool("submit_result", _valid_result())

    assert result.ok is True
    assert result.content == "Structured SubAgent result accepted."
    assert result.metadata["role"] == "explorer"
    assert result.metadata["outcome"] == "no_match"
    assert tool.get_permission_profile().capability == "control"
    assert isinstance(tool.submitted_result, ExplorerResult)


def test_submit_result_rejects_invalid_role_payload() -> None:
    tool = SubmitResultTool(role="explorer", result_model=ExplorerResult)
    registry = ToolRegistry.from_tools([tool])

    result = registry.run_tool(
        "submit_result",
        {"status": "completed", "summary": "Missing scope and evidence."},
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert tool.submitted_result is None


def test_submit_result_rejects_oversized_serialized_result() -> None:
    tool = SubmitResultTool(
        role="explorer",
        result_model=ExplorerResult,
        max_result_chars=20,
    )
    registry = ToolRegistry.from_tools([tool])

    result = registry.run_tool("submit_result", _valid_result())

    assert result.ok is False
    assert result.metadata["reason"] == "result_too_large"
    assert result.metadata["result_chars"] > 20
    assert tool.submitted_result is None


def test_submit_result_accepts_only_one_result() -> None:
    tool = SubmitResultTool(role="explorer", result_model=ExplorerResult)
    registry = ToolRegistry.from_tools([tool])

    first = registry.run_tool("submit_result", _valid_result())
    second = registry.run_tool("submit_result", _valid_result())

    assert first.ok is True
    assert second.ok is False
    assert second.metadata["reason"] == "result_already_submitted"


def test_submit_result_runtime_validator_can_reject_then_accept() -> None:
    validation_errors = ["Runtime evidence does not match the submitted status.", None]
    tool = SubmitResultTool(
        role="explorer",
        result_model=ExplorerResult,
        acceptance_validator=lambda args: validation_errors.pop(0),
    )
    registry = ToolRegistry.from_tools([tool])

    rejected = registry.run_tool("submit_result", _valid_result())
    assert tool.submitted_result is None
    accepted = registry.run_tool("submit_result", _valid_result())

    assert rejected.ok is False
    assert rejected.metadata["reason"] == "result_runtime_validation_failed"
    assert accepted.ok is True
    assert tool.submitted_result is not None
