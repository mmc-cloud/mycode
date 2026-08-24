from pathlib import Path
import json
import sys

from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.tools import RunValidationTool, ToolRegistry, Workspace
from mycode.tools.validation_command import analyze_validation_command


def test_validation_policy_allows_expected_test_compile_and_lint_commands() -> None:
    commands = [
        (["uv", "run", "pytest", "-q"], "test"),
        ([sys.executable, "-m", "compileall", "mycode"], "compile"),
        (["ruff", "check", "mycode"], "lint"),
        (["ruff", "format", "--check", "mycode"], "lint"),
        (["mypy", "mycode"], "lint"),
    ]

    for command, category in commands:
        analysis = analyze_validation_command(command)
        assert analysis.allowed is True
        assert analysis.category == category


def test_validation_policy_rejects_arbitrary_or_mutating_commands() -> None:
    commands = [
        [sys.executable, "-c", "print('hello')"],
        ["pwsh", "-Command", "pytest"],
        ["uv", "pip", "install", "package"],
        ["ruff", "check", "--fix", "mycode"],
        ["ruff", "check", "--fix=true", "mycode"],
        ["git", "status"],
    ]

    for command in commands:
        analysis = analyze_validation_command(command)
        assert analysis.allowed is False
        assert analysis.category == "unsupported"


def test_run_validation_requires_confirmation_for_allowed_command(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools([RunValidationTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "run_validation",
        {"command": [sys.executable, "-m", "compileall", "-q", "."]},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["validation_allowed"] is True
    assert result.metadata["validation_category"] == "compile"
    assert result.metadata["permission_status"] == "ask"


def test_run_validation_executes_after_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [RunValidationTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_validation",
        {"command": [sys.executable, "-m", "compileall", "-q", "."]},
    )

    assert result.ok is True
    assert result.metadata["exit_code"] == 0
    assert result.metadata["duration_ms"] >= 0
    assert result.metadata["validation_category"] == "compile"
    assert result.metadata["confirmation_status"] == "approved"


def test_run_validation_applies_allowlist_after_json_argv_normalization(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunValidationTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {
            "command": json.dumps(
                [sys.executable, "-m", "compileall", "-q", "."]
            )
        },
    )

    assert result.ok is True
    assert result.metadata["validation_allowed"] is True
    assert result.metadata["validation_category"] == "compile"
    assert confirmer.requests[0].permission_request.arguments["command"] == [
        sys.executable,
        "-m",
        "compileall",
        "-q",
        ".",
    ]


def test_run_validation_denies_unknown_command_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunValidationTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {"command": [sys.executable, "-c", "print('hello')"]},
    )

    assert result.ok is False
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "unsupported_operation"
    assert result.metadata["validation_allowed"] is False
    assert confirmer.requests == []


def test_main_agent_validation_mode_uses_normal_command_risk_for_project_runner(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [
            RunValidationTool(
                Workspace(tmp_path),
                restrict_to_known_validators=False,
            )
        ],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {"command": [sys.executable, "manage.py", "test"]},
    )

    assert result.ok is False
    assert result.metadata["command_risk_category"] == "python_execution"
    assert result.metadata["command_risk_decision"] == "ask"
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["exit_code"] == 2
    assert len(confirmer.requests) == 1


def test_main_agent_validation_mode_keeps_dangerous_command_denied(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [
            RunValidationTool(
                Workspace(tmp_path),
                restrict_to_known_validators=False,
            )
        ],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {
            "command": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('notes.txt').unlink()",
            ]
        },
    )

    assert result.ok is False
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert "exit_code" not in result.metadata
    assert confirmer.requests == []


def test_run_validation_keeps_workspace_cwd_boundary(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunValidationTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {
            "command": [sys.executable, "-m", "compileall", "-q", "."],
            "cwd": "..",
        },
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "outside_workspace"
    assert confirmer.requests == []


def test_run_validation_rejects_oversized_command_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunValidationTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_validation",
        {
            "command": [
                sys.executable,
                "-m",
                "pytest",
                *[f"tests/test_{index}_{'x' * 100}.py" for index in range(50)],
            ]
        },
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert confirmer.requests == []


class ApprovingConfirmer:
    def __init__(self) -> None:
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return ConfirmationResult.approved()
