import subprocess

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.subagents.limits import (
    MAX_VALIDATION_COMMAND_CHARS,
    MAX_VALIDATION_COMMAND_PART_CHARS,
    MAX_VALIDATION_COMMAND_PARTS,
)
from mycode.tools.base import ToolResult
from mycode.tools.permission_metadata import with_permission_metadata
from mycode.tools.run_command import RunCommandArgs, RunCommandTool
from mycode.tools.validation_command import analyze_validation_command
from mycode.tools.workspace import Workspace


class RunValidationArgs(RunCommandArgs):
    command: list[str] = Field(
        min_length=1,
        max_length=MAX_VALIDATION_COMMAND_PARTS,
    )
    cwd: str = Field(default=".", min_length=1, max_length=500)

    @field_validator("command")
    @classmethod
    def validation_command_must_be_bounded(cls, value: list[str]) -> list[str]:
        if any(len(part) > MAX_VALIDATION_COMMAND_PART_CHARS for part in value):
            raise ValueError(
                "Validation command parts must not exceed "
                f"{MAX_VALIDATION_COMMAND_PART_CHARS} characters."
            )
        if sum(len(part) for part in value) > MAX_VALIDATION_COMMAND_CHARS:
            raise ValueError(
                "Validation command must not exceed "
                f"{MAX_VALIDATION_COMMAND_CHARS} total characters."
            )
        return value


class RunValidationTool(RunCommandTool):
    name = "run_validation"
    description = (
        "Run a non-interactive validation command inside the workspace using "
        "the same safety and permission checks as run_command."
    )
    args_model = RunValidationArgs

    def __init__(
        self,
        workspace: Workspace,
        *,
        restrict_to_known_validators: bool = True,
    ) -> None:
        super().__init__(workspace)
        self.restrict_to_known_validators = restrict_to_known_validators

    def build_permission_request(self, args: RunCommandArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=subprocess.list2cmdline(args.command),
            arguments=args.model_dump(),
            description="Run a validation command.",
        )

    def check_permission(
        self,
        args: RunCommandArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        if not self.restrict_to_known_validators:
            return super().check_permission(args, permission_checker)

        request = self.build_permission_request(args)
        validation = analyze_validation_command(args.command)
        validation_metadata = {
            "validation_allowed": validation.allowed,
            "validation_classification": validation.classification,
            "validation_category": validation.category,
            "validation_reason": validation.reason,
        }
        if not validation.allowed:
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=validation.reason,
                metadata={
                    "tool_name": self.name,
                    "command": list(args.command),
                    "cwd": args.cwd,
                    **validation_metadata,
                },
            )

        request, decision = super().check_permission(args, permission_checker)
        return request, with_permission_metadata(decision, validation_metadata)

    def _run(self, args: RunCommandArgs) -> ToolResult:
        return ToolResult.failure(
            error="run_validation must be run through ToolRegistry.run_tool().",
            metadata={
                "command": args.command,
                "cwd": args.cwd,
                "reason": "permission_required",
            },
        )
