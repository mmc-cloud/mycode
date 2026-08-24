from pathlib import Path
import re
import shutil
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.command_executor import CommandExecutionArgs, execute_command
from mycode.tools.ignore import is_sensitive_path
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.permission_metadata import with_permission_metadata
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_INSPECT_TIMEOUT_SECONDS = 30.0
DEFAULT_INSPECT_MAX_OUTPUT_CHARS = 20000
MAX_INSPECT_OUTPUT_CHARS = 100000
MAX_INSPECT_PATHS = 20
SAFE_REF_PATTERN = re.compile(
    r"(?:HEAD|[A-Za-z0-9][A-Za-z0-9._/-]*)(?:[~^][0-9]+)?"
)

InspectChangesAction = Literal["status", "diff"]


class InspectChangesArgs(ToolArgs):
    action: InspectChangesAction
    paths: list[str] = Field(default_factory=list, max_length=MAX_INSPECT_PATHS)
    staged: bool = False
    base_ref: str | None = Field(default=None, max_length=200)
    max_output_chars: int = Field(
        default=DEFAULT_INSPECT_MAX_OUTPUT_CHARS,
        ge=0,
        le=MAX_INSPECT_OUTPUT_CHARS,
    )

    @field_validator("paths")
    @classmethod
    def validate_plain_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            path = value.strip()
            if path == "":
                raise ValueError("inspect_changes paths must not be blank.")
            if len(path) > 500:
                raise ValueError("inspect_changes paths must not exceed 500 characters.")
            if path.startswith(":") or any(marker in path for marker in ("*", "?", "[", "]", "\x00")):
                raise ValueError("inspect_changes paths must be literal paths, not pathspecs.")
            normalized.append(path)
        return normalized

    @field_validator("base_ref")
    @classmethod
    def validate_base_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            normalized == ""
            or normalized.startswith("-")
            or SAFE_REF_PATTERN.fullmatch(normalized) is None
        ):
            raise ValueError("base_ref must be a plain Git revision, not an option.")
        return normalized

    @model_validator(mode="after")
    def action_arguments_must_match(self) -> Self:
        if self.action == "status" and (
            self.paths or self.staged or self.base_ref is not None
        ):
            raise ValueError(
                "status does not accept paths, staged, or base_ref arguments."
            )
        if self.action == "diff" and not self.paths:
            raise ValueError("diff requires at least one explicit path.")
        return self


class InspectChangesTool(BaseTool[InspectChangesArgs]):
    name = "inspect_changes"
    description = (
        "Inspect Git status or a bounded diff for explicit non-sensitive workspace paths."
    )
    args_model = InspectChangesArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: InspectChangesArgs) -> PermissionRequest:
        target = "repository status" if args.action == "status" else ", ".join(args.paths)
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=f"{self.name}.{args.action}",
            target=target,
            arguments=args.model_dump(),
            description="Inspect repository changes without arbitrary command execution.",
        )

    def check_permission(
        self,
        args: InspectChangesArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        resolved_paths: list[str] = []
        for path in args.paths:
            try:
                resolved = self.workspace.resolve_path(path)
            except WorkspacePathError:
                return request, PermissionDecision.deny(
                    reason="outside_workspace",
                    message=f"inspect_changes path is outside workspace: {path}",
                    metadata={
                        "path": path,
                        "workspace_root": self.workspace.root.as_posix(),
                        "inspect_action": args.action,
                    },
                )
            if is_sensitive_path(resolved, self.workspace.root):
                return request, PermissionDecision.deny(
                    reason="sensitive_path",
                    message=f"inspect_changes refuses sensitive path content: {path}",
                    metadata={
                        "path": path,
                        "resolved_path": resolved.as_posix(),
                        "workspace_root": self.workspace.root.as_posix(),
                        "inspect_action": args.action,
                    },
                )

            path_decision = PathPermissionPolicy(self.workspace).check_path(request, path)
            if path_decision.status != "allow":
                return request, with_permission_metadata(
                    path_decision,
                    {
                        "inspect_action": args.action,
                        "resolved_paths": [*resolved_paths, resolved.as_posix()],
                        "staged": args.staged,
                        "base_ref": args.base_ref,
                    },
                )
            resolved_paths.append(resolved.as_posix())

        decision = permission_checker.check(request, self.get_permission_profile())
        if decision.status != "allow":
            return request, decision
        return request, PermissionDecision.allow(
            message=decision.message,
            metadata={
                **decision.metadata,
                "workspace_root": self.workspace.root.as_posix(),
                "resolved_cwd": self.workspace.root.as_posix(),
                "inspect_action": args.action,
                "resolved_paths": resolved_paths,
                "staged": args.staged,
                "base_ref": args.base_ref,
            },
        )

    def run_authorized(
        self,
        args: InspectChangesArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            return self._run_authorized(args, decision)
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata={
                    **decision.metadata,
                    "exception_type": type(error).__name__,
                },
            )

    def _run_authorized(
        self,
        args: InspectChangesArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        git_executable = shutil.which("git")
        if git_executable is None:
            return ToolResult.failure(
                error="Git executable was not found.",
                metadata={**decision.metadata, "reason": "git_not_found"},
            )
        resolved_git = Path(git_executable).resolve(strict=False)
        if resolved_git.is_relative_to(self.workspace.root):
            return ToolResult.failure(
                error="Refusing to execute a workspace-local Git executable.",
                metadata={
                    **decision.metadata,
                    "reason": "workspace_local_executable",
                    "git_executable": resolved_git.as_posix(),
                },
            )

        command = self._build_command(args, resolved_git)
        safe_git_flags = [
            "--no-optional-locks",
            "-c core.fsmonitor=false",
            "--no-pager",
            "--ignore-submodules=all",
        ]
        if args.action == "diff":
            safe_git_flags.extend(["--no-ext-diff", "--no-textconv"])
        return execute_command(
            args=CommandExecutionArgs(
                command=command,
                timeout_seconds=DEFAULT_INSPECT_TIMEOUT_SECONDS,
                max_output_chars=args.max_output_chars,
            ),
            cwd=self.workspace.root,
            permission_metadata={
                **decision.metadata,
                "command": command,
                "safe_git_flags": safe_git_flags,
            },
            permission_status=decision.status,
        )

    def _run(self, args: InspectChangesArgs) -> ToolResult:
        return ToolResult.failure(
            error="inspect_changes must be run through ToolRegistry.run_tool().",
            metadata={"action": args.action, "reason": "permission_required"},
        )

    def _build_command(self, args: InspectChangesArgs, git_executable: Path) -> list[str]:
        command = [
            git_executable.as_posix(),
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "--no-pager",
        ]
        if args.action == "status":
            return [
                *command,
                "status",
                "--short",
                "--branch",
                "--untracked-files=all",
                "--ignore-submodules=all",
            ]

        command.extend(
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=all",
            ]
        )
        if args.staged:
            command.append("--cached")
        if args.base_ref is not None:
            command.append(args.base_ref)
        resolved_paths = [
            self.workspace.resolve_path(path).relative_to(self.workspace.root).as_posix()
            for path in args.paths
        ]
        command.extend(["--", *resolved_paths])
        return command
