import json
import subprocess
from pathlib import Path

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.command_executor import CommandExecutionArgs, execute_command
from mycode.tools.command_risk import CommandRiskAnalysis, analyze_command_risk
from mycode.tools.permission_metadata import with_permission_metadata
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 12000
MAX_TIMEOUT_SECONDS = 300.0
MAX_OUTPUT_CHARS = 200000
COMMAND_DESCRIPTION = (
    "要执行的 argv 字符串数组。第一个元素是可执行程序，后续元素是参数。"
    "不是 Shell 命令字符串，不支持 &&、||、|、>、< 等 Shell 操作符，也不会展开"
    " *.py 等通配符。需要切换工作目录时使用 cwd，不要执行 cd。例如："
    "[\"pytest\", \"tests/test_a.py\", \"-q\"]。"
)
COMMAND_VALIDATION_ERROR = (
    "command 必须是 argv 字符串数组，而不是 Shell 命令字符串。请使用 cwd 代替 cd；"
    "不支持 &&、||、|、>、< 等 Shell 语法。"
)
CWD_DESCRIPTION = (
    "命令执行时的工作目录，必须位于 workspace 内。需要切换目录时使用该参数，不要执行 cd。"
)


class RunCommandArgs(ToolArgs):
    command: list[str] = Field(min_length=1, description=COMMAND_DESCRIPTION)
    cwd: str = Field(default=".", min_length=1, description=CWD_DESCRIPTION)
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_TIMEOUT_SECONDS,
    )
    max_output_chars: int = Field(
        default=DEFAULT_MAX_OUTPUT_CHARS,
        ge=0,
        le=MAX_OUTPUT_CHARS,
    )

    @field_validator("command", mode="before")
    @classmethod
    def normalize_json_encoded_command(cls, value: object) -> object:
        """Accept one common Provider mistake without introducing shell parsing."""

        if not isinstance(value, str):
            if not isinstance(value, (list, tuple)) or not value or any(
                not isinstance(part, str) for part in value
            ):
                raise ValueError(COMMAND_VALIDATION_ERROR)
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(COMMAND_VALIDATION_ERROR) from error
        if not isinstance(decoded, list) or not decoded or any(
            not isinstance(part, str) for part in decoded
        ):
            raise ValueError(COMMAND_VALIDATION_ERROR)
        return decoded

    @field_validator("command")
    @classmethod
    def command_parts_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if any(part == "" for part in value):
            raise ValueError(COMMAND_VALIDATION_ERROR)
        if any("\x00" in part for part in value):
            raise ValueError(COMMAND_VALIDATION_ERROR)

        return value


class RunCommandTool(BaseTool[RunCommandArgs]):
    name = "run_command"
    description = (
        "在 workspace 内执行非交互命令。command 使用结构化 argv，命令直接执行，不经过 Shell 解释。"
    )
    args_model = RunCommandArgs
    capability = "command"
    risk = "high"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: RunCommandArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=_display_command(args.command),
            arguments=args.model_dump(),
            description="Run a non-interactive command.",
        )

    def check_permission(
        self,
        args: RunCommandArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        try:
            resolved_cwd = self.workspace.resolve_path(args.cwd)
        except WorkspacePathError:
            return request, PermissionDecision.deny(
                reason="outside_workspace",
                message=f"Command working directory is outside workspace: {args.cwd}",
                metadata=_command_metadata(
                    args=args,
                    workspace_root=self.workspace.root,
                    resolved_cwd=_resolve_candidate_path(self.workspace, args.cwd),
                    cwd_scope="outside_workspace",
                ),
            )

        risk_analysis = analyze_command_risk(args.command)
        metadata = _command_metadata(
            args=args,
            workspace_root=self.workspace.root,
            resolved_cwd=resolved_cwd,
            cwd_scope="inside_workspace",
            risk_analysis=risk_analysis,
        )

        if not resolved_cwd.exists():
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Command working directory does not exist: {args.cwd}",
                metadata=metadata,
            )

        if not resolved_cwd.is_dir():
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Command working directory is not a directory: {args.cwd}",
                metadata=metadata,
            )

        if risk_analysis.decision == "deny":
            return request, PermissionDecision.deny(
                reason="dangerous_command",
                message=risk_analysis.reason,
                metadata=metadata,
            )

        decision = permission_checker.check(
            request,
            self.get_permission_profile(),
        )

        return request, with_permission_metadata(decision, metadata)

    def run_authorized(
        self,
        args: RunCommandArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            cwd = Path(str(decision.metadata["resolved_cwd"]))
            return execute_command(
                args=CommandExecutionArgs(
                    command=args.command,
                    timeout_seconds=args.timeout_seconds,
                    max_output_chars=args.max_output_chars,
                ),
                cwd=cwd,
                permission_metadata=decision.metadata,
                permission_status=decision.status,
            )
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata={
                    **decision.metadata,
                    "permission_status": decision.status,
                    "exception_type": type(error).__name__,
                },
            )

    def _run(self, args: RunCommandArgs) -> ToolResult:
        return ToolResult.failure(
            error="run_command must be run through ToolRegistry.run_tool().",
            metadata={
                "command": args.command,
                "cwd": args.cwd,
                "reason": "permission_required",
            },
        )


def _resolve_candidate_path(workspace: Workspace, path: str) -> Path:
    requested_path = Path(path)
    if requested_path.is_absolute():
        return requested_path.resolve(strict=False)

    return (workspace.root / requested_path).resolve(strict=False)


def _command_metadata(
    *,
    args: RunCommandArgs,
    workspace_root: Path,
    resolved_cwd: Path,
    cwd_scope: str,
    risk_analysis: CommandRiskAnalysis | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "command": list(args.command),
        "command_display": _display_command(args.command),
        "cwd": args.cwd,
        "resolved_cwd": resolved_cwd.as_posix(),
        "workspace_root": workspace_root.as_posix(),
        "cwd_scope": cwd_scope,
        "timeout_seconds": args.timeout_seconds,
        "max_output_chars": args.max_output_chars,
    }
    if risk_analysis is not None:
        metadata.update(
            {
                "command_risk_category": risk_analysis.category,
                "command_risk": risk_analysis.risk,
                "command_risk_decision": risk_analysis.decision,
                "command_risk_reason": risk_analysis.reason,
            }
        )

    return metadata


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)
