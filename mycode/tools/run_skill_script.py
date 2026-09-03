from pathlib import Path
import shutil
import sys

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.skills import ActiveSkillState, SkillPathError, SkillRegistry
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.command_executor import CommandExecutionArgs, execute_command
from mycode.tools.permission_metadata import with_permission_metadata
from mycode.tools.run_command import (
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS,
    MAX_TIMEOUT_SECONDS,
)
from mycode.tools.workspace import Workspace


class RunSkillScriptArgs(ToolArgs):
    skill: str = Field(min_length=1)
    script: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS, gt=0, le=MAX_TIMEOUT_SECONDS
    )
    max_output_chars: int = Field(
        default=DEFAULT_MAX_OUTPUT_CHARS, ge=0, le=MAX_OUTPUT_CHARS
    )

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        if any(not isinstance(part, str) or part == "" for part in value):
            raise ValueError("Script arguments must be non-empty strings.")
        if any("\x00" in part for part in value):
            raise ValueError("Script arguments must not contain null bytes.")
        return value


class RunSkillScriptTool(BaseTool[RunSkillScriptArgs]):
    name = "run_skill_script"
    description = "在当前 workspace 中运行已激活 Skill 的受支持脚本。"
    args_model = RunSkillScriptArgs
    capability = "command"
    risk = "high"

    def __init__(
        self,
        workspace: Workspace,
        registry: SkillRegistry,
        state: ActiveSkillState,
    ) -> None:
        self.workspace = workspace
        self.registry = registry
        self.state = state

    def build_permission_request(self, args: RunSkillScriptArgs) -> PermissionRequest:
        logical_script = _logical_script_path(args.script)
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=f"{args.skill}:{logical_script}",
            arguments=args.model_dump(),
            description=(
                f'Skill "{args.skill}" 请求在 workspace 中执行 {logical_script}，'
                f"参数为 {args.args!r}。"
            ),
        )

    def check_permission(
        self,
        args: RunSkillScriptArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        skill = self.registry.get(args.skill)
        metadata = {
            "skill_name": args.skill,
            "script": _logical_script_path(args.script),
            "cwd": str(self.workspace.root),
            "arguments": list(args.args),
        }
        if skill is None or not self.state.is_active(args.skill):
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Skill is not active: {args.skill}",
                metadata=metadata,
            )
        metadata["skill_source"] = skill.source
        try:
            script_path = self.registry.resolve_script(skill, args.script)
        except SkillPathError as error:
            return request, PermissionDecision.deny(
                reason="outside_workspace", message=str(error), metadata=metadata
            )
        if not script_path.exists():
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Skill script not found: {args.script}",
                metadata=metadata,
            )
        if not script_path.is_file():
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Skill script is not a file: {args.script}",
                metadata=metadata,
            )
        runtime = _runtime_for_script(script_path)
        if runtime is None:
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Unsupported Skill script type: {script_path.suffix or '<none>'}",
                metadata=metadata,
            )
        executable = runtime[0]
        if Path(executable).is_absolute():
            runtime_available = Path(executable).is_file()
        else:
            runtime_available = shutil.which(executable) is not None
        if not runtime_available:
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Required Skill script runtime is unavailable: {executable}",
                metadata=metadata,
            )
        decision = permission_checker.check(request, self.get_permission_profile())
        decision = with_permission_metadata(decision, metadata)
        if decision.status == "ask":
            decision = PermissionDecision.ask(
                reason=decision.reason,
                message=(
                    f'Skill "{skill.name}" 请求执行 '
                    f"{_logical_script_path(args.script)}。"
                ),
                metadata=decision.metadata,
            )
        return request, decision

    def run_authorized(
        self,
        args: RunSkillScriptArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            skill = self.registry.require(args.skill)
            if not self.state.is_active(skill.name):
                raise ValueError(f"Skill is not active: {skill.name}")
            script_path = self.registry.resolve_script(skill, args.script)
            if not script_path.exists() or not script_path.is_file():
                raise ValueError(f"Skill script is unavailable: {args.script}")
            runtime = _runtime_for_script(script_path)
            if runtime is None:
                raise ValueError(
                    f"Unsupported Skill script type: {script_path.suffix or '<none>'}"
                )
            command = [*runtime, str(script_path), *args.args]
            return execute_command(
                args=CommandExecutionArgs(
                    command=command,
                    timeout_seconds=args.timeout_seconds,
                    max_output_chars=args.max_output_chars,
                ),
                cwd=self.workspace.root,
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

    def _run(self, args: RunSkillScriptArgs) -> ToolResult:
        return ToolResult.failure(
            error="run_skill_script must be run through ToolRegistry.run_tool().",
            metadata={
                "skill_name": args.skill,
                "script": _logical_script_path(args.script),
                "reason": "permission_required",
            },
        )


def _runtime_for_script(script: Path) -> list[str] | None:
    suffix = script.suffix.lower()
    if suffix == ".py":
        return [sys.executable]
    if suffix in {".js", ".mjs"}:
        return ["node"]
    if suffix == ".sh":
        return ["bash"]
    if suffix == ".ps1":
        return ["pwsh", "-File"]
    return None


def _logical_script_path(script: str) -> str:
    normalized = script.replace("\\", "/")
    return normalized if normalized.startswith("scripts/") else f"scripts/{normalized}"
