from pathlib import Path

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult
from mycode.tools.bounds import clamp_positive_int_upper_bound
from mycode.tools.ignore import is_ignored_path, is_low_relevance_path
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.patterns import (
    is_explicit_path_pattern,
    is_sensitive_path_pattern,
    validate_relative_pattern,
)
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_LIMIT = 1000


class GlobArgs(ToolArgs):
    pattern: str = Field(
        min_length=1,
        description=(
            "用于匹配 workspace 内文件路径的相对 glob pattern，例如 **/*.py。"
        ),
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_RESULTS_LIMIT,
        strict=True,
    )

    @field_validator("max_results", mode="before")
    @classmethod
    def clamp_max_results(cls, value: object) -> object:
        return clamp_positive_int_upper_bound(
            value,
            upper_bound=MAX_RESULTS_LIMIT,
        )


class GlobTool(PydanticTool[GlobArgs]):
    name = "glob"
    description = "按 glob pattern 查找 workspace 内文件。"
    args_model = GlobArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: GlobArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=args.pattern,
            arguments=args.model_dump(),
        )

    def check_permission(
        self,
        args: GlobArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        pattern_error = validate_relative_pattern(
            args.pattern,
            label="Glob pattern",
        )
        if pattern_error is not None:
            return request, PermissionDecision.deny(
                reason="outside_workspace",
                message=pattern_error,
                metadata={"pattern": args.pattern},
            )

        if is_explicit_path_pattern(args.pattern):
            path_decision = PathPermissionPolicy(self.workspace).check_path(
                request,
                args.pattern,
            )
            if path_decision.status != "allow":
                return request, path_decision

            decision = permission_checker.check(
                request,
                self.get_permission_profile(),
            )
            if decision.status != "allow":
                return request, decision

            return request, PermissionDecision.allow(
                message=decision.message,
                metadata={**decision.metadata, **path_decision.metadata},
            )

        if is_sensitive_path_pattern(args.pattern):
            return request, PermissionDecision.ask(
                reason="sensitive_path",
                message=f"Sensitive path pattern requires confirmation: {args.pattern}",
                metadata={
                    "pattern": args.pattern,
                    "pattern_scope": "sensitive_pattern",
                },
            )

        return request, permission_checker.check(
            request,
            self.get_permission_profile(),
        )

    def run_authorized(
        self,
        args: GlobArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        return self._run_with_options(
            args,
            include_ignored=_can_include_ignored(decision),
            include_low_relevance=is_explicit_path_pattern(args.pattern),
        )

    def _run(self, args: GlobArgs) -> ToolResult:
        return self._run_with_options(
            args,
            include_ignored=False,
            include_low_relevance=is_explicit_path_pattern(args.pattern),
        )

    def _run_with_options(
        self,
        args: GlobArgs,
        *,
        include_ignored: bool,
        include_low_relevance: bool,
    ) -> ToolResult:
        pattern_error = validate_relative_pattern(
            args.pattern,
            label="Glob pattern",
        )
        if pattern_error is not None:
            return ToolResult.failure(
                error=pattern_error,
                metadata={"pattern": args.pattern},
            )

        try:
            result = _find_file_matches(
                self.workspace,
                args.pattern,
                include_ignored=include_ignored,
                include_low_relevance=include_low_relevance,
            )
        except ValueError as error:
            return ToolResult.failure(
                error=f"Invalid glob pattern: {args.pattern}",
                metadata={"pattern": args.pattern, "reason": str(error)},
            )

        matches = result.matches
        selected_matches = matches[: args.max_results]
        content = "\n".join(selected_matches)
        if not content:
            content = "No files matched."
            if result.filtered_count > 0:
                content += " Some matches were filtered by workspace relevance or safety rules."

        return ToolResult.success(
            content=content,
            metadata={
                "pattern": args.pattern,
                "result_count": len(selected_matches),
                "total_matches": len(matches),
                "filtered_count": result.filtered_count,
                "filtered_reasons": result.filtered_reasons,
                "max_results": args.max_results,
                "truncated": len(matches) > args.max_results,
            },
        )


class GlobMatchResult:
    def __init__(
        self,
        matches: list[str],
        filtered_reasons: dict[str, int],
    ) -> None:
        self.matches = matches
        self.filtered_reasons = filtered_reasons

    @property
    def filtered_count(self) -> int:
        return sum(self.filtered_reasons.values())


def _find_file_matches(
    workspace: Workspace,
    pattern: str,
    *,
    include_ignored: bool = False,
    include_low_relevance: bool = False,
) -> GlobMatchResult:
    matches: list[str] = []
    filtered_reasons: dict[str, int] = {}

    for candidate in workspace.root.glob(pattern):
        try:
            workspace.resolve_path(candidate)
            if not candidate.is_file():
                continue
            if not include_ignored and is_ignored_path(candidate, workspace.root):
                _increment_reason(filtered_reasons, "ignored")
                continue
            if (
                not include_low_relevance
                and is_low_relevance_path(candidate, workspace.root)
            ):
                _increment_reason(filtered_reasons, "low_relevance")
                continue
        except (OSError, WorkspacePathError):
            continue

        matches.append(candidate.relative_to(workspace.root).as_posix())

    return GlobMatchResult(
        matches=sorted(matches),
        filtered_reasons=filtered_reasons,
    )


def _increment_reason(reasons: dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1


def _can_include_ignored(decision: PermissionDecision) -> bool:
    return (
        decision.metadata.get("confirmation_status") == "approved"
        and (
            decision.metadata.get("path_scope") in {"ignored_path", "sensitive_path"}
            or decision.metadata.get("pattern_scope") == "sensitive_pattern"
        )
    )
