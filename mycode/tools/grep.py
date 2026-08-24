from pathlib import Path

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.bounds import clamp_positive_int_upper_bound
from mycode.tools.ignore import is_ignored_path, is_low_relevance_path
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.patterns import (
    is_explicit_path_pattern,
    is_sensitive_path_pattern,
    validate_relative_pattern,
)
from mycode.tools.text import contains_nul_byte, decode_text
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_PATH_PATTERN = "**/*"
DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_LIMIT = 1000
MAX_MATCH_LINE_CHARS = 240


class GrepArgs(ToolArgs):
    query: str = Field(min_length=1)
    path_pattern: str = Field(default=DEFAULT_PATH_PATTERN, min_length=1)
    case_sensitive: bool = False
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


class GrepTool(BaseTool[GrepArgs]):
    name = "grep"
    description = "Search text inside workspace files."
    args_model = GrepArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: GrepArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=args.path_pattern,
            arguments=args.model_dump(),
        )

    def check_permission(
        self,
        args: GrepArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        pattern_error = validate_relative_pattern(
            args.path_pattern,
            label="Path pattern",
        )
        if pattern_error is not None:
            return request, PermissionDecision.deny(
                reason="outside_workspace",
                message=pattern_error,
                metadata={"path_pattern": args.path_pattern},
            )

        if is_explicit_path_pattern(args.path_pattern):
            path_decision = PathPermissionPolicy(self.workspace).check_path(
                request,
                args.path_pattern,
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

        if is_sensitive_path_pattern(args.path_pattern):
            return request, PermissionDecision.ask(
                reason="sensitive_path",
                message=(
                    "Sensitive path pattern requires confirmation: "
                    f"{args.path_pattern}"
                ),
                metadata={
                    "path_pattern": args.path_pattern,
                    "pattern_scope": "sensitive_pattern",
                },
            )

        return request, permission_checker.check(
            request,
            self.get_permission_profile(),
        )

    def run_authorized(
        self,
        args: GrepArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        return self._run_with_options(
            args,
            include_ignored=_can_include_ignored(decision),
            include_low_relevance=is_explicit_path_pattern(args.path_pattern),
        )

    def _run(self, args: GrepArgs) -> ToolResult:
        return self._run_with_options(
            args,
            include_ignored=False,
            include_low_relevance=is_explicit_path_pattern(args.path_pattern),
        )

    def _run_with_options(
        self,
        args: GrepArgs,
        *,
        include_ignored: bool,
        include_low_relevance: bool,
    ) -> ToolResult:
        pattern_error = validate_relative_pattern(
            args.path_pattern,
            label="Path pattern",
        )
        if pattern_error is not None:
            return ToolResult.failure(
                error=pattern_error,
                metadata={"path_pattern": args.path_pattern},
            )

        try:
            search_result = _find_search_files(
                self.workspace,
                args.path_pattern,
                include_ignored=include_ignored,
                include_low_relevance=include_low_relevance,
            )
        except ValueError as error:
            return ToolResult.failure(
                error=f"Invalid path pattern: {args.path_pattern}",
                metadata={"path_pattern": args.path_pattern, "reason": str(error)},
            )

        files = search_result.files
        matches: list[str] = []
        searched_files = 0
        skipped_files = 0
        truncated = False

        for path in files:
            raw_content = _read_bytes(path)
            if raw_content is None or contains_nul_byte(raw_content):
                skipped_files += 1
                continue

            decoded = decode_text(raw_content)
            if decoded is None:
                skipped_files += 1
                continue

            text, _encoding = decoded
            searched_files += 1
            relative_path = path.relative_to(self.workspace.root).as_posix()

            for line_number, line in enumerate(text.splitlines(), start=1):
                if not _line_matches(line, args.query, args.case_sensitive):
                    continue

                if len(matches) >= args.max_results:
                    truncated = True
                    break

                matches.append(_format_match(relative_path, line_number, line))

            if truncated:
                break

        content = "\n".join(matches)
        if not content:
            content = "No matches found."

        return ToolResult.success(
            content=content,
            metadata={
                "query": args.query,
                "path_pattern": args.path_pattern,
                "case_sensitive": args.case_sensitive,
                "result_count": len(matches),
                "max_results": args.max_results,
                "truncated": truncated,
                "searched_files": searched_files,
                "skipped_files": skipped_files,
                "filtered_count": search_result.filtered_count,
                "filtered_reasons": search_result.filtered_reasons,
            },
        )


class GrepSearchResult:
    def __init__(
        self,
        files: list[Path],
        filtered_reasons: dict[str, int],
    ) -> None:
        self.files = files
        self.filtered_reasons = filtered_reasons

    @property
    def filtered_count(self) -> int:
        return sum(self.filtered_reasons.values())


def _find_search_files(
    workspace: Workspace,
    path_pattern: str,
    *,
    include_ignored: bool = False,
    include_low_relevance: bool = False,
) -> GrepSearchResult:
    files: list[Path] = []
    filtered_reasons: dict[str, int] = {}

    for candidate in workspace.root.glob(path_pattern):
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

        files.append(candidate)

    return GrepSearchResult(
        files=sorted(files),
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


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _line_matches(line: str, query: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return query in line

    return query.casefold() in line.casefold()


def _format_match(path: str, line_number: int, line: str) -> str:
    return f"{path}:{line_number} | {_truncate_line(line)}"


def _truncate_line(line: str) -> str:
    if len(line) <= MAX_MATCH_LINE_CHARS:
        return line

    return f"{line[: MAX_MATCH_LINE_CHARS - 3]}..."
