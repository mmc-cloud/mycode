from pathlib import Path

from pydantic import Field, field_validator

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.bounds import clamp_positive_int_upper_bound
from mycode.tools.ignore import is_sensitive_path
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.text import (
    SUPPORTED_TEXT_ENCODINGS,
    contains_nul_byte,
    decode_text,
)
from mycode.tools.workspace import Workspace, WorkspacePathError


SUPPORTED_ENCODINGS = SUPPORTED_TEXT_ENCODINGS
DEFAULT_MAX_LINES = 200
MAX_LINES_LIMIT = 1000


class ReadFileArgs(ToolArgs):
    path: str
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(
        default=DEFAULT_MAX_LINES,
        ge=1,
        le=MAX_LINES_LIMIT,
        strict=True,
    )

    @field_validator("max_lines", mode="before")
    @classmethod
    def clamp_max_lines(cls, value: object) -> object:
        return clamp_positive_int_upper_bound(
            value,
            upper_bound=MAX_LINES_LIMIT,
        )


class ReadFileTool(BaseTool[ReadFileArgs]):
    name = "read_file"
    description = "Read a text file inside the workspace with line numbers."
    args_model = ReadFileArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: ReadFileArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=args.path,
            arguments=args.model_dump(),
        )

    def check_permission(
        self,
        args: ReadFileArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        path_decision = PathPermissionPolicy(self.workspace).check_path(
            request,
            args.path,
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

    def run_authorized(
        self,
        args: ReadFileArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            resolved_path = decision.metadata.get("resolved_path")
            if isinstance(resolved_path, str):
                return self._read_path(args, Path(resolved_path))

            return self.run_parsed(args)
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata={"exception_type": type(error).__name__},
            )

    def _run(self, args: ReadFileArgs) -> ToolResult:
        try:
            path = self.workspace.resolve_path(args.path)
        except WorkspacePathError as error:
            return ToolResult.failure(
                error=str(error),
                metadata={"path": args.path},
            )

        if not path.exists():
            return ToolResult.failure(
                error=f"File not found: {args.path}",
                metadata={"path": args.path},
            )

        if not path.is_file():
            return ToolResult.failure(
                error=f"Path is not a file: {args.path}",
                metadata={"path": args.path},
            )

        if is_sensitive_path(path, self.workspace.root):
            return ToolResult.failure(
                error=f"Refusing to read sensitive file: {args.path}",
                metadata={"path": args.path, "reason": "sensitive_file"},
            )

        return self._read_path(args, path)

    def _read_path(self, args: ReadFileArgs, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult.failure(
                error=f"File not found: {args.path}",
                metadata={"path": args.path},
            )

        if not path.is_file():
            return ToolResult.failure(
                error=f"Path is not a file: {args.path}",
                metadata={"path": args.path},
            )

        raw_content = path.read_bytes()
        if contains_nul_byte(raw_content):
            return ToolResult.failure(
                error=f"File is not a supported text file: {args.path}",
                metadata={"path": args.path, "reason": "nul_byte"},
            )

        decoded = decode_text(raw_content)
        if decoded is None:
            return ToolResult.failure(
                error=f"File is not a supported text file: {args.path}",
                metadata={
                    "path": args.path,
                    "supported_encodings": list(SUPPORTED_ENCODINGS),
                },
            )

        text, encoding = decoded
        lines = text.splitlines()
        total_lines = len(lines)
        start_index = args.start_line - 1
        end_index = min(start_index + args.max_lines, total_lines)
        selected_lines = lines[start_index:end_index]
        end_line = args.start_line + len(selected_lines) - 1
        if not selected_lines:
            end_line = args.start_line - 1
        has_more = end_index < total_lines
        next_start_line = end_line + 1 if has_more else None
        display_path = _display_path(path, self.workspace.root)
        content = _format_read_result(
            path=display_path,
            selected_lines=selected_lines,
            start_line=args.start_line,
            end_line=end_line,
            total_lines=total_lines,
            has_more=has_more,
            next_start_line=next_start_line,
        )

        return ToolResult.success(
            content=content,
            metadata={
                "path": display_path,
                "encoding": encoding,
                "start_line": args.start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "has_more": has_more,
                "next_start_line": next_start_line,
            },
        )


def _format_numbered_lines(lines: list[str], *, start_line: int) -> str:
    return "\n".join(
        f"{line_number} | {line}"
        for line_number, line in enumerate(lines, start=start_line)
    )


def _format_read_result(
    *,
    path: str,
    selected_lines: list[str],
    start_line: int,
    end_line: int,
    total_lines: int,
    has_more: bool,
    next_start_line: int | None,
) -> str:
    lines_display = (
        f"{start_line}-{end_line}"
        if end_line >= start_line
        else f"none (requested start: {start_line})"
    )
    header = [
        f"File: {path}",
        f"Lines: {lines_display} / {total_lines}",
        f"Has more: {'yes' if has_more else 'no'}",
    ]
    if next_start_line is not None:
        header.append(f"Next start line: {next_start_line}")
    body = _format_numbered_lines(selected_lines, start_line=start_line)
    header_text = "\n".join(header)
    return header_text if not body else f"{header_text}\n\n{body}"


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
