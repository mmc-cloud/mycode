from pathlib import Path

from pydantic import Field

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import BaseTool, ToolArgs, ToolResult
from mycode.tools.file_mutation import (
    display_path,
    failure_metadata,
    resolved_path_from_decision,
    with_permission_metadata,
)
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.text import contains_nul_byte, decode_text
from mycode.tools.workspace import Workspace

SNIPPET_CONTEXT_LINES = 3


class _ReplacementMatch:
    def __init__(
        self,
        *,
        exact_match_count: int,
        normalized_match_count: int | None = None,
        start: int | None = None,
        end: int | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        newline_normalized: bool = False,
    ) -> None:
        self.exact_match_count = exact_match_count
        self.normalized_match_count = normalized_match_count
        self.start = start
        self.end = end
        self.old_text = old_text
        self.new_text = new_text
        self.newline_normalized = newline_normalized

    @property
    def match_count(self) -> int:
        if self.start is not None:
            return 1
        if self.exact_match_count > 0:
            return self.exact_match_count
        if self.normalized_match_count is not None:
            return self.normalized_match_count
        return 0


class EditFileArgs(ToolArgs):
    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class EditFileTool(BaseTool[EditFileArgs]):
    name = "edit_file"
    description = "Replace one exact text occurrence in an existing text file."
    args_model = EditFileArgs
    capability = "write"
    risk = "medium"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: EditFileArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=args.path,
            arguments=args.model_dump(),
            description="Replace one exact text occurrence in a file.",
        )

    def check_permission(
        self,
        args: EditFileArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        path_decision = PathPermissionPolicy(self.workspace).check_path(
            request,
            args.path,
        )
        metadata = _edit_metadata(
            path=resolved_path_from_decision(path_decision),
            args=args,
            path_decision=path_decision,
        )

        if not metadata["target_exists"]:
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"File not found: {args.path}",
                metadata=metadata,
            )

        if not metadata["target_is_file"]:
            return request, PermissionDecision.deny(
                reason="unsupported_operation",
                message=f"Path is not a file: {args.path}",
                metadata=metadata,
            )

        if path_decision.status != "allow":
            return request, PermissionDecision.ask(
                reason=path_decision.reason,
                message=path_decision.message,
                metadata=metadata,
            )

        decision = permission_checker.check(
            request,
            self.get_permission_profile(),
        )

        return request, with_permission_metadata(decision, metadata)

    def run_authorized(
        self,
        args: EditFileArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            return self._edit_path(
                args,
                resolved_path_from_decision(decision),
                decision,
            )
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata=failure_metadata(
                    decision,
                    {"exception_type": type(error).__name__},
                ),
            )

    def _run(self, args: EditFileArgs) -> ToolResult:
        return ToolResult.failure(
            error="edit_file must be run through ToolRegistry.run_tool().",
            metadata={"path": args.path, "reason": "permission_required"},
        )

    def _edit_path(
        self,
        args: EditFileArgs,
        path: Path,
        decision: PermissionDecision,
    ) -> ToolResult:
        if not path.exists():
            return ToolResult.failure(
                error=f"File not found: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {
                        "path": args.path,
                        "target_exists": False,
                    },
                ),
            )

        if not path.is_file():
            return ToolResult.failure(
                error=f"Path is not a file: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {
                        "path": args.path,
                        "target_is_file": False,
                    },
                ),
            )

        raw_content = path.read_bytes()
        if contains_nul_byte(raw_content):
            return ToolResult.failure(
                error=f"File is not a supported text file: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {"path": args.path, "reason": "nul_byte"},
                ),
            )

        decoded = decode_text(raw_content)
        if decoded is None:
            return ToolResult.failure(
                error=f"File is not a supported text file: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {"path": args.path, "supported_encodings": ["utf-8", "gbk"]},
                ),
            )

        text, encoding = decoded
        replacement = _find_replacement_match(text, args.old_text, args.new_text)
        if replacement.match_count == 0:
            return ToolResult.failure(
                error="Target text not found.",
                metadata=failure_metadata(
                    decision,
                    _result_metadata(
                        args,
                        path,
                        self.workspace.root,
                        text,
                        encoding,
                        replacement,
                    ),
                ),
            )

        if replacement.match_count > 1:
            return ToolResult.failure(
                error="Target text is not unique.",
                metadata=failure_metadata(
                    decision,
                    _result_metadata(
                        args,
                        path,
                        self.workspace.root,
                        text,
                        encoding,
                        replacement,
                    ),
                ),
            )

        replacement_start = replacement.start
        if (
            replacement_start is None
            or replacement.end is None
            or replacement.new_text is None
        ):
            return ToolResult.failure(
                error="Target text not found.",
                metadata=failure_metadata(
                    decision,
                    _result_metadata(
                        args,
                        path,
                        self.workspace.root,
                        text,
                        encoding,
                        replacement,
                    ),
                ),
            )

        edited_text = (
            text[:replacement_start] + replacement.new_text + text[replacement.end :]
        )
        try:
            path.write_bytes(edited_text.encode(encoding))
        except UnicodeEncodeError:
            return ToolResult.failure(
                error=f"Edited content cannot be encoded as {encoding}.",
                metadata=failure_metadata(
                    decision,
                    _result_metadata(
                        args,
                        path,
                        self.workspace.root,
                        text,
                        encoding,
                        replacement,
                    ),
                ),
            )

        snippet = _build_edit_snippet(
            edited_text=edited_text,
            replacement_start=replacement_start,
            replacement_chars=len(args.new_text),
            context_lines=SNIPPET_CONTEXT_LINES,
        )
        result_metadata = {
            **decision.metadata,
            **_result_metadata(
                args,
                path,
                self.workspace.root,
                text,
                encoding,
                replacement,
            ),
            "edited_chars": len(edited_text),
            "edited_bytes": len(edited_text.encode(encoding)),
            "permission_status": decision.status,
            "line_start": snippet.line_start,
            "line_end": snippet.line_end,
            "snippet_truncated": snippet.truncated,
        }
        if _should_return_snippet(decision):
            result_metadata["snippet"] = snippet.content
            result_content = (
                f"Edited file: {display_path(path, self.workspace.root)}\n\n"
                f"Changed context (lines {snippet.line_start}-{snippet.line_end}):\n"
                f"{snippet.content}"
            )
        else:
            result_metadata["snippet_suppressed"] = True
            result_content = (
                f"Edited file: {display_path(path, self.workspace.root)}\n"
                "Changed context suppressed for sensitive path."
            )

        return ToolResult.success(
            content=result_content,
            metadata=result_metadata,
        )


def _edit_metadata(
    *,
    path: Path,
    args: EditFileArgs,
    path_decision: PermissionDecision,
) -> dict[str, object]:
    target_exists = path.exists()
    return {
        **path_decision.metadata,
        "operation": "replace",
        "target_exists": target_exists,
        "target_is_file": path.is_file() if target_exists else None,
        "old_text_chars": len(args.old_text),
        "new_text_chars": len(args.new_text),
    }


def _result_metadata(
    args: EditFileArgs,
    path: Path,
    root: Path,
    text: str,
    encoding: str,
    replacement: _ReplacementMatch,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "path": display_path(path, root),
        "encoding": encoding,
        "operation": "replace",
        "match_count": replacement.match_count,
        "old_text_chars": len(args.old_text),
        "new_text_chars": len(args.new_text),
        "original_chars": len(text),
        "original_bytes": len(text.encode(encoding)),
    }
    if replacement.normalized_match_count is not None:
        metadata.update(
            {
                "exact_match_count": replacement.exact_match_count,
                "newline_normalized_match_count": replacement.normalized_match_count,
                "newline_normalized_match": replacement.newline_normalized,
            }
        )

    return metadata


def _find_replacement_match(
    text: str,
    old_text: str,
    new_text: str,
) -> _ReplacementMatch:
    exact_match_count = text.count(old_text)
    if exact_match_count == 1:
        start = text.index(old_text)
        return _ReplacementMatch(
            exact_match_count=exact_match_count,
            start=start,
            end=start + len(old_text),
            old_text=old_text,
            new_text=new_text,
        )
    if exact_match_count > 1:
        return _ReplacementMatch(exact_match_count=exact_match_count)

    normalized_text, normalized_to_original = _normalize_newlines_with_mapping(text)
    normalized_old_text = _normalize_newlines(old_text)
    normalized_match_count = normalized_text.count(normalized_old_text)
    if normalized_match_count != 1:
        return _ReplacementMatch(
            exact_match_count=exact_match_count,
            normalized_match_count=normalized_match_count,
        )

    normalized_start = normalized_text.index(normalized_old_text)
    normalized_end = normalized_start + len(normalized_old_text)
    original_start = normalized_to_original[normalized_start]
    original_end = (
        normalized_to_original[normalized_end]
        if normalized_end < len(normalized_to_original)
        else len(text)
    )
    original_old_text = text[original_start:original_end]
    adapted_new_text = _adapt_replacement_newlines(
        new_text,
        original_old_text=original_old_text,
    )
    return _ReplacementMatch(
        exact_match_count=exact_match_count,
        normalized_match_count=normalized_match_count,
        start=original_start,
        end=original_end,
        old_text=original_old_text,
        new_text=adapted_new_text,
        newline_normalized=True,
    )


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_newlines_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    normalized_to_original: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            normalized.append("\n")
            normalized_to_original.append(index)
            if index + 1 < len(text) and text[index + 1] == "\n":
                index += 2
            else:
                index += 1
            continue

        normalized.append(char)
        normalized_to_original.append(index)
        index += 1

    return "".join(normalized), normalized_to_original


def _adapt_replacement_newlines(new_text: str, *, original_old_text: str) -> str:
    newline = _dominant_newline(original_old_text)
    if newline is None:
        return new_text

    return _normalize_newlines(new_text).replace("\n", newline)


def _dominant_newline(text: str) -> str | None:
    counts = {
        "\r\n": text.count("\r\n"),
        "\n": text.count("\n") - text.count("\r\n"),
        "\r": text.count("\r") - text.count("\r\n"),
    }
    newline, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        return None

    return newline


def _should_return_snippet(decision: PermissionDecision) -> bool:
    return decision.metadata.get("path_scope") != "sensitive_path"


class _EditSnippet:
    def __init__(
        self,
        *,
        content: str,
        line_start: int,
        line_end: int,
        truncated: bool,
    ) -> None:
        self.content = content
        self.line_start = line_start
        self.line_end = line_end
        self.truncated = truncated


def _build_edit_snippet(
    *,
    edited_text: str,
    replacement_start: int,
    replacement_chars: int,
    context_lines: int,
) -> _EditSnippet:
    if edited_text == "":
        return _EditSnippet(
            content="",
            line_start=1,
            line_end=1,
            truncated=False,
        )

    lines = edited_text.splitlines()
    if not lines:
        lines = [edited_text]

    first_changed_line = edited_text[:replacement_start].count("\n")
    replacement_end = replacement_start + replacement_chars
    last_changed_line = edited_text[:replacement_end].count("\n")
    last_changed_line = min(last_changed_line, len(lines) - 1)

    snippet_start = max(0, first_changed_line - context_lines)
    snippet_end = min(len(lines) - 1, last_changed_line + context_lines)
    snippet_lines = [
        f"{line_number} | {lines[line_index]}"
        for line_index, line_number in enumerate(
            range(snippet_start + 1, snippet_end + 2),
            start=snippet_start,
        )
    ]

    return _EditSnippet(
        content="\n".join(snippet_lines),
        line_start=snippet_start + 1,
        line_end=snippet_end + 1,
        truncated=snippet_start > 0 or snippet_end < len(lines) - 1,
    )
