from pathlib import Path

from pydantic import Field

from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult
from mycode.tools.file_mutation import (
    display_path,
    failure_metadata,
    resolved_path_from_decision,
    with_permission_metadata,
)
from mycode.tools.path_permissions import PathPermissionPolicy
from mycode.tools.workspace import Workspace


class WriteFileArgs(ToolArgs):
    path: str = Field(
        min_length=1,
        description="目标文件路径，必须位于 workspace 内，父目录需要已存在。",
    )
    content: str = Field(
        description=(
            "要写入文件的完整 UTF-8 文本内容；目标文件已存在时会覆盖原内容。"
        ),
    )


class WriteFileTool(PydanticTool[WriteFileArgs]):
    name = "write_file"
    description = "Write complete UTF-8 text content to a file."
    args_model = WriteFileArgs
    capability = "write"
    risk = "medium"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def build_permission_request(self, args: WriteFileArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            target=args.path,
            arguments=args.model_dump(),
            description="Write complete file content.",
        )

    def check_permission(
        self,
        args: WriteFileArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        path_decision = PathPermissionPolicy(self.workspace).check_path(
            request,
            args.path,
        )
        metadata = _write_metadata(
            path=resolved_path_from_decision(path_decision),
            content=args.content,
            path_decision=path_decision,
        )

        if metadata["target_exists"] and not metadata["target_is_file"]:
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
        args: WriteFileArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            return self._write_path(
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

    def _run(self, args: WriteFileArgs) -> ToolResult:
        return ToolResult.failure(
            error="write_file must be run through ToolRegistry.run_tool().",
            metadata={"path": args.path, "reason": "permission_required"},
        )

    def _write_path(
        self,
        args: WriteFileArgs,
        path: Path,
        decision: PermissionDecision,
    ) -> ToolResult:
        if path.exists() and not path.is_file():
            return ToolResult.failure(
                error=f"Path is not a file: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {"path": args.path, "target_is_file": False},
                ),
            )

        if not path.parent.exists():
            return ToolResult.failure(
                error=f"Parent directory does not exist: {args.path}",
                metadata=failure_metadata(
                    decision,
                    {"path": args.path, "parent": path.parent.as_posix()},
                ),
            )

        existed_before = path.exists()
        path.write_text(args.content, encoding="utf-8")

        operation = "overwrite" if existed_before else "create"
        return ToolResult.success(
            content=f"Wrote file: {display_path(path, self.workspace.root)}",
            metadata={
                **decision.metadata,
                "path": display_path(path, self.workspace.root),
                "operation": operation,
                "encoding": "utf-8",
                "content_chars": len(args.content),
                "content_bytes": len(args.content.encode("utf-8")),
                "target_existed_before": existed_before,
                "permission_status": decision.status,
            },
        )


def _write_metadata(
    *,
    path: Path,
    content: str,
    path_decision: PermissionDecision,
) -> dict[str, object]:
    target_exists = path.exists()
    return {
        **path_decision.metadata,
        "operation": "overwrite" if target_exists else "create",
        "target_exists": target_exists,
        "target_is_file": path.is_file() if target_exists else None,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
    }
