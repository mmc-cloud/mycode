from pydantic import field_validator

from mycode.memory import (
    MemoryKind,
    MemoryScope,
    MemoryStore,
    SensitiveMemoryError,
    validate_memory_content,
    validate_memory_key,
)
from mycode.permissions import PermissionChecker, PermissionDecision, PermissionRequest
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult
from mycode.tools.permission_metadata import with_permission_metadata


class ListMemoriesArgs(ToolArgs):
    scope: MemoryScope | None = None


class SaveMemoryArgs(ToolArgs):
    scope: MemoryScope
    kind: MemoryKind
    key: str
    content: str

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_memory_key(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        try:
            return validate_memory_content(value, max_chars=2000)
        except SensitiveMemoryError as error:
            raise ValueError(str(error)) from error


class DeleteMemoryArgs(ToolArgs):
    scope: MemoryScope
    key: str

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_memory_key(value)


class ListMemoriesTool(PydanticTool[ListMemoriesArgs]):
    name = "list_memories"
    description = "List safe long-term memories for the current user or project."
    args_model = ListMemoriesArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def _run(self, args: ListMemoriesArgs) -> ToolResult:
        entries = self.store.list_entries(args.scope)
        issues = self.store.list_issues(args.scope)
        lines: list[str] = []
        for entry in entries:
            lines.extend(
                [
                    f"[{entry.scope}/{entry.kind}] {entry.key}",
                    entry.content,
                    "",
                ]
            )
        if issues:
            lines.append("WITHHELD_OR_INVALID")
            lines.extend(issue.display for issue in issues)
        content = "\n".join(lines).rstrip()
        if content == "":
            content = "No long-term memories found."
        return ToolResult.success(
            content=content,
            metadata={
                "scope": args.scope or "all",
                "memory_count": len(entries),
                "issue_count": len(issues),
            },
        )


class SaveMemoryTool(PydanticTool[SaveMemoryArgs]):
    name = "save_memory"
    description = (
        "Create or correct one user-level or project-level long-term memory. "
        "Only use when the user explicitly asks to remember the information."
    )
    args_model = SaveMemoryArgs
    capability = "write"
    risk = "medium"

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build_permission_request(self, args: SaveMemoryArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=f"save {args.scope} memory",
            target=str(self.store.path_for_scope(args.scope)),
            arguments={
                "scope": args.scope,
                "kind": args.kind,
                "key": args.key,
                "content_chars": len(args.content),
            },
            description="Save one reviewed long-term memory.",
        )

    def check_permission(
        self,
        args: SaveMemoryArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        decision = permission_checker.check(request, self.get_permission_profile())
        return request, with_permission_metadata(
            decision,
            {
                "memory_scope": args.scope,
                "memory_kind": args.kind,
                "memory_key": args.key,
                "memory_content": args.content,
                "memory_path": str(self.store.path_for_scope(args.scope)),
            },
        )

    def run_authorized(
        self,
        args: SaveMemoryArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            result = self.store.save(
                scope=args.scope,
                kind=args.kind,
                key=args.key,
                content=args.content,
            )
        except Exception as error:
            return ToolResult.failure(
                error=f"Memory write failed: {error}",
                metadata={
                    **_safe_permission_metadata(decision),
                    "memory_scope": args.scope,
                    "memory_kind": args.kind,
                    "memory_key": args.key,
                    "exception_type": type(error).__name__,
                },
            )
        return ToolResult.success(
            content=(
                f"Memory {result.action}: "
                f"[{args.scope}/{args.kind}] {args.key}"
            ),
            metadata={
                **_safe_permission_metadata(decision),
                "memory_scope": args.scope,
                "memory_kind": args.kind,
                "memory_key": args.key,
                "memory_action": result.action,
                "memory_path": str(result.path),
            },
        )

    def _run(self, args: SaveMemoryArgs) -> ToolResult:
        return ToolResult.failure(
            error="save_memory must be run through ToolRegistry.run_tool().",
            metadata={"memory_key": args.key, "reason": "permission_required"},
        )


class DeleteMemoryTool(PydanticTool[DeleteMemoryArgs]):
    name = "delete_memory"
    description = "Delete one user-level or project-level long-term memory by key."
    args_model = DeleteMemoryArgs
    capability = "write"
    risk = "medium"

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build_permission_request(self, args: DeleteMemoryArgs) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=f"delete {args.scope} memory",
            target=str(self.store.path_for_scope(args.scope)),
            arguments={"scope": args.scope, "key": args.key},
            description="Delete one long-term memory.",
        )

    def check_permission(
        self,
        args: DeleteMemoryArgs,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        decision = permission_checker.check(request, self.get_permission_profile())
        return request, with_permission_metadata(
            decision,
            {
                "memory_scope": args.scope,
                "memory_key": args.key,
                "memory_path": str(self.store.path_for_scope(args.scope)),
            },
        )

    def run_authorized(
        self,
        args: DeleteMemoryArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        try:
            deleted = self.store.delete(scope=args.scope, key=args.key)
        except Exception as error:
            return ToolResult.failure(
                error=f"Memory delete failed: {error}",
                metadata={
                    **_safe_permission_metadata(decision),
                    "memory_scope": args.scope,
                    "memory_key": args.key,
                    "exception_type": type(error).__name__,
                },
            )
        if not deleted:
            return ToolResult.failure(
                error=f"Memory not found: [{args.scope}] {args.key}",
                metadata={
                    **_safe_permission_metadata(decision),
                    "memory_scope": args.scope,
                    "memory_key": args.key,
                },
            )
        return ToolResult.success(
            content=f"Memory deleted: [{args.scope}] {args.key}",
            metadata={
                **_safe_permission_metadata(decision),
                "memory_scope": args.scope,
                "memory_key": args.key,
                "memory_action": "deleted",
            },
        )

    def _run(self, args: DeleteMemoryArgs) -> ToolResult:
        return ToolResult.failure(
            error="delete_memory must be run through ToolRegistry.run_tool().",
            metadata={"memory_key": args.key, "reason": "permission_required"},
        )


def _safe_permission_metadata(decision: PermissionDecision) -> dict[str, object]:
    return {
        key: value
        for key, value in decision.metadata.items()
        if key != "memory_content"
    }
