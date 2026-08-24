import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import ValidationError

from mycode.permissions import (
    Confirmer,
    ConfirmationRequest,
    DefaultPermissionChecker,
    PermissionChecker,
    PermissionDecision,
    PermissionRequest,
    RejectingConfirmer,
)
from mycode.tools.base import BaseTool, ToolResult


class DuplicateToolError(ValueError):
    pass


class ToolNotFoundError(KeyError):
    pass


@dataclass
class ToolRegistry:
    _tools: dict[str, BaseTool] = field(default_factory=dict)
    permission_checker: PermissionChecker = field(default_factory=DefaultPermissionChecker)
    confirmer: Confirmer = field(default_factory=RejectingConfirmer)

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[BaseTool],
        *,
        permission_checker: PermissionChecker | None = None,
        confirmer: Confirmer | None = None,
    ) -> "ToolRegistry":
        registry = cls()
        if permission_checker is not None:
            registry.permission_checker = permission_checker
        if confirmer is not None:
            registry.confirmer = confirmer

        for tool in tools:
            registry.register(tool)

        return registry

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def require(self, name: str) -> BaseTool:
        tool = self.get(name)

        if tool is None:
            raise ToolNotFoundError(f"Tool not found: {name}")

        return tool

    def run_tool(self, name: str, arguments: dict[str, object]) -> ToolResult:
        tool = self.get(name)

        if tool is None:
            return ToolResult.failure(
                error=f"Tool not found: {name}",
                metadata={"tool_name": name},
            )

        try:
            args = tool.parse_arguments(arguments)
        except ValidationError as error:
            return ToolResult.failure(
                error="Invalid tool arguments",
                metadata={"validation_errors": error.errors()},
            )

        try:
            request, decision = tool.check_permission(args, self.permission_checker)
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool permission check failed: {error}",
                metadata={
                    "tool_name": name,
                    "exception_type": type(error).__name__,
                },
            )

        decision, permission_failure = _resolve_permission(
            request,
            decision,
            self.confirmer,
        )
        if permission_failure is not None:
            return permission_failure

        return tool.run_authorized(args, decision)

    def is_concurrency_safe(self, name: str) -> bool:
        tool = self.get(name)
        if tool is None:
            return False

        return bool(
            getattr(tool, "concurrency_safe", False)
            and getattr(tool, "capability", None) == "read"
            and getattr(tool, "risk", None) == "low"
        )

    async def run_tool_async(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        permission_lock: asyncio.Lock,
    ) -> ToolResult:
        """Run one opt-in read tool without overlapping permission interaction."""
        tool = self.get(name)

        if tool is None:
            return ToolResult.failure(
                error=f"Tool not found: {name}",
                metadata={"tool_name": name},
            )

        try:
            args = tool.parse_arguments(arguments)
        except ValidationError as error:
            return ToolResult.failure(
                error="Invalid tool arguments",
                metadata={"validation_errors": error.errors()},
            )

        async with permission_lock:
            try:
                request, decision = tool.check_permission(
                    args,
                    self.permission_checker,
                )
            except Exception as error:
                return ToolResult.failure(
                    error=f"Tool permission check failed: {error}",
                    metadata={
                        "tool_name": name,
                        "exception_type": type(error).__name__,
                    },
                )

            decision, permission_failure = _resolve_permission(
                request,
                decision,
                self.confirmer,
            )
            if permission_failure is not None:
                return permission_failure

        try:
            return await asyncio.to_thread(tool.run_authorized, args, decision)
        except Exception as error:
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata={"exception_type": type(error).__name__},
            )

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, object]]:
        return [tool.get_schema() for tool in self.list_tools()]


def _permission_failure(decision: PermissionDecision) -> ToolResult:
    return ToolResult.failure(
        error=decision.message or "Permission denied",
        metadata={
            "permission_status": decision.status,
            "permission_reason": decision.reason,
            **decision.metadata,
        },
    )


def _resolve_permission(
    request: PermissionRequest,
    decision: PermissionDecision,
    confirmer: Confirmer,
) -> tuple[PermissionDecision, ToolResult | None]:
    if decision.status == "deny":
        return decision, _permission_failure(decision)

    if decision.status != "ask":
        return decision, None

    confirmation_request = ConfirmationRequest(
        permission_request=request,
        permission_decision=decision,
        prompt=decision.message,
        metadata=decision.metadata,
    )
    confirmation_result = confirmer.confirm(confirmation_request)
    if confirmation_result.status == "rejected":
        return decision, ToolResult.failure(
            error=confirmation_result.message
            or "Permission confirmation rejected",
            metadata={
                "permission_status": decision.status,
                "permission_reason": decision.reason,
                "confirmation_status": confirmation_result.status,
                "confirmation_message": confirmation_result.message,
                **decision.metadata,
                "confirmation_metadata": confirmation_result.metadata,
            },
        )

    return PermissionDecision.allow(
        message=decision.message,
        metadata={
            **decision.metadata,
            "confirmation_status": confirmation_result.status,
            "confirmation_message": confirmation_result.message,
            "confirmation_metadata": confirmation_result.metadata,
        },
    ), None
