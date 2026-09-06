import copy
import hashlib
import re
from collections.abc import Awaitable, Callable

from jsonschema import Draft202012Validator, SchemaError
from jsonschema.validators import validator_for
from mcp.types import CallToolResult, Tool

from mycode.mcp.errors import classify_mcp_error
from mycode.mcp.result_adapter import adapt_mcp_result
from mycode.permissions import (
    PermissionDecision,
    PermissionRequest,
    ToolPermissionProfile,
)
from mycode.tools.base import BaseTool, ToolArgumentValidationError, ToolResult

MCPCall = Callable[[str, str, dict[str, object]], Awaitable[CallToolResult]]
MAX_REGISTRY_NAME_LENGTH = 64
_INVALID_REGISTRY_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def build_registry_name(server_alias: str, remote_name: str) -> str:
    """Build a bounded model-facing name while preserving remote identity."""
    safe_remote = _INVALID_REGISTRY_NAME_CHARS.sub("-", remote_name).strip("-")
    if not safe_remote:
        safe_remote = "tool"
    base = f"mcp__{server_alias}__{safe_remote}"
    changed = safe_remote != remote_name
    if not changed and len(base) <= MAX_REGISTRY_NAME_LENGTH:
        return base

    digest = hashlib.sha256(
        f"{server_alias}\0{remote_name}".encode()
    ).hexdigest()[:8]
    suffix = f"__{digest}"
    return f"{base[: MAX_REGISTRY_NAME_LENGTH - len(suffix)]}{suffix}"


class MCPToolAdapter(BaseTool[dict[str, object]]):
    concurrency_safe = False

    def __init__(self, server_alias: str, remote_tool: Tool, call: MCPCall) -> None:
        self.server_alias = server_alias
        self.remote_name = remote_tool.name
        self.registry_name = build_registry_name(server_alias, remote_tool.name)
        self.name = self.registry_name
        self.description = remote_tool.description or f"MCP tool {remote_tool.name}."
        self._input_schema = copy.deepcopy(remote_tool.input_schema)
        self.output_schema = copy.deepcopy(remote_tool.output_schema)
        self.annotations = (
            {} if remote_tool.annotations is None
            else remote_tool.annotations.model_dump(by_alias=True, exclude_none=True)
        )
        self.capability, self.risk = _permission_profile(self.annotations)
        self._call = call
        try:
            if "$schema" in self._input_schema:
                validator_class = validator_for(self._input_schema)
            else:
                validator_class = Draft202012Validator
            validator_class.check_schema(self._input_schema)
            self._validator = validator_class(self._input_schema)
        except SchemaError as error:
            raise ValueError(f"Invalid inputSchema for {self.name}") from error

    @property
    def input_schema(self) -> dict[str, object]:
        return copy.deepcopy(self._input_schema)

    def parse_arguments(self, arguments: dict[str, object]) -> dict[str, object]:
        errors = sorted(self._validator.iter_errors(arguments), key=lambda item: list(item.path))
        if errors:
            raise ToolArgumentValidationError(
                [
                    {
                        "location": ".".join(str(part) for part in error.path),
                        "message": error.message,
                        "validator": error.validator,
                    }
                    for error in errors[:10]
                ]
            )
        return dict(arguments)

    def arguments_to_dict(
        self, args: dict[str, object]
    ) -> dict[str, object]:
        return dict(args)

    def get_permission_profile(self) -> ToolPermissionProfile:
        return ToolPermissionProfile(capability=self.capability, risk=self.risk)

    def build_permission_request(
        self, args: dict[str, object]
    ) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.remote_name,
            arguments=args,
            description=f"MCP server: {self.server_alias}",
        )

    async def run_authorized_async(
        self, args: dict[str, object], decision: PermissionDecision
    ) -> ToolResult:
        try:
            result = await self._call(self.server_alias, self.remote_name, args)
        except Exception as error:  # noqa: BLE001 - normalize provider failures
            classified = classify_mcp_error(error)
            return ToolResult.failure(
                f"MCP tool call failed: {classified.summary}",
                {
                    "server_alias": self.server_alias,
                    "tool_name": self.remote_name,
                    "error_type": type(error).__name__,
                    "root_error_type": classified.root_error_type,
                    "error_category": classified.category,
                    "retryable": classified.retryable,
                },
            )
        adapted = adapt_mcp_result(result)
        return ToolResult(
            ok=adapted.ok,
            content=adapted.content,
            error=adapted.error,
            metadata={
                "server_alias": self.server_alias,
                "tool_name": self.remote_name,
                **adapted.metadata,
            },
        )

def _permission_profile(
    annotations: dict[str, object],
) -> tuple[str, str]:
    read_only = annotations.get("readOnlyHint") is True
    destructive = annotations.get("destructiveHint") is True
    if destructive:
        return "write", "high"
    if read_only:
        return "read", "low"
    return "write", "medium"
