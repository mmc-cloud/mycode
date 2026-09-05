import asyncio
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from mycode.permissions import (
    PermissionChecker,
    PermissionDecision,
    PermissionRequest,
    ToolCapability,
    ToolPermissionProfile,
    ToolRisk,
)


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


ArgsT = TypeVar("ArgsT")
PydanticArgsT = TypeVar("PydanticArgsT", bound=ToolArgs)


class ToolPermissionProfileError(ValueError):
    pass


class ToolArgumentValidationError(ValueError):
    def __init__(self, errors: list[dict[str, object]]) -> None:
        super().__init__("Invalid tool arguments")
        self.errors = errors


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            content=content,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def failure(
        cls,
        error: str,
        metadata: dict[str, object] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            error=error,
            metadata={} if metadata is None else dict(metadata),
        )


class BaseTool(ABC, Generic[ArgsT]):
    name: ClassVar[str]
    description: ClassVar[str]
    capability: ClassVar[ToolCapability]
    risk: ClassVar[ToolRisk]
    # Tools must opt in explicitly. The scheduler also checks capability and
    # risk, so a write/command/control tool cannot become concurrent by mistake.
    concurrency_safe: ClassVar[bool] = False

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, object]:
        """Return the JSON Schema accepted by this tool."""
        raise NotImplementedError

    def get_schema(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.input_schema),
        }

    def get_permission_profile(self) -> ToolPermissionProfile:
        capability = getattr(self, "capability", None)
        if capability is None:
            raise ToolPermissionProfileError(
                f"Tool must declare capability: {self.name}"
            )

        risk = getattr(self, "risk", None)
        if risk is None:
            raise ToolPermissionProfileError(
                f"Tool must declare risk: {self.name}"
            )

        return ToolPermissionProfile(
            capability=capability,
            risk=risk,
        )

    def build_permission_request(self, args: ArgsT) -> PermissionRequest:
        return PermissionRequest(
            tool_name=self.name,
            capability=self.capability,
            action=self.name,
            arguments=self.arguments_to_dict(args),
        )

    def check_permission(
        self,
        args: ArgsT,
        permission_checker: PermissionChecker,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        request = self.build_permission_request(args)
        decision = permission_checker.check(
            request,
            self.get_permission_profile(),
        )

        return request, decision

    @abstractmethod
    def parse_arguments(self, arguments: dict[str, object]) -> ArgsT:
        """Validate raw arguments and return the tool's execution value."""
        raise NotImplementedError

    @abstractmethod
    def arguments_to_dict(self, args: ArgsT) -> dict[str, object]:
        """Serialize validated arguments for the permission request."""
        raise NotImplementedError

    @abstractmethod
    async def run_authorized_async(
        self,
        args: ArgsT,
        decision: PermissionDecision,
    ) -> ToolResult:
        """Execute validated, authorized arguments through the common path."""
        raise NotImplementedError


class SyncTool(BaseTool[ArgsT], Generic[ArgsT]):
    """Adapt a synchronous tool implementation to the common async contract."""

    def run(self, arguments: dict[str, object]) -> ToolResult:
        try:
            args = self.parse_arguments(arguments)
        except ToolArgumentValidationError as error:
            return ToolResult.failure(
                error="Invalid tool arguments",
                metadata={"validation_errors": error.errors},
            )

        return self.run_parsed(args)

    def run_parsed(self, args: ArgsT) -> ToolResult:
        try:
            return self._run(args)
        except Exception as error:  # noqa: BLE001 - normalize tool boundary failures
            return ToolResult.failure(
                error=f"Tool execution failed: {error}",
                metadata={"exception_type": type(error).__name__},
            )

    def run_authorized(
        self,
        args: ArgsT,
        decision: PermissionDecision,
    ) -> ToolResult:
        return self.run_parsed(args)

    async def run_authorized_async(
        self,
        args: ArgsT,
        decision: PermissionDecision,
    ) -> ToolResult:
        return await asyncio.to_thread(self.run_authorized, args, decision)

    @abstractmethod
    def _run(self, args: ArgsT) -> ToolResult:
        pass


class PydanticTool(SyncTool[PydanticArgsT], Generic[PydanticArgsT]):
    """Base class for tools whose arguments are defined by a Pydantic model."""

    args_model: ClassVar[type[PydanticArgsT]]

    @property
    def input_schema(self) -> dict[str, object]:
        return _remove_schema_titles(self.args_model.model_json_schema())

    def parse_arguments(self, arguments: dict[str, object]) -> PydanticArgsT:
        try:
            return self.args_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolArgumentValidationError(error.errors()) from error

    def arguments_to_dict(self, args: PydanticArgsT) -> dict[str, object]:
        return args.model_dump()


def _remove_schema_titles(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _remove_schema_titles(item)
            for key, item in value.items()
            if key != "title"
        }

    if isinstance(value, list):
        return [_remove_schema_titles(item) for item in value]

    return value
