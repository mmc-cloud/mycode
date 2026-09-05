from dataclasses import dataclass, field
from typing import Literal, Protocol


PermissionStatus = Literal["allow", "deny", "ask"]
ConfirmationStatus = Literal["approved", "rejected"]
ApprovalScope = Literal["once", "task", "session"]


@dataclass
class ScopedApprovalState:
    """In-memory approvals owned by one registry; never override a deny."""

    task_approved: bool = False
    session_approved: bool = False

    def approve(self, scope: ApprovalScope) -> None:
        if scope == "task":
            self.task_approved = True
        elif scope == "session":
            self.session_approved = True
        elif scope != "once":
            raise ValueError("Invalid approval scope")

    def allows_ask(self) -> ApprovalScope | None:
        if self.session_approved:
            return "session"
        if self.task_approved:
            return "task"
        return None

    def begin_task(self) -> None:
        self.task_approved = False

    def end_task(self) -> None:
        self.task_approved = False

ToolCapability = Literal["read", "write", "command", "control"]
ToolRisk = Literal["low", "medium", "high"]

PermissionReason = Literal[
    "allowed",
    "requires_confirmation",
    "outside_workspace",
    "ignored_path",
    "sensitive_path",
    "unsupported_operation",
    "dangerous_command",
]


@dataclass(frozen=True)
class ToolPermissionProfile:
    capability: ToolCapability
    risk: ToolRisk


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    capability: ToolCapability
    action: str
    target: str | None = None
    arguments: dict[str, object] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class PermissionDecision:
    status: PermissionStatus
    reason: PermissionReason
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def allow(
        cls,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> "PermissionDecision":
        return cls(
            status="allow",
            reason="allowed",
            message=message,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def deny(
        cls,
        reason: PermissionReason,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> "PermissionDecision":
        return cls(
            status="deny",
            reason=reason,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def ask(
        cls,
        reason: PermissionReason = "requires_confirmation",
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> "PermissionDecision":
        return cls(
            status="ask",
            reason=reason,
            message=message,
            metadata={} if metadata is None else metadata,
        )


class PermissionChecker(Protocol):
    def check(
        self,
        request: PermissionRequest,
        profile: ToolPermissionProfile,
    ) -> PermissionDecision:
        pass


class DefaultPermissionChecker:
    def check(
        self,
        request: PermissionRequest,
        profile: ToolPermissionProfile,
    ) -> PermissionDecision:
        metadata = {
            "tool_name": request.tool_name,
            "capability": request.capability,
            "action": request.action,
            "target": request.target,
            "risk": profile.risk,
        }

        if request.capability != profile.capability:
            return PermissionDecision.deny(
                reason="unsupported_operation",
                message=(
                    f"Tool {request.tool_name} cannot perform "
                    f"{request.capability} operations."
                ),
                metadata={
                    **metadata,
                    "tool_capability": profile.capability,
                    "requested_capability": request.capability,
                },
            )

        if request.capability == "read":
            return PermissionDecision.allow(
                message=f"Read operation allowed: {request.action}",
                metadata=metadata,
            )

        if request.capability == "write":
            return PermissionDecision.ask(
                message=f"Write operation requires confirmation: {request.action}",
                metadata=metadata,
            )

        if request.capability == "command":
            return PermissionDecision.ask(
                message=f"Command operation requires confirmation: {request.action}",
                metadata=metadata,
            )

        if request.capability == "control":
            return PermissionDecision.allow(
                message=f"Internal control operation allowed: {request.action}",
                metadata=metadata,
            )

        return PermissionDecision.deny(
            reason="unsupported_operation",
            message=f"Unsupported operation: {request.action}",
            metadata=metadata,
        )


@dataclass(frozen=True)
class ConfirmationRequest:
    permission_request: PermissionRequest
    permission_decision: PermissionDecision
    prompt: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ConfirmationResult:
    status: ConfirmationStatus
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    scope: ApprovalScope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.status == "approved":
            scope = "once" if self.scope is None else self.scope
            if scope not in {"once", "task", "session"}:
                raise ValueError("Invalid approval scope")
            object.__setattr__(self, "scope", scope)
        elif self.scope is not None:
            raise ValueError("Rejected confirmation cannot carry an approval scope")

    @classmethod
    def approved(
        cls,
        message: str = "",
        metadata: dict[str, object] | None = None,
        *,
        scope: ApprovalScope = "once",
    ) -> "ConfirmationResult":
        return cls(
            status="approved",
            scope=scope,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def rejected(
        cls,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> "ConfirmationResult":
        return cls(
            status="rejected",
            message=message,
            metadata={} if metadata is None else metadata,
        )


class Confirmer(Protocol):
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        pass


class RejectingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.rejected(
            message="Confirmation is not available.",
            metadata={
                "tool_name": request.permission_request.tool_name,
                "capability": request.permission_request.capability,
                "action": request.permission_request.action,
                "target": request.permission_request.target,
            },
        )
