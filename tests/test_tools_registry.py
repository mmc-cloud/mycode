import asyncio

import pytest

from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    PermissionDecision,
    PermissionRequest,
    ToolPermissionProfile,
)
from mycode.tools import (
    BaseTool,
    DuplicateToolError,
    PydanticTool,
    ToolArgs,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)


def test_register_and_get_tool_by_name() -> None:
    registry = ToolRegistry()
    tool = FakeTool(name="fake")

    registry.register(tool)

    assert registry.get("fake") is tool


def test_get_returns_none_for_missing_tool() -> None:
    registry = ToolRegistry()

    assert registry.get("missing") is None


def test_require_returns_tool_by_name() -> None:
    registry = ToolRegistry.from_tools([FakeTool(name="fake")])

    assert registry.require("fake").name == "fake"


def test_require_raises_for_missing_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="Tool not found: missing"):
        registry.require("missing")


def test_register_raises_for_duplicate_tool_name() -> None:
    registry = ToolRegistry.from_tools([FakeTool(name="fake")])

    with pytest.raises(DuplicateToolError, match="Tool already registered: fake"):
        registry.register(FakeTool(name="fake"))


def test_list_tools_returns_registered_tools_in_order() -> None:
    first = FakeTool(name="first")
    second = FakeTool(name="second")
    registry = ToolRegistry.from_tools([first, second])

    assert registry.list_tools() == [first, second]


def test_get_schemas_returns_registered_tool_schemas_in_order() -> None:
    registry = ToolRegistry.from_tools(
        [
            FakeTool(name="first"),
            FakeTool(name="second"),
        ]
    )

    assert registry.get_schemas() == [
        {
            "name": "first",
            "description": "Fake tool first.",
            "parameters": {
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "type": "object",
            },
        },
        {
            "name": "second",
            "description": "Fake tool second.",
            "parameters": {
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "type": "object",
            },
        },
    ]


def test_run_tool_executes_registered_tool() -> None:
    checker = RecordingPermissionChecker(PermissionDecision.allow())
    registry = ToolRegistry.from_tools(
        [FakeTool(name="fake")],
        permission_checker=checker,
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result == ToolResult.success("hello")
    assert checker.requests == [
        PermissionRequest(
            tool_name="fake",
            capability="read",
            action="fake",
            arguments={"text": "hello"},
        )
    ]
    assert checker.profiles == [ToolPermissionProfile(capability="read", risk="low")]


def test_run_tool_returns_failure_for_missing_tool() -> None:
    registry = ToolRegistry()

    result = registry.run_tool("missing", {"text": "hello"})

    assert result == ToolResult.failure(
        error="Tool not found: missing",
        metadata={"tool_name": "missing"},
    )


def test_run_tool_rejects_async_only_tool_before_permission() -> None:
    checker = RecordingPermissionChecker(PermissionDecision.allow())
    registry = ToolRegistry.from_tools(
        [AsyncOnlyTool()], permission_checker=checker
    )

    result = registry.run_tool("async_only", {"text": "hello"})

    assert result == ToolResult.failure(
        error="Tool requires async execution path",
        metadata={
            "tool_name": "async_only",
            "reason": "async_execution_required",
        },
    )
    assert checker.requests == []


def test_run_tool_async_executes_sync_and_native_async_tools() -> None:
    registry = ToolRegistry.from_tools(
        [FakeTool(name="sync"), AsyncOnlyTool()],
        permission_checker=RecordingPermissionChecker(
            PermissionDecision.allow()
        ),
    )

    async def run_tools() -> tuple[ToolResult, ToolResult]:
        permission_lock = asyncio.Lock()
        sync_result = await registry.run_tool_async(
            "sync", {"text": "one"}, permission_lock=permission_lock
        )
        async_result = await registry.run_tool_async(
            "async_only",
            {"text": "two"},
            permission_lock=permission_lock,
        )
        return sync_result, async_result

    assert asyncio.run(run_tools()) == (
        ToolResult.success("one"),
        ToolResult.success("two"),
    )


def test_run_tool_returns_failure_for_invalid_arguments() -> None:
    checker = RecordingPermissionChecker(PermissionDecision.allow())
    registry = ToolRegistry.from_tools(
        [FakeTool(name="fake")],
        permission_checker=checker,
    )

    result = registry.run_tool("fake", {"text": 123})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert checker.requests == []


def test_run_tool_returns_failure_when_permission_check_raises() -> None:
    tool = BrokenPermissionTool(name="fake")
    registry = ToolRegistry.from_tools(
        [tool],
        permission_checker=RecordingPermissionChecker(PermissionDecision.allow()),
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result.ok is False
    assert result.error == "Tool permission check failed: broken permission metadata"
    assert result.metadata == {
        "tool_name": "fake",
        "exception_type": "ValueError",
    }
    assert tool.run_count == 0


def test_run_tool_returns_failure_when_permission_denied() -> None:
    tool = FakeTool(name="fake")
    registry = ToolRegistry.from_tools(
        [tool],
        permission_checker=RecordingPermissionChecker(
            PermissionDecision.deny(
                reason="unsupported_operation",
                message="Permission denied by test.",
                metadata={"detail": "blocked"},
            )
        ),
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result == ToolResult.failure(
        error="Permission denied by test.",
        metadata={
            "permission_status": "deny",
            "permission_reason": "unsupported_operation",
            "detail": "blocked",
        },
    )
    assert tool.run_count == 0


def test_run_tool_rejects_ask_decision_when_confirmation_is_unavailable() -> None:
    tool = FakeTool(name="fake")
    registry = ToolRegistry.from_tools(
        [tool],
        permission_checker=RecordingPermissionChecker(
            PermissionDecision.ask(
                message="Please confirm.",
                metadata={"detail": "needs confirmation"},
            )
        ),
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "requires_confirmation"
    assert result.metadata["confirmation_status"] == "rejected"
    assert result.metadata["detail"] == "needs confirmation"
    assert tool.run_count == 0


def test_run_tool_executes_after_confirmation_is_approved() -> None:
    tool = FakeTool(name="fake")
    confirmer = RecordingConfirmer(ConfirmationResult.approved("approved"))
    registry = ToolRegistry.from_tools(
        [tool],
        permission_checker=RecordingPermissionChecker(
            PermissionDecision.ask(message="Please confirm.")
        ),
        confirmer=confirmer,
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result.ok and result.content == "hello"
    assert result.metadata["confirmation_scope"] == "once"
    assert result.metadata["confirmation_source"] == "explicit"
    assert tool.run_count == 1
    assert len(confirmer.requests) == 1
    assert confirmer.requests[0].prompt == "Please confirm."
    assert confirmer.requests[0].permission_request.arguments == {"text": "hello"}


def test_run_tool_passes_confirmation_context_to_authorized_execution() -> None:
    tool = AuthorizedDecisionTool(name="fake")
    confirmer = RecordingConfirmer(
        ConfirmationResult.approved(
            "approved by test",
            metadata={"ticket": "confirm-1"},
        )
    )
    registry = ToolRegistry.from_tools(
        [tool],
        permission_checker=RecordingPermissionChecker(
            PermissionDecision.ask(
                message="Please confirm.",
                metadata={"detail": "needs confirmation"},
            )
        ),
        confirmer=confirmer,
    )

    result = registry.run_tool("fake", {"text": "hello"})

    assert result.ok is True
    assert result.metadata["decision_status"] == "allow"
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["confirmation_message"] == "approved by test"
    assert result.metadata["confirmation_metadata"] == {"ticket": "confirm-1"}
    assert result.metadata["detail"] == "needs confirmation"


class FakeArgs(ToolArgs):
    text: str


class AsyncOnlyTool(BaseTool[dict[str, object]]):
    name = "async_only"
    description = "Native async tool for registry tests."
    capability = "read"
    risk = "low"

    @property
    def input_schema(self) -> dict[str, object]:
        return {"type": "object"}

    def parse_arguments(
        self, arguments: dict[str, object]
    ) -> dict[str, object]:
        return dict(arguments)

    def arguments_to_dict(
        self, args: dict[str, object]
    ) -> dict[str, object]:
        return dict(args)

    async def run_authorized_async(
        self,
        args: dict[str, object],
        decision: PermissionDecision,
    ) -> ToolResult:
        return ToolResult.success(str(args["text"]))


class FakeTool(PydanticTool[FakeArgs]):
    args_model = FakeArgs
    capability = "read"
    risk = "low"

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Fake tool {name}."
        self.run_count = 0

    def _run(self, args: FakeArgs) -> ToolResult:
        self.run_count += 1
        return ToolResult.success(args.text)


class AuthorizedDecisionTool(FakeTool):
    def run_authorized(
        self,
        args: FakeArgs,
        decision: PermissionDecision,
    ) -> ToolResult:
        self.run_count += 1
        return ToolResult.success(
            args.text,
            metadata={
                "decision_status": decision.status,
                **decision.metadata,
            },
        )


class BrokenPermissionTool(FakeTool):
    def check_permission(
        self,
        args: FakeArgs,
        permission_checker: object,
    ) -> tuple[PermissionRequest, PermissionDecision]:
        raise ValueError("broken permission metadata")


class RecordingPermissionChecker:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.requests: list[PermissionRequest] = []
        self.profiles: list[ToolPermissionProfile] = []

    def check(
        self,
        request: PermissionRequest,
        profile: ToolPermissionProfile,
    ) -> PermissionDecision:
        self.requests.append(request)
        self.profiles.append(profile)
        return self.decision


class RecordingConfirmer:
    def __init__(self, result: ConfirmationResult) -> None:
        self.result = result
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return self.result
