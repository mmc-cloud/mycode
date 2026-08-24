from dataclasses import FrozenInstanceError

import pytest

from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    DefaultPermissionChecker,
    PermissionDecision,
    PermissionRequest,
    RejectingConfirmer,
    ToolPermissionProfile,
)


def test_tool_permission_profile_stores_capability_and_risk() -> None:
    profile = ToolPermissionProfile(capability="read", risk="low")

    assert profile.capability == "read"
    assert profile.risk == "low"


def test_tool_permission_profile_is_frozen() -> None:
    profile = ToolPermissionProfile(capability="read", risk="low")

    with pytest.raises(FrozenInstanceError):
        profile.capability = "write"


def test_permission_request_stores_capability_action_and_target() -> None:
    request = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
        target="README.md",
        arguments={"path": "README.md"},
        description="Read README.md",
    )

    assert request.tool_name == "read_file"
    assert request.capability == "read"
    assert request.action == "read_file"
    assert request.target == "README.md"
    assert request.arguments == {"path": "README.md"}
    assert request.description == "Read README.md"


def test_permission_request_defaults_are_independent() -> None:
    first = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
    )
    second = PermissionRequest(
        tool_name="grep",
        capability="read",
        action="search_text",
    )

    first.arguments["path"] = "README.md"

    assert second.arguments == {}


def test_permission_request_copies_arguments() -> None:
    arguments: dict[str, object] = {"path": "README.md"}

    request = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
        arguments=arguments,
    )
    arguments["path"] = "changed.md"

    assert request.arguments == {"path": "README.md"}


def test_permission_request_is_frozen() -> None:
    request = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
    )

    with pytest.raises(FrozenInstanceError):
        request.tool_name = "grep"


def test_permission_decision_allow_factory() -> None:
    decision = PermissionDecision.allow(
        message="Read-only operation is allowed.",
        metadata={"tool_name": "read_file"},
    )

    assert decision.status == "allow"
    assert decision.reason == "allowed"
    assert decision.message == "Read-only operation is allowed."
    assert decision.metadata == {"tool_name": "read_file"}


def test_permission_decision_deny_factory() -> None:
    decision = PermissionDecision.deny(
        reason="sensitive_path",
        message="Refusing to access sensitive file.",
        metadata={"path": ".env"},
    )

    assert decision.status == "deny"
    assert decision.reason == "sensitive_path"
    assert decision.message == "Refusing to access sensitive file."
    assert decision.metadata == {"path": ".env"}


def test_permission_decision_ask_factory() -> None:
    decision = PermissionDecision.ask(
        message="Writing a file requires confirmation.",
        metadata={"path": "mycode/app.py"},
    )

    assert decision.status == "ask"
    assert decision.reason == "requires_confirmation"
    assert decision.message == "Writing a file requires confirmation."
    assert decision.metadata == {"path": "mycode/app.py"}


def test_permission_decision_defaults_are_independent() -> None:
    first = PermissionDecision.allow()
    second = PermissionDecision.allow()

    first.metadata["tool_name"] = "read_file"

    assert second.metadata == {}


def test_permission_decision_copies_metadata() -> None:
    metadata: dict[str, object] = {"path": ".env"}

    decision = PermissionDecision.deny(
        reason="sensitive_path",
        metadata=metadata,
    )
    metadata["path"] = "changed.env"

    assert decision.metadata == {"path": ".env"}


def test_permission_decision_is_frozen() -> None:
    decision = PermissionDecision.allow()

    with pytest.raises(FrozenInstanceError):
        decision.status = "deny"


def test_default_permission_checker_allows_read_capability() -> None:
    checker = DefaultPermissionChecker()
    request = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
        target="README.md",
    )
    profile = ToolPermissionProfile(capability="read", risk="low")

    decision = checker.check(request, profile)

    assert decision.status == "allow"
    assert decision.reason == "allowed"
    assert decision.message == "Read operation allowed: read_file"
    assert decision.metadata == {
        "tool_name": "read_file",
        "capability": "read",
        "action": "read_file",
        "target": "README.md",
        "risk": "low",
    }


def test_default_permission_checker_allows_internal_control_capability() -> None:
    checker = DefaultPermissionChecker()
    request = PermissionRequest(
        tool_name="submit_result",
        capability="control",
        action="submit_result",
    )
    profile = ToolPermissionProfile(capability="control", risk="low")

    decision = checker.check(request, profile)

    assert decision.status == "allow"
    assert decision.reason == "allowed"
    assert decision.message == "Internal control operation allowed: submit_result"
    assert decision.metadata["capability"] == "control"


def test_default_permission_checker_asks_for_write_capability() -> None:
    checker = DefaultPermissionChecker()
    request = PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write_file",
        target="mycode/app.py",
    )
    profile = ToolPermissionProfile(capability="write", risk="medium")

    decision = checker.check(request, profile)

    assert decision.status == "ask"
    assert decision.reason == "requires_confirmation"
    assert decision.message == "Write operation requires confirmation: write_file"
    assert decision.metadata["tool_name"] == "write_file"
    assert decision.metadata["risk"] == "medium"


def test_default_permission_checker_asks_for_command_capability() -> None:
    checker = DefaultPermissionChecker()
    request = PermissionRequest(
        tool_name="run_command",
        capability="command",
        action="run_command",
        target="uv run pytest",
    )
    profile = ToolPermissionProfile(capability="command", risk="high")

    decision = checker.check(request, profile)

    assert decision.status == "ask"
    assert decision.reason == "requires_confirmation"
    assert decision.message == "Command operation requires confirmation: run_command"
    assert decision.metadata["target"] == "uv run pytest"
    assert decision.metadata["risk"] == "high"


def test_default_permission_checker_denies_capability_mismatch() -> None:
    checker = DefaultPermissionChecker()
    request = PermissionRequest(
        tool_name="read_file",
        capability="write",
        action="write_file",
        target="README.md",
    )
    profile = ToolPermissionProfile(capability="read", risk="low")

    decision = checker.check(request, profile)

    assert decision.status == "deny"
    assert decision.reason == "unsupported_operation"
    assert decision.message == "Tool read_file cannot perform write operations."
    assert decision.metadata == {
        "tool_name": "read_file",
        "capability": "write",
        "action": "write_file",
        "target": "README.md",
        "risk": "low",
        "tool_capability": "read",
        "requested_capability": "write",
    }


def test_confirmation_request_stores_permission_context() -> None:
    permission_request = write_request()
    permission_decision = PermissionDecision.ask(
        message="Writing requires confirmation.",
        metadata={"path": "notes.txt"},
    )
    confirmation_request = ConfirmationRequest(
        permission_request=permission_request,
        permission_decision=permission_decision,
        prompt="Allow writing notes.txt?",
        metadata={"source": "cli"},
    )

    assert confirmation_request.permission_request == permission_request
    assert confirmation_request.permission_decision == permission_decision
    assert confirmation_request.prompt == "Allow writing notes.txt?"
    assert confirmation_request.metadata == {"source": "cli"}


def test_confirmation_request_copies_metadata() -> None:
    metadata: dict[str, object] = {"source": "cli"}

    request = ConfirmationRequest(
        permission_request=write_request(),
        permission_decision=PermissionDecision.ask(),
        prompt="Allow?",
        metadata=metadata,
    )
    metadata["source"] = "changed"

    assert request.metadata == {"source": "cli"}


def test_confirmation_request_is_frozen() -> None:
    request = ConfirmationRequest(
        permission_request=write_request(),
        permission_decision=PermissionDecision.ask(),
        prompt="Allow?",
    )

    with pytest.raises(FrozenInstanceError):
        request.prompt = "Changed?"


def test_confirmation_result_approved_factory() -> None:
    result = ConfirmationResult.approved(
        message="Approved by user.",
        metadata={"source": "cli"},
    )

    assert result.status == "approved"
    assert result.message == "Approved by user."
    assert result.metadata == {"source": "cli"}


def test_confirmation_result_rejected_factory() -> None:
    result = ConfirmationResult.rejected(
        message="Rejected by user.",
        metadata={"source": "cli"},
    )

    assert result.status == "rejected"
    assert result.message == "Rejected by user."
    assert result.metadata == {"source": "cli"}


def test_confirmation_result_defaults_are_independent() -> None:
    first = ConfirmationResult.approved()
    second = ConfirmationResult.approved()

    first.metadata["source"] = "cli"

    assert second.metadata == {}


def test_confirmation_result_copies_metadata() -> None:
    metadata: dict[str, object] = {"source": "cli"}

    result = ConfirmationResult.approved(metadata=metadata)
    metadata["source"] = "changed"

    assert result.metadata == {"source": "cli"}


def test_confirmation_result_is_frozen() -> None:
    result = ConfirmationResult.approved()

    with pytest.raises(FrozenInstanceError):
        result.status = "rejected"


def test_fake_confirmer_can_implement_confirm_protocol_shape() -> None:
    confirmer = RecordingConfirmer(ConfirmationResult.approved(message="yes"))
    request = ConfirmationRequest(
        permission_request=write_request(),
        permission_decision=PermissionDecision.ask(),
        prompt="Allow?",
    )

    result = confirmer.confirm(request)

    assert result == ConfirmationResult.approved(message="yes")
    assert confirmer.requests == [request]


def test_rejecting_confirmer_rejects_when_confirmation_is_unavailable() -> None:
    confirmer = RejectingConfirmer()
    request = ConfirmationRequest(
        permission_request=write_request(),
        permission_decision=PermissionDecision.ask(),
        prompt="Allow writing notes.txt?",
    )

    result = confirmer.confirm(request)

    assert result.status == "rejected"
    assert result.message == "Confirmation is not available."
    assert result.metadata == {
        "tool_name": "write_file",
        "capability": "write",
        "action": "write_file",
        "target": "notes.txt",
    }


def write_request() -> PermissionRequest:
    return PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write_file",
        target="notes.txt",
        arguments={"path": "notes.txt"},
    )


class RecordingConfirmer:
    def __init__(self, result: ConfirmationResult) -> None:
        self.result = result
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return self.result
