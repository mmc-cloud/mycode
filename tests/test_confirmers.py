from mycode.confirmers import TerminalConfirmer
from mycode.permissions import ConfirmationRequest, PermissionDecision, PermissionRequest


def test_terminal_confirmer_approves_yes_input() -> None:
    outputs: list[str] = []
    prompts: list[str] = []
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: prompts.append(prompt) or "y",
        output_func=outputs.append,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "approved"
    assert result.message == "Permission confirmation approved."
    assert prompts == ["是否批准？[y/yes 本次 | t/task 当前任务 | s/session 当前会话 | N 拒绝] "]
    assert outputs == [
        "permission> read_file 需要确认",
        "target> .env",
        "resolved_path> C:/repo/.env",
        "workspace_root> C:/repo",
        "path_scope> sensitive_path",
        "reason> sensitive_path",
        "message> 敏感路径需要确认。",
    ]


def test_terminal_confirmer_approves_full_yes_input() -> None:
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: "YES",
        output_func=lambda message: None,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "approved"


def test_terminal_confirmer_outputs_command_metadata() -> None:
    outputs: list[str] = []
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: "y",
        output_func=outputs.append,
    )

    result = confirmer.confirm(_command_confirmation_request())

    assert result.status == "approved"
    assert outputs == [
        "permission> run_command 需要确认",
        "target> uv run pytest",
        "workspace_root> C:/repo",
        "command_display> uv run pytest",
        "resolved_cwd> C:/repo",
        "cwd_scope> inside_workspace",
        "command_risk_category> test",
        "command_risk> low",
        "command_risk_reason> Command appears to run tests or validation.",
        "reason> requires_confirmation",
        "message> 命令操作需要确认：run_command",
    ]


def test_terminal_confirmer_outputs_memory_content_for_explicit_review() -> None:
    outputs: list[str] = []
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: "y",
        output_func=outputs.append,
    )
    decision = PermissionDecision.ask(
        message="Write operation requires confirmation: save user memory",
        metadata={
            "memory_scope": "user",
            "memory_kind": "preference",
            "memory_key": "response.language",
            "memory_content": "默认使用中文。",
            "memory_path": "C:/Users/test/.mycode/MEMORY.md",
        },
    )
    request = ConfirmationRequest(
        permission_request=PermissionRequest(
            tool_name="save_memory",
            capability="write",
            action="save user memory",
            target="C:/Users/test/.mycode/MEMORY.md",
        ),
        permission_decision=decision,
        prompt=decision.message,
        metadata=decision.metadata,
    )

    result = confirmer.confirm(request)

    assert result.status == "approved"
    assert "memory_scope> user" in outputs
    assert "memory_kind> preference" in outputs
    assert "memory_key> response.language" in outputs
    assert "memory_content> '默认使用中文。'" in outputs
    assert "memory_path> C:/Users/test/.mycode/MEMORY.md" in outputs


def test_terminal_confirmer_rejects_empty_input() -> None:
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: "",
        output_func=lambda message: None,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "rejected"
    assert result.message == "Permission confirmation rejected."


def test_terminal_confirmer_rejects_unknown_input() -> None:
    confirmer = TerminalConfirmer(
        input_func=lambda prompt: "maybe",
        output_func=lambda message: None,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "rejected"
    assert result.message == "Permission confirmation rejected."


def test_terminal_confirmer_rejects_eof() -> None:
    def raise_eof(prompt: str) -> str:
        raise EOFError

    confirmer = TerminalConfirmer(
        input_func=raise_eof,
        output_func=lambda message: None,
    )

    result = confirmer.confirm(_confirmation_request())

    assert result.status == "rejected"
    assert result.message == "Permission confirmation unavailable."


def _confirmation_request() -> ConfirmationRequest:
    permission_request = PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
        target=".env",
        arguments={"path": ".env"},
    )
    permission_decision = PermissionDecision.ask(
        reason="sensitive_path",
        message="Sensitive path requires confirmation.",
        metadata={
            "resolved_path": "C:/repo/.env",
            "workspace_root": "C:/repo",
            "path_scope": "sensitive_path",
        },
    )
    return ConfirmationRequest(
        permission_request=permission_request,
        permission_decision=permission_decision,
        prompt=permission_decision.message,
    )


def _command_confirmation_request() -> ConfirmationRequest:
    permission_request = PermissionRequest(
        tool_name="run_command",
        capability="command",
        action="run_command",
        target="uv run pytest",
        arguments={"command": ["uv", "run", "pytest"]},
    )
    permission_decision = PermissionDecision.ask(
        message="Command operation requires confirmation: run_command",
        metadata={
            "workspace_root": "C:/repo",
            "command_display": "uv run pytest",
            "resolved_cwd": "C:/repo",
            "cwd_scope": "inside_workspace",
            "command_risk_category": "test",
            "command_risk": "low",
            "command_risk_reason": "Command appears to run tests or validation.",
        },
    )
    return ConfirmationRequest(
        permission_request=permission_request,
        permission_decision=permission_decision,
        prompt=permission_decision.message,
    )
