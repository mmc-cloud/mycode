from mycode.tools.command_risk import INSPECT_COMMANDS, analyze_command_risk


def test_analyze_command_risk_identifies_test_command() -> None:
    risk = analyze_command_risk(["uv", "run", "pytest"])

    assert risk.category == "test"
    assert risk.risk == "low"
    assert risk.decision == "ask"


def test_analyze_command_risk_identifies_python_version_command() -> None:
    risk = analyze_command_risk(["python", "--version"])

    assert risk.category == "inspect"
    assert risk.risk == "low"
    assert risk.decision == "ask"


def test_python_commands_are_not_generic_inspect_commands() -> None:
    assert {"python", "python3", "py"}.isdisjoint(INSPECT_COMMANDS)


def test_git_is_not_a_generic_inspect_command() -> None:
    assert "git" not in INSPECT_COMMANDS


def test_analyze_command_risk_identifies_python_inline_as_medium() -> None:
    risk = analyze_command_risk(["python", "-c", "print('hello')"])

    assert risk.category == "python_inline"
    assert risk.risk == "medium"
    assert risk.decision == "ask"


def test_analyze_command_risk_denies_python_inline_delete() -> None:
    risk = analyze_command_risk(
        ["python", "-c", "from pathlib import Path; Path('notes.txt').unlink()"]
    )

    assert risk.category == "delete_or_destructive"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_delegates_uv_run_python() -> None:
    risk = analyze_command_risk(["uv", "run", "python", "-c", "print('hello')"])

    assert risk.category == "python_inline"
    assert risk.risk == "medium"
    assert risk.decision == "ask"


def test_analyze_command_risk_denies_uv_run_python_inline_delete() -> None:
    risk = analyze_command_risk(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pathlib import Path; Path('notes.txt').unlink()",
        ]
    )

    assert risk.category == "delete_or_destructive"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_identifies_uv_version_command() -> None:
    risk = analyze_command_risk(["uv", "--version"])

    assert risk.category == "inspect"
    assert risk.risk == "low"
    assert risk.decision == "ask"


def test_analyze_command_risk_identifies_dependency_install() -> None:
    risk = analyze_command_risk(["uv", "pip", "install", "rich"])

    assert risk.category == "install_dependency"
    assert risk.risk == "medium"
    assert risk.decision == "ask"


def test_analyze_command_risk_identifies_network_command() -> None:
    risk = analyze_command_risk(["curl", "https://example.com"])

    assert risk.category == "network_access"
    assert risk.risk == "medium"
    assert risk.decision == "ask"


def test_analyze_command_risk_identifies_unknown_command() -> None:
    risk = analyze_command_risk(["custom-tool", "--version"])

    assert risk.category == "unknown"
    assert risk.risk == "medium"
    assert risk.decision == "ask"


def test_analyze_command_risk_denies_additional_high_risk_commands() -> None:
    commands = [
        ["dd", "if=a", "of=b"],
        ["diskpart"],
        ["reboot"],
        ["kill", "1234"],
        ["killall", "python"],
    ]

    for command in commands:
        risk = analyze_command_risk(command)

        assert risk.risk == "high"
        assert risk.decision == "deny"


def test_analyze_command_risk_identifies_git_status_as_inspect() -> None:
    risk = analyze_command_risk(["git", "status"])

    assert risk.category == "inspect"
    assert risk.risk == "low"
    assert risk.decision == "ask"


def test_analyze_command_risk_denies_git_checkout() -> None:
    risk = analyze_command_risk(["git", "checkout", "--", "notes.txt"])

    assert risk.category == "git_checkout"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_git_reset_after_global_option() -> None:
    risk = analyze_command_risk(["git", "-c", "x", "reset", "--hard"])

    assert risk.category == "git_reset"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_git_checkout_after_global_option() -> None:
    risk = analyze_command_risk(["git", "-C", "repo", "checkout", "--", "notes.txt"])

    assert risk.category == "git_checkout"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_git_restore() -> None:
    risk = analyze_command_risk(["git", "restore", "notes.txt"])

    assert risk.category == "git_restore"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_git_force_push() -> None:
    risk = analyze_command_risk(["git", "push", "--force-with-lease"])

    assert risk.category == "git_force_push"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_git_force_push_after_global_option() -> None:
    risk = analyze_command_risk(["git", "-c", "x", "push", "--force"])

    assert risk.category == "git_force_push"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_denies_posix_shell_wrapped_delete() -> None:
    risk = analyze_command_risk(["bash", "-c", "rm -rf notes.txt"])

    assert risk.category == "delete_or_destructive"
    assert risk.risk == "high"
    assert risk.decision == "deny"


def test_analyze_command_risk_requires_high_risk_confirmation_for_posix_shell() -> None:
    risk = analyze_command_risk(["sh", "-c", "pytest -q"])

    assert risk.category == "shell_wrapper"
    assert risk.risk == "high"
    assert risk.decision == "ask"


def test_analyze_command_risk_identifies_common_local_inspection_commands() -> None:
    for command in (
        ["sed", "-n", "1,80p", "app.py"],
        ["find", ".", "-name", "*.py"],
        ["diff", "expected.txt", "actual.txt"],
        ["xxd", "database.db-wal"],
    ):
        risk = analyze_command_risk(command)

        assert risk.category == "inspect"
        assert risk.risk == "low"
        assert risk.decision == "ask"


def test_analyze_command_risk_identifies_git_version_as_inspect() -> None:
    risk = analyze_command_risk(["git", "--version"])

    assert risk.category == "inspect"
    assert risk.risk == "low"
    assert risk.decision == "ask"


def test_analyze_command_risk_treats_unhandled_git_as_state_change() -> None:
    risk = analyze_command_risk(["git", "commit", "-m", "change"])

    assert risk.category == "git_operation"
    assert risk.risk == "medium"
    assert risk.decision == "ask"
