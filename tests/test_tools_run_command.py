from pathlib import Path
import importlib
import json
import os
import sys
import time

import pytest

from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.tools import RunCommandArgs, RunCommandTool, ToolRegistry, Workspace


def test_run_command_schema_describes_arguments(tmp_path: Path) -> None:
    schema = RunCommandTool(workspace=Workspace(tmp_path)).get_schema()

    assert schema["name"] == "run_command"
    assert schema["parameters"]["properties"]["command"]["type"] == "array"
    assert schema["parameters"]["properties"]["command"]["items"]["type"] == "string"
    assert schema["parameters"]["properties"]["cwd"]["default"] == "."
    assert schema["parameters"]["properties"]["timeout_seconds"]["default"] == 30.0
    assert schema["parameters"]["properties"]["max_output_chars"]["default"] == 12000


def test_run_command_args_reject_empty_command_part() -> None:
    registry = ToolRegistry.from_tools([RunCommandTool(Workspace(Path.cwd()))])

    result = registry.run_tool("run_command", {"command": [sys.executable, ""]})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_run_command_normalizes_json_encoded_argv_before_permission(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": json.dumps(
                [sys.executable, "-c", "print('normalized')"]
            )
        },
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "normalized\n"
    assert result.metadata["command"] == [
        sys.executable,
        "-c",
        "print('normalized')",
    ]
    assert result.metadata["command_risk_category"] == "python_inline"
    assert confirmer.requests[0].permission_request.arguments["command"] == [
        sys.executable,
        "-c",
        "print('normalized')",
    ]


def test_run_command_denies_json_encoded_dangerous_command_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": '["git", "reset", "--hard"]'},
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command"] == ["git", "reset", "--hard"]
    assert confirmer.requests == []


def test_run_command_denies_json_encoded_posix_shell_delete_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": '["bash", "-c", "rm -rf notes.txt"]'},
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "delete_or_destructive"
    assert confirmer.requests == []


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest",
        "['python', '-m', 'pytest']",
        "- python\n- -m\n- pytest",
        '{"program":"python"}',
        '["python", 1]',
        '"[\\"python\\"]"',
        "command=[\"python\"]",
    ],
)
def test_run_command_rejects_unstructured_command_strings(
    command: str,
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools([RunCommandTool(Workspace(tmp_path))])

    result = registry.run_tool("run_command", {"command": command})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_run_command_rejects_null_byte_after_json_normalization(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools([RunCommandTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "run_command",
        {"command": '["python", "bad\\u0000argument"]'},
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_run_command_permission_request_targets_display_command(
    tmp_path: Path,
) -> None:
    tool = RunCommandTool(workspace=Workspace(tmp_path))

    request = tool.build_permission_request(
        RunCommandArgs(command=[sys.executable, "-c", "print('hello')"])
    )

    assert request.tool_name == "run_command"
    assert request.capability == "command"
    assert request.action == "run_command"
    assert sys.executable in str(request.target)
    assert request.arguments["command"] == [sys.executable, "-c", "print('hello')"]


def test_run_command_direct_run_requires_registry(tmp_path: Path) -> None:
    tool = RunCommandTool(workspace=Workspace(tmp_path))

    result = tool.run({"command": [sys.executable, "-c", "print('hello')"]})

    assert result.ok is False
    assert result.error == "run_command must be run through ToolRegistry.run_tool()."


def test_run_command_requires_confirmation_by_default(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([RunCommandTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"]},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "requires_confirmation"
    assert result.metadata["cwd_scope"] == "inside_workspace"
    assert result.metadata["command"] == [sys.executable, "-c", "print('hello')"]
    assert result.metadata["command_risk_category"] == "python_inline"
    assert result.metadata["command_risk"] == "medium"
    assert result.metadata["command_risk_decision"] == "ask"


def test_run_command_runs_after_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"]},
    )

    assert result.ok is True
    assert result.content == "Command exited with code 0.\n\nSTDOUT\nhello\n"
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert result.metadata["stdout"] == "hello\n"
    assert result.metadata["stderr"] == ""
    assert result.metadata["stdout_truncated"] is False
    assert result.metadata["stderr_truncated"] is False
    assert result.metadata["output_capture_complete"] is True
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["command_risk_category"] == "python_inline"


def test_run_command_runs_in_workspace_subdirectory(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('created.txt').write_text('ok')",
            ],
            "cwd": "workdir",
        },
    )

    assert result.ok is True
    assert (workdir / "created.txt").read_text(encoding="utf-8") == "ok"
    assert result.metadata["cwd"] == "workdir"
    assert result.metadata["resolved_cwd"] == workdir.as_posix()


def test_run_command_returns_nonzero_exit_code(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "import sys; sys.exit(7)"]},
    )

    assert result.ok is False
    assert result.error == "Command exited with code 7."
    assert result.metadata["exit_code"] == 7


def test_run_command_closes_stdin_for_non_interactive_commands(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; data = sys.stdin.read(); print('eof' if data == '' else 'input')",
            ],
            "timeout_seconds": 1,
        },
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "eof\n"
    assert result.metadata["exit_code"] == 0


def test_run_command_times_out(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": [sys.executable, "-c", "import time; time.sleep(5)"],
            "timeout_seconds": 0.5,
        },
    )

    assert result.ok is False
    assert result.error == "Command timed out after 0.5 seconds."
    assert result.metadata["timed_out"] is True
    assert result.metadata["exit_code"] is None


def test_run_command_denies_outside_workspace_cwd(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"], "cwd": ".."},
    )

    assert result.ok is False
    assert result.error == "Command working directory is outside workspace: .."
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "outside_workspace"
    assert result.metadata["cwd_scope"] == "outside_workspace"
    assert confirmer.requests == []


def test_run_command_denies_missing_cwd(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"], "cwd": "missing"},
    )

    assert result.ok is False
    assert result.error == "Command working directory does not exist: missing"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "unsupported_operation"
    assert confirmer.requests == []


def test_run_command_denies_file_cwd(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir", encoding="utf-8")
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"], "cwd": "file.txt"},
    )

    assert result.ok is False
    assert result.error == "Command working directory is not a directory: file.txt"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "unsupported_operation"
    assert confirmer.requests == []


def test_run_command_truncates_stdout_and_stderr(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; print('abcdef', end=''); print('uvwxyz', end='', file=sys.stderr)",
            ],
            "max_output_chars": 3,
        },
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "def"
    assert result.metadata["stderr"] == "xyz"
    assert result.metadata["stdout_chars"] == 6
    assert result.metadata["stderr_chars"] == 6
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True
    assert result.metadata["output_truncation_strategy"] == "tail"


def test_run_command_uses_streaming_capture_instead_of_subprocess_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    command_executor_module = importlib.import_module("mycode.tools.command_executor")

    def fail_run(*args, **kwargs):
        raise AssertionError("run_command should stream output through Popen")

    monkeypatch.setattr(command_executor_module.subprocess, "run", fail_run)
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"]},
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "hello\n"


def test_run_command_streaming_capture_keeps_only_output_tail(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('a' * 50000 + 'TAIL'); sys.stderr.write('b' * 40000 + 'ERR!')",
            ],
            "max_output_chars": 4,
        },
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "TAIL"
    assert result.metadata["stderr"] == "ERR!"
    assert result.metadata["stdout_chars"] == 50004
    assert result.metadata["stderr_chars"] == 40004
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True


def test_run_command_cleans_up_grandchild_that_keeps_pipe_open(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )
    child_pid_path = tmp_path / "child.pid"

    result = registry.run_tool(
        "run_command",
        {
            "command": [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys, time; "
                    "time.sleep(0.5); "
                    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                    "stdout=sys.stdout, stderr=sys.stderr); "
                    "pathlib.Path('child.pid').write_text(str(child.pid), encoding='utf-8'); "
                    "print('parent-done')"
                ),
            ],
            "timeout_seconds": 5,
        },
    )

    assert result.ok is True
    assert result.metadata["stdout"] == "parent-done\n"
    assert result.metadata["timed_out"] is False
    assert result.metadata["output_capture_complete"] is True
    assert result.metadata["process_tree_cleanup_success"] is True
    assert _process_exits_soon(int(child_pid_path.read_text(encoding="utf-8")))


def test_run_command_requires_confirmation_for_test_command(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([RunCommandTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "run_command",
        {"command": ["uv", "run", "pytest"]},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["command_risk_category"] == "test"
    assert result.metadata["command_risk"] == "low"
    assert result.metadata["command_risk_decision"] == "ask"


def test_run_command_confirmation_request_includes_risk_metadata(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "-c", "print('hello')"]},
    )

    assert result.ok is True
    assert confirmer.requests[0].metadata["command_risk_category"] == "python_inline"
    assert confirmer.requests[0].metadata["command_risk"] == "medium"
    assert confirmer.requests[0].metadata["command_risk_decision"] == "ask"


def test_run_command_denies_delete_command_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["rm", "-rf", "notes.txt"]},
    )

    assert result.ok is False
    assert result.error == "Destructive command is blocked: rm"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "delete_or_destructive"
    assert result.metadata["command_risk"] == "high"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


def test_run_command_denies_git_reset_before_confirmation(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["git", "reset", "--hard"]},
    )

    assert result.ok is False
    assert result.error == "Git reset is blocked because it can discard work."
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "git_reset"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


def test_run_command_denies_git_checkout_before_confirmation(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["git", "checkout", "--", "notes.txt"]},
    )

    assert result.ok is False
    assert result.error == "Git checkout is blocked because it can discard work."
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "git_checkout"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


def test_run_command_denies_system_command_before_confirmation(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["chmod", "777", "notes.txt"]},
    )

    assert result.ok is False
    assert result.error == "System-level command is blocked: chmod"
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "system_operation"
    assert confirmer.requests == []


def test_run_command_denies_cmd_wrapped_delete_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["cmd", "/c", "del", "notes.txt"]},
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "delete_or_destructive"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


def test_run_command_denies_powershell_wrapped_delete_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["powershell", "-Command", "Remove-Item notes.txt"]},
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "delete_or_destructive"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


def test_run_command_denies_pwsh_wrapped_delete_before_confirmation(
    tmp_path: Path,
) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [RunCommandTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "run_command",
        {"command": ["pwsh", "-Command", "Remove-Item notes.txt"]},
    )

    assert result.ok is False
    assert result.metadata["permission_reason"] == "dangerous_command"
    assert result.metadata["command_risk_category"] == "delete_or_destructive"
    assert result.metadata["command_risk_decision"] == "deny"
    assert confirmer.requests == []


class ApprovingConfirmer:
    def __init__(self) -> None:
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return ConfirmationResult.approved()


def _process_exits_soon(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return True
        time.sleep(0.1)

    return not _process_is_running(pid)


def _process_is_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False

        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except OSError:
        return False

    return True
