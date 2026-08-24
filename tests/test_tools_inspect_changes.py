from pathlib import Path
import subprocess

from mycode.tools import InspectChangesTool, ToolRegistry, Workspace


def test_inspect_changes_status_reports_worktree_state(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "notes.txt").write_text("changed\n", encoding="utf-8")
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool("inspect_changes", {"action": "status"})

    assert result.ok is True
    assert "notes.txt" in result.content
    assert result.metadata["inspect_action"] == "status"
    assert result.metadata["permission_status"] == "allow"
    assert "--no-optional-locks" in result.metadata["command"]
    assert "core.fsmonitor=false" in result.metadata["command"]
    assert "--ignore-submodules=all" in result.metadata["command"]


def test_inspect_changes_diff_uses_safe_flags_and_explicit_paths(
    tmp_path: Path,
) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "notes.txt").write_text("changed\n", encoding="utf-8")
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": ["notes.txt"]},
    )

    assert result.ok is True
    assert "-original" in result.content
    assert "+changed" in result.content
    assert "--no-ext-diff" in result.metadata["command"]
    assert "--no-textconv" in result.metadata["command"]
    assert "--ignore-submodules=all" in result.metadata["command"]
    assert result.metadata["resolved_paths"] == [
        (tmp_path / "notes.txt").as_posix()
    ]


def test_inspect_changes_rejects_sensitive_diff_path(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": [".env"]},
    )

    assert result.ok is False
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert "secret" not in (result.error or "")


def test_inspect_changes_rejects_outside_workspace_path(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": ["../outside.py"]},
    )

    assert result.ok is False
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "outside_workspace"


def test_inspect_changes_requires_confirmation_for_ignored_path(
    tmp_path: Path,
) -> None:
    ignored_directory = tmp_path / ".venv"
    ignored_directory.mkdir()
    (ignored_directory / "debug.log").write_text("details\n", encoding="utf-8")
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": [".venv/debug.log"]},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_reason"] == "ignored_path"
    assert result.metadata["inspect_action"] == "diff"
    assert result.metadata["resolved_paths"] == [
        (ignored_directory / "debug.log").as_posix()
    ]


def test_inspect_changes_rejects_pathspec_and_option_like_ref(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    path_result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": ["*.py"]},
    )
    ref_result = registry.run_tool(
        "inspect_changes",
        {"action": "diff", "paths": ["app.py"], "base_ref": "HEAD:.env"},
    )

    assert path_result.ok is False
    assert path_result.error == "Invalid tool arguments"
    assert ref_result.ok is False
    assert ref_result.error == "Invalid tool arguments"


def test_inspect_changes_status_rejects_diff_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([InspectChangesTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "inspect_changes",
        {"action": "status", "paths": ["notes.txt"]},
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_inspect_changes_converts_execution_setup_error_to_tool_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = InspectChangesTool(Workspace(tmp_path))
    registry = ToolRegistry.from_tools([tool])

    def fail_build(*args, **kwargs):
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(tool, "_build_command", fail_build)
    result = registry.run_tool("inspect_changes", {"action": "status"})

    assert result.ok is False
    assert result.error == "Tool execution failed: synthetic build failure"
    assert result.metadata["exception_type"] == "RuntimeError"


def _initialize_repository(path: Path) -> None:
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "Tests")
    (path / "notes.txt").write_text("original\n", encoding="utf-8")
    _git(path, "add", "notes.txt")
    _git(path, "commit", "--quiet", "-m", "initial")


def _git(path: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
