from pathlib import Path
import sys

from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.memory import MemoryStore
from mycode.session_store import ProjectIdentity
from mycode.tools import Workspace, create_default_tool_registry, create_read_only_tool_registry
from mycode.tools import GlobTool, GrepTool, ReadFileTool, ToolRegistry


def test_create_read_only_tool_registry_registers_read_only_tools(
    tmp_path: Path,
) -> None:
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    assert [tool.name for tool in registry.list_tools()] == [
        "read_file",
        "glob",
        "grep",
    ]


def test_create_read_only_tool_registry_exports_all_tool_schemas(
    tmp_path: Path,
) -> None:
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    assert [schema["name"] for schema in registry.get_schemas()] == [
        "read_file",
        "glob",
        "grep",
    ]


def test_create_read_only_tool_registry_registers_read_only_low_risk_tools(
    tmp_path: Path,
) -> None:
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    assert [
        (tool.name, tool.get_permission_profile().capability, tool.get_permission_profile().risk)
        for tool in registry.list_tools()
    ] == [
        ("read_file", "read", "low"),
        ("glob", "read", "low"),
        ("grep", "read", "low"),
    ]


def test_create_default_tool_registry_registers_read_and_write_tools(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(Workspace(tmp_path))

    assert [tool.name for tool in registry.list_tools()] == [
        "read_file",
        "glob",
        "grep",
        "write_file",
        "edit_file",
        "run_command",
        "run_validation",
    ]


def test_create_default_tool_registry_exports_all_tool_schemas(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(Workspace(tmp_path))

    assert [schema["name"] for schema in registry.get_schemas()] == [
        "read_file",
        "glob",
        "grep",
        "write_file",
        "edit_file",
        "run_command",
        "run_validation",
    ]


def test_create_default_tool_registry_registers_expected_risk_profiles(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(Workspace(tmp_path))

    assert [
        (tool.name, tool.get_permission_profile().capability, tool.get_permission_profile().risk)
        for tool in registry.list_tools()
    ] == [
        ("read_file", "read", "low"),
        ("glob", "read", "low"),
        ("grep", "read", "low"),
        ("write_file", "write", "medium"),
        ("edit_file", "write", "medium"),
        ("run_command", "command", "high"),
        ("run_validation", "command", "high"),
    ]


def test_default_registry_only_marks_read_tools_as_concurrency_safe(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(Workspace(tmp_path))

    assert registry.is_concurrency_safe("read_file") is True
    assert registry.is_concurrency_safe("glob") is True
    assert registry.is_concurrency_safe("grep") is True
    assert registry.is_concurrency_safe("write_file") is False
    assert registry.is_concurrency_safe("edit_file") is False
    assert registry.is_concurrency_safe("run_command") is False
    assert registry.is_concurrency_safe("run_validation") is False


def test_create_default_tool_registry_adds_memory_tools_when_store_is_provided(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    store = MemoryStore(
        ProjectIdentity.from_workspace(tmp_path),
        base_directory=tmp_path / "user-state",
    )

    registry = create_default_tool_registry(workspace, memory_store=store)

    assert [tool.name for tool in registry.list_tools()][-3:] == [
        "list_memories",
        "save_memory",
        "delete_memory",
    ]
    assert [tool.capability for tool in registry.list_tools()][-3:] == [
        "read",
        "write",
        "write",
    ]


def test_default_registry_can_write_and_edit_after_confirmation(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )

    write_result = registry.run_tool(
        "write_file",
        {"path": "notes.txt", "content": "hello old"},
    )
    edit_result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "old", "new_text": "new"},
    )

    assert write_result.ok is True
    assert edit_result.ok is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello new"


def test_default_registry_can_run_command_after_confirmation(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_command",
        {"command": [sys.executable, "--version"]},
    )

    assert result.ok is True
    assert result.metadata["command_risk_category"] == "inspect"
    assert result.metadata["confirmation_status"] == "approved"


def test_default_registry_can_run_project_validation_after_confirmation(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "run_validation",
        {"command": [sys.executable, "-c", "raise SystemExit(0)"]},
    )

    assert result.ok is True
    assert result.metadata["command_risk_category"] == "python_inline"
    assert result.metadata["confirmation_status"] == "approved"


def test_read_only_registry_can_run_read_file(tmp_path: Path) -> None:
    file_path = tmp_path / "README.md"
    file_path.write_text("hello\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("read_file", {"path": "README.md"})

    assert result.ok is True
    assert result.content == "1 | hello"


def test_read_only_registry_requires_confirmation_for_sensitive_read_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / ".env"
    file_path.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("read_file", {"path": ".env"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["confirmation_status"] == "rejected"
    assert result.metadata["path"] == ".env"
    assert result.metadata["path_scope"] == "sensitive_path"


def test_read_only_registry_reads_env_example_without_confirmation(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / ".env.example"
    file_path.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("read_file", {"path": ".env.example"})

    assert result.ok is True
    assert result.content == "1 | OPENAI_API_KEY="
    assert result.metadata["path"] == ".env.example"


def test_read_only_registry_reads_sensitive_file_after_confirmation(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / ".env"
    file_path.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = create_read_only_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("read_file", {"path": ".env"})

    assert result.ok is True
    assert result.content == "1 | OPENAI_API_KEY=secret"
    assert result.metadata["path"] == ".env"


def test_read_only_registry_reads_outside_workspace_file_after_confirmation(
    tmp_path: Path,
) -> None:
    file_path = tmp_path.parent / "outside.txt"
    file_path.write_text("outside\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [ReadFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("read_file", {"path": "../outside.txt"})

    assert result.ok is True
    assert result.content == "1 | outside"
    assert result.metadata["path"] == file_path.as_posix()


def test_read_only_registry_can_run_glob(tmp_path: Path) -> None:
    file_path = tmp_path / "mycode" / "cli.py"
    file_path.parent.mkdir()
    file_path.write_text("", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("glob", {"pattern": "mycode/*.py"})

    assert result.ok is True
    assert result.content == "mycode/cli.py"


def test_read_only_registry_filters_sensitive_files_from_broad_glob(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("glob", {"pattern": "**/*"})

    assert result.ok is True
    assert result.content == "app.py"


def test_read_only_registry_requires_confirmation_for_explicit_sensitive_glob(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("glob", {"pattern": ".env"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["path_scope"] == "sensitive_path"


def test_read_only_registry_globs_explicit_sensitive_file_after_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [GlobTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("glob", {"pattern": ".env"})

    assert result.ok is True
    assert result.content == ".env"


def test_read_only_registry_requires_confirmation_for_sensitive_glob_pattern(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("glob", {"pattern": "**/.env*"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["pattern_scope"] == "sensitive_pattern"


def test_read_only_registry_globs_env_example_pattern_without_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("glob", {"pattern": "**/.env.example"})

    assert result.ok is True
    assert result.content == ".env.example"


def test_read_only_registry_globs_sensitive_pattern_after_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [GlobTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("glob", {"pattern": "**/.env*"})

    assert result.ok is True
    assert result.content == ".env.local"


def test_read_only_registry_denies_outside_workspace_glob_without_confirmation(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools(
        [GlobTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("glob", {"pattern": "../outside.txt"})

    assert result.ok is False
    assert result.error == "Glob pattern must not contain '..': ../outside.txt"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "outside_workspace"


def test_read_only_registry_can_run_grep(tmp_path: Path) -> None:
    file_path = tmp_path / "mycode" / "cli.py"
    file_path.parent.mkdir()
    file_path.write_text("def main():\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool(
        "grep",
        {"query": "main", "path_pattern": "mycode/*.py"},
    )

    assert result.ok is True
    assert result.content == "mycode/cli.py:1 | def main():"


def test_read_only_registry_filters_sensitive_files_from_broad_grep(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("grep", {"query": "needle", "path_pattern": "**/*"})

    assert result.ok is True
    assert result.content == "app.py:1 | needle"
    assert result.metadata["searched_files"] == 1


def test_read_only_registry_requires_confirmation_for_explicit_sensitive_grep(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("grep", {"query": "needle", "path_pattern": ".env"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["path_scope"] == "sensitive_path"


def test_read_only_registry_greps_explicit_sensitive_file_after_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [GrepTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("grep", {"query": "needle", "path_pattern": ".env"})

    assert result.ok is True
    assert result.content == ".env:1 | OPENAI_API_KEY=needle"


def test_read_only_registry_requires_confirmation_for_sensitive_grep_pattern(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool("grep", {"query": "needle", "path_pattern": "**/.env*"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["pattern_scope"] == "sensitive_pattern"


def test_read_only_registry_greps_env_example_pattern_without_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = create_read_only_tool_registry(Workspace(tmp_path))

    result = registry.run_tool(
        "grep",
        {"query": "needle", "path_pattern": "**/.env.example"},
    )

    assert result.ok is True
    assert result.content == ".env.example:1 | OPENAI_API_KEY=needle"


def test_read_only_registry_greps_sensitive_pattern_after_confirmation(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=needle\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [GrepTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("grep", {"query": "needle", "path_pattern": "**/.env*"})

    assert result.ok is True
    assert result.content == ".env.local:1 | OPENAI_API_KEY=needle"


def test_read_only_registry_denies_outside_workspace_grep_without_confirmation(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.from_tools(
        [GrepTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "grep",
        {"query": "needle", "path_pattern": "../outside.txt"},
    )

    assert result.ok is False
    assert result.error == "Path pattern must not contain '..': ../outside.txt"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["permission_reason"] == "outside_workspace"


class ApprovingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.approved()
