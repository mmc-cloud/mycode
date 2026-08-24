from pathlib import Path

from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    PermissionDecision,
)
from mycode.tools import ToolRegistry, Workspace, WriteFileArgs, WriteFileTool


def test_write_file_schema_describes_arguments(tmp_path: Path) -> None:
    schema = WriteFileTool(workspace=Workspace(tmp_path)).get_schema()

    assert schema["name"] == "write_file"
    assert schema["parameters"]["properties"]["path"]["type"] == "string"
    assert schema["parameters"]["properties"]["content"] == {"type": "string"}


def test_write_file_args_accept_empty_content() -> None:
    args = WriteFileArgs(path="empty.txt", content="")

    assert args.path == "empty.txt"
    assert args.content == ""


def test_write_file_permission_request_targets_path(tmp_path: Path) -> None:
    tool = WriteFileTool(workspace=Workspace(tmp_path))

    request = tool.build_permission_request(
        WriteFileArgs(path="notes.txt", content="hello")
    )

    assert request.tool_name == "write_file"
    assert request.capability == "write"
    assert request.action == "write_file"
    assert request.target == "notes.txt"
    assert request.arguments == {"path": "notes.txt", "content": "hello"}


def test_write_file_direct_run_requires_registry(tmp_path: Path) -> None:
    tool = WriteFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "notes.txt", "content": "hello"})

    assert result.ok is False
    assert result.error == "write_file must be run through ToolRegistry.run_tool()."
    assert not (tmp_path / "notes.txt").exists()


def test_write_file_requires_confirmation_by_default(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([WriteFileTool(Workspace(tmp_path))])

    result = registry.run_tool("write_file", {"path": "notes.txt", "content": "hello"})

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "requires_confirmation"
    assert result.metadata["operation"] == "create"
    assert result.metadata["path_scope"] == "inside_workspace"
    assert not (tmp_path / "notes.txt").exists()


def test_write_file_creates_file_after_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("write_file", {"path": "notes.txt", "content": "hello"})

    assert result.ok is True
    assert result.content == "Wrote file: notes.txt"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert result.metadata["path"] == "notes.txt"
    assert result.metadata["operation"] == "create"
    assert result.metadata["encoding"] == "utf-8"
    assert result.metadata["content_chars"] == 5
    assert result.metadata["content_bytes"] == 5
    assert result.metadata["target_existed_before"] is False
    assert result.metadata["confirmation_status"] == "approved"


def test_write_file_overwrites_file_after_confirmation(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("old", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("write_file", {"path": "notes.txt", "content": "new"})

    assert result.ok is True
    assert file_path.read_text(encoding="utf-8") == "new"
    assert result.metadata["operation"] == "overwrite"
    assert result.metadata["target_existed_before"] is True


def test_write_file_rejects_confirmation_decline(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=RejectingConfirmer(),
    )

    result = registry.run_tool("write_file", {"path": "notes.txt", "content": "hello"})

    assert result.ok is False
    assert result.error == "rejected by test"
    assert result.metadata["confirmation_status"] == "rejected"
    assert not (tmp_path / "notes.txt").exists()


def test_write_file_requires_confirmation_for_sensitive_path(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools([WriteFileTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "write_file",
        {"path": ".env", "content": "OPENAI_API_KEY=fake"},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["path_scope"] == "sensitive_path"
    assert not (tmp_path / ".env").exists()


def test_write_file_writes_sensitive_path_after_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "write_file",
        {"path": ".env", "content": "OPENAI_API_KEY=fake"},
    )

    assert result.ok is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=fake"
    assert result.metadata["path_scope"] == "sensitive_path"
    assert result.metadata["confirmation_status"] == "approved"


def test_write_file_requires_confirmation_for_ignored_path(tmp_path: Path) -> None:
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    registry = ToolRegistry.from_tools([WriteFileTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "write_file",
        {"path": ".venv/Lib/site-packages/pkg.py", "content": "value = 1"},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_reason"] == "ignored_path"
    assert result.metadata["path_scope"] == "ignored_path"


def test_write_file_writes_outside_workspace_after_confirmation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_path = tmp_path / "outside.txt"
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(workspace_root))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "write_file",
        {"path": "../outside.txt", "content": "outside"},
    )

    assert result.ok is True
    assert outside_path.read_text(encoding="utf-8") == "outside"
    assert result.metadata["path"] == outside_path.as_posix()
    assert result.metadata["path_scope"] == "outside_workspace"


def test_write_file_rejects_directory_target(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool("write_file", {"path": "src", "content": "hello"})

    assert result.ok is False
    assert result.error == "Path is not a file: src"


def test_write_file_directory_failure_preserves_decision_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    tool = WriteFileTool(Workspace(tmp_path))
    decision = PermissionDecision.allow(
        metadata={
            "path_scope": "inside_workspace",
            "resolved_path": (tmp_path / "src").as_posix(),
            "confirmation_status": "approved",
        }
    )

    result = tool._write_path(
        WriteFileArgs(path="src", content="hello"),
        tmp_path / "src",
        decision,
    )

    assert result.ok is False
    assert result.error == "Path is not a file: src"
    assert result.metadata["path_scope"] == "inside_workspace"
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["resolved_path"] == (tmp_path / "src").as_posix()
    assert result.metadata["permission_status"] == "allow"
    assert result.metadata["target_is_file"] is False


def test_write_file_rejects_missing_parent_directory(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [WriteFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "write_file",
        {"path": "missing/notes.txt", "content": "hello"},
    )

    assert result.ok is False
    assert result.error == "Parent directory does not exist: missing/notes.txt"
    assert result.metadata["path_scope"] == "inside_workspace"
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["permission_status"] == "allow"
    assert result.metadata["parent"] == (tmp_path / "missing").as_posix()
    assert not (tmp_path / "missing" / "notes.txt").exists()


class ApprovingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.approved()


class RejectingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.rejected("rejected by test")
