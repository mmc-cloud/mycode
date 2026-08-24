from pathlib import Path

from mycode.memory import MemoryStore
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.session_store import ProjectIdentity
from mycode.tools import ToolRegistry
from mycode.tools.memory import (
    DeleteMemoryTool,
    ListMemoriesTool,
    SaveMemoryTool,
)


def memory_store(tmp_path: Path) -> MemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return MemoryStore(
        ProjectIdentity.from_workspace(workspace),
        base_directory=tmp_path / "user-state",
    )


def test_save_memory_requires_confirmation(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    registry = ToolRegistry.from_tools([SaveMemoryTool(store)])

    result = registry.run_tool(
        "save_memory",
        {
            "scope": "user",
            "kind": "preference",
            "key": "response.language",
            "content": "中文",
        },
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert not store.path_for_scope("user").exists()


def test_approved_save_list_update_and_delete_memory(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    confirmer = RecordingApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [
            ListMemoriesTool(store),
            SaveMemoryTool(store),
            DeleteMemoryTool(store),
        ],
        confirmer=confirmer,
    )

    created = registry.run_tool(
        "save_memory",
        {
            "scope": "project",
            "kind": "fact",
            "key": "test.command",
            "content": "pytest",
        },
    )
    updated = registry.run_tool(
        "save_memory",
        {
            "scope": "project",
            "kind": "fact",
            "key": "test.command",
            "content": "uv run pytest",
        },
    )
    listed = registry.run_tool("list_memories", {"scope": "project"})
    deleted = registry.run_tool(
        "delete_memory",
        {"scope": "project", "key": "test.command"},
    )

    assert created.ok is True
    assert "created" in created.content
    assert updated.ok is True
    assert "updated" in updated.content
    assert listed.ok is True
    assert "uv run pytest" in listed.content
    assert deleted.ok is True
    assert store.list_entries("project") == ()
    save_request = confirmer.requests[0]
    assert save_request.metadata["memory_scope"] == "project"
    assert save_request.metadata["memory_kind"] == "fact"
    assert save_request.metadata["memory_key"] == "test.command"
    assert save_request.metadata["memory_content"] == "pytest"
    assert "memory_content" not in created.metadata


def test_save_memory_rejects_sensitive_content_before_confirmation(
    tmp_path: Path,
) -> None:
    store = memory_store(tmp_path)
    confirmer = RecordingApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [SaveMemoryTool(store)],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "save_memory",
        {
            "scope": "user",
            "kind": "fact",
            "key": "provider.key",
            "content": "api_key=super-secret-value",
        },
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"
    assert confirmer.requests == []


def test_direct_write_tool_run_cannot_bypass_registry(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    tool = SaveMemoryTool(store)

    result = tool.run(
        {
            "scope": "user",
            "kind": "preference",
            "key": "response.language",
            "content": "中文",
        }
    )

    assert result.ok is False
    assert result.error == "save_memory must be run through ToolRegistry.run_tool()."


class RecordingApprovingConfirmer:
    def __init__(self) -> None:
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return ConfirmationResult.approved()
