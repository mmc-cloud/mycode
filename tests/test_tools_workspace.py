from pathlib import Path

import pytest

from mycode.tools import Workspace, WorkspacePathError


def test_workspace_root_is_resolved_to_absolute_path(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)

    assert workspace.root == tmp_path.resolve()
    assert workspace.root.is_absolute()


def test_workspace_rejects_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(WorkspacePathError, match="Workspace root does not exist"):
        Workspace(root=missing_root)


def test_workspace_rejects_file_root(tmp_path: Path) -> None:
    file_root = tmp_path / "file.txt"
    file_root.write_text("content", encoding="utf-8")

    with pytest.raises(WorkspacePathError, match="Workspace root is not a directory"):
        Workspace(root=file_root)


def test_resolve_path_allows_relative_path_inside_workspace(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)

    resolved_path = workspace.resolve_path("src/main.py")

    assert resolved_path == (tmp_path / "src" / "main.py").resolve()


def test_resolve_path_allows_absolute_path_inside_workspace(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)
    inside_path = tmp_path / "src" / "main.py"

    resolved_path = workspace.resolve_path(inside_path)

    assert resolved_path == inside_path.resolve()


def test_resolve_path_allows_workspace_root(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)

    resolved_path = workspace.resolve_path(".")

    assert resolved_path == tmp_path.resolve()


def test_resolve_path_rejects_relative_path_outside_workspace(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)

    with pytest.raises(WorkspacePathError, match="Path is outside workspace"):
        workspace.resolve_path("../outside.txt")


def test_resolve_path_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace(root=workspace_root)
    outside_path = tmp_path / "outside.txt"

    with pytest.raises(WorkspacePathError, match="Path is outside workspace"):
        workspace.resolve_path(outside_path)
