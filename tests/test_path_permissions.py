from pathlib import Path

from mycode.permissions import PermissionRequest
from mycode.tools import PathPermissionPolicy, Workspace


def test_path_permission_policy_allows_normal_workspace_path(tmp_path: Path) -> None:
    policy = PathPermissionPolicy(Workspace(tmp_path))
    request = read_request("README.md")

    decision = policy.check_path(request, "README.md")

    assert decision.status == "allow"
    assert decision.reason == "allowed"
    assert decision.message == "Path inside workspace allowed: README.md"
    assert decision.metadata["path_scope"] == "inside_workspace"
    assert decision.metadata["path"] == "README.md"


def test_path_permission_policy_asks_for_relative_path_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    policy = PathPermissionPolicy(Workspace(workspace_root))
    request = read_request("../outside.txt")

    decision = policy.check_path(request, "../outside.txt")

    assert decision.status == "ask"
    assert decision.reason == "outside_workspace"
    assert decision.message == "Path outside workspace requires confirmation: ../outside.txt"
    assert decision.metadata["path_scope"] == "outside_workspace"


def test_path_permission_policy_asks_for_absolute_path_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_path = tmp_path / "outside.txt"
    policy = PathPermissionPolicy(Workspace(workspace_root))
    request = read_request(str(outside_path))

    decision = policy.check_path(request, outside_path)

    assert decision.status == "ask"
    assert decision.reason == "outside_workspace"
    assert decision.metadata["path_scope"] == "outside_workspace"
    assert decision.metadata["resolved_path"] == outside_path.resolve().as_posix()


def test_path_permission_policy_asks_for_sensitive_path(tmp_path: Path) -> None:
    policy = PathPermissionPolicy(Workspace(tmp_path))
    request = read_request(".env")

    decision = policy.check_path(request, ".env")

    assert decision.status == "ask"
    assert decision.reason == "sensitive_path"
    assert decision.message == "Sensitive path requires confirmation: .env"
    assert decision.metadata["path_scope"] == "sensitive_path"


def test_path_permission_policy_asks_for_sensitive_path_before_ignored_path(
    tmp_path: Path,
) -> None:
    policy = PathPermissionPolicy(Workspace(tmp_path))
    request = read_request(".env.local")

    decision = policy.check_path(request, ".env.local")

    assert decision.status == "ask"
    assert decision.reason == "sensitive_path"
    assert decision.metadata["path_scope"] == "sensitive_path"


def test_path_permission_policy_asks_for_ignored_path(tmp_path: Path) -> None:
    policy = PathPermissionPolicy(Workspace(tmp_path))
    request = read_request(".venv/Lib/site-packages/pkg.py")

    decision = policy.check_path(request, ".venv/Lib/site-packages/pkg.py")

    assert decision.status == "ask"
    assert decision.reason == "ignored_path"
    assert decision.message == (
        "Ignored path requires confirmation: .venv/Lib/site-packages/pkg.py"
    )
    assert decision.metadata["path_scope"] == "ignored_path"


def test_path_permission_policy_metadata_includes_request_context(
    tmp_path: Path,
) -> None:
    policy = PathPermissionPolicy(Workspace(tmp_path))
    request = PermissionRequest(
        tool_name="write_file",
        capability="write",
        action="write_file",
        target="notes.txt",
        arguments={"path": "notes.txt"},
    )

    decision = policy.check_path(request, "notes.txt")

    assert decision.metadata["tool_name"] == "write_file"
    assert decision.metadata["capability"] == "write"
    assert decision.metadata["action"] == "write_file"
    assert decision.metadata["target"] == "notes.txt"
    assert decision.metadata["workspace_root"] == tmp_path.resolve().as_posix()


def read_request(target: str) -> PermissionRequest:
    return PermissionRequest(
        tool_name="read_file",
        capability="read",
        action="read_file",
        target=target,
        arguments={"path": target},
    )
