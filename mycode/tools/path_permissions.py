from dataclasses import dataclass
from pathlib import Path

from mycode.permissions import PermissionDecision, PermissionRequest
from mycode.tools.ignore import is_ignored_path, is_sensitive_path
from mycode.tools.workspace import Workspace, WorkspacePathError


@dataclass(frozen=True)
class PathPermissionPolicy:
    workspace: Workspace

    def check_path(
        self,
        request: PermissionRequest,
        path: str | Path,
    ) -> PermissionDecision:
        resolved_path = _resolve_candidate_path(self.workspace, path)
        metadata = _build_metadata(
            request=request,
            path=path,
            resolved_path=resolved_path,
            workspace_root=self.workspace.root,
        )

        try:
            self.workspace.resolve_path(path)
        except WorkspacePathError:
            return PermissionDecision.ask(
                reason="outside_workspace",
                message=f"Path outside workspace requires confirmation: {path}",
                metadata={**metadata, "path_scope": "outside_workspace"},
            )

        if is_sensitive_path(resolved_path, self.workspace.root):
            return PermissionDecision.ask(
                reason="sensitive_path",
                message=f"Sensitive path requires confirmation: {path}",
                metadata={**metadata, "path_scope": "sensitive_path"},
            )

        if is_ignored_path(resolved_path, self.workspace.root):
            return PermissionDecision.ask(
                reason="ignored_path",
                message=f"Ignored path requires confirmation: {path}",
                metadata={**metadata, "path_scope": "ignored_path"},
            )

        return PermissionDecision.allow(
            message=f"Path inside workspace allowed: {path}",
            metadata={**metadata, "path_scope": "inside_workspace"},
        )


def _resolve_candidate_path(workspace: Workspace, path: str | Path) -> Path:
    requested_path = Path(path)
    if requested_path.is_absolute():
        return requested_path.resolve(strict=False)

    return (workspace.root / requested_path).resolve(strict=False)


def _build_metadata(
    *,
    request: PermissionRequest,
    path: str | Path,
    resolved_path: Path,
    workspace_root: Path,
) -> dict[str, object]:
    return {
        "tool_name": request.tool_name,
        "capability": request.capability,
        "action": request.action,
        "target": request.target,
        "path": str(path),
        "resolved_path": resolved_path.as_posix(),
        "workspace_root": workspace_root.as_posix(),
    }
