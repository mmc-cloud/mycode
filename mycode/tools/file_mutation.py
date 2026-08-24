from pathlib import Path

from mycode.permissions import PermissionDecision
from mycode.tools.permission_metadata import with_permission_metadata


def resolved_path_from_decision(decision: PermissionDecision) -> Path:
    resolved_path = decision.metadata.get("resolved_path")
    if isinstance(resolved_path, str):
        return Path(resolved_path)

    raise ValueError("Path permission decision did not include resolved_path.")


def display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def failure_metadata(
    decision: PermissionDecision,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **decision.metadata,
        **metadata,
        "permission_status": decision.status,
    }
