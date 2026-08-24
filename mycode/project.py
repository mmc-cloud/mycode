from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectIdentity:
    key: str
    workspace_root: Path

    @classmethod
    def from_workspace(cls, workspace_root: str | Path) -> "ProjectIdentity":
        resolved = Path(workspace_root).resolve(strict=False)
        if not resolved.exists():
            raise ValueError(f"Workspace root does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Workspace root is not a directory: {resolved}")

        normalized = os.path.normcase(str(resolved)).replace("\\", "/")
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(key=key, workspace_root=resolved)
