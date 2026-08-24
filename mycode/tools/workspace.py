from dataclasses import dataclass
from pathlib import Path


class WorkspacePathError(ValueError):
    pass


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        resolved_root = self.root.resolve(strict=False)
        if not resolved_root.exists():
            raise WorkspacePathError(f"Workspace root does not exist: {self.root}")
        if not resolved_root.is_dir():
            raise WorkspacePathError(f"Workspace root is not a directory: {self.root}")

        object.__setattr__(self, "root", resolved_root)

    def resolve_path(self, path: str | Path) -> Path:
        requested_path = Path(path)

        if requested_path.is_absolute():
            resolved_path = requested_path.resolve(strict=False)
        else:
            resolved_path = (self.root / requested_path).resolve(strict=False)

        if not resolved_path.is_relative_to(self.root):
            raise WorkspacePathError(f"Path is outside workspace: {path}")

        return resolved_path
