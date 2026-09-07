from dataclasses import dataclass
import os
from pathlib import Path
import re

from mycode.filesystem import (
    JsonSnapshotError,
    StorageBoundaryError,
    ensure_storage_directory,
    read_json_snapshot,
    write_json_snapshot,
    validate_storage_path,
)
from mycode.project import ProjectIdentity


PROJECT_METADATA_VERSION = 1
PROJECT_HASH_LENGTH = 12
MAX_PROJECT_BASENAME_CHARS = 80
PROJECT_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


class ProjectStorageError(RuntimeError):
    pass


class ProjectMetadataError(ProjectStorageError):
    pass


@dataclass(frozen=True)
class SessionStorageLayout:
    root: Path
    transcript_path: Path
    meta_path: Path
    compact_path: Path
    artifacts_directory: Path
    subagents_directory: Path
    lock_path: Path

    def subagent_log_path(self, run_id: str) -> Path:
        validate_storage_component(run_id, field_name="run_id")
        return self.subagents_directory / f"{run_id}.jsonl"


@dataclass(frozen=True)
class ProjectStorage:
    identity: ProjectIdentity
    projects_root: Path
    project_directory: Path
    metadata_path: Path
    sessions_directory: Path
    locks_directory: Path

    @classmethod
    def open(
        cls,
        identity: ProjectIdentity,
        *,
        projects_root: str | Path | None = None,
    ) -> "ProjectStorage":
        _validate_project_identity(identity)
        root = Path(
            default_projects_root() if projects_root is None else projects_root
        ).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        directory_name = project_directory_name(identity)
        try:
            project_directory = ensure_storage_directory(root, directory_name)
            metadata_path = project_directory / "project.json"
            sessions_directory = project_directory / "sessions"
            locks_directory = project_directory / "locks"
            storage = cls(
                identity=identity,
                projects_root=root,
                project_directory=project_directory,
                metadata_path=metadata_path,
                sessions_directory=sessions_directory,
                locks_directory=locks_directory,
            )
            storage._ensure_metadata()
            ensure_storage_directory(project_directory, sessions_directory)
            ensure_storage_directory(project_directory, locks_directory)
            return storage
        except (OSError, JsonSnapshotError, StorageBoundaryError) as error:
            raise ProjectStorageError(
                f"Could not open project storage for {identity.workspace_root}"
            ) from error

    def session(
        self,
        session_id: str,
        *,
        create: bool = False,
    ) -> SessionStorageLayout:
        validate_storage_component(session_id, field_name="session_id")
        session_root = self.sessions_directory / session_id
        try:
            # Check the project entry before using it as the narrower boundary.
            validate_storage_path(self.projects_root, self.project_directory)
            validate_storage_path(self.project_directory, session_root)
            validate_storage_path(self.project_directory, self.locks_directory)
            if create:
                session_root = ensure_storage_directory(
                    self.project_directory,
                    session_root,
                )
                artifacts = ensure_storage_directory(session_root, "artifacts")
                subagents = ensure_storage_directory(session_root, "subagents")
            else:
                if session_root.is_symlink():
                    raise StorageBoundaryError(
                        f"Session directory cannot be a symlink: {session_root}"
                    )
                resolved = session_root.resolve(strict=False)
                if not resolved.is_relative_to(self.sessions_directory):
                    raise StorageBoundaryError(
                        f"Session directory escapes project storage: {session_root}"
                    )
                artifacts = session_root / "artifacts"
                subagents = session_root / "subagents"
            for target in (artifacts, subagents):
                validate_storage_path(self.project_directory, target)
        except (OSError, StorageBoundaryError) as error:
            raise ProjectStorageError(
                f"Could not resolve session storage for {session_id!r}"
            ) from error
        return SessionStorageLayout(
            root=session_root,
            transcript_path=session_root / "transcript.jsonl",
            meta_path=session_root / "meta.json",
            compact_path=session_root / "compact.json",
            artifacts_directory=artifacts,
            subagents_directory=subagents,
            lock_path=self.locks_directory / f"{session_id}.lock",
        )

    def _ensure_metadata(self) -> None:
        expected = {
            "version": PROJECT_METADATA_VERSION,
            "project_key": self.identity.key,
            "workspace_root": str(self.identity.workspace_root),
        }
        if not self.metadata_path.exists():
            write_json_snapshot(
                self.project_directory,
                self.metadata_path,
                expected,
            )
            return
        actual = read_json_snapshot(self.project_directory, self.metadata_path)
        if actual != expected:
            raise ProjectMetadataError(
                "Project metadata does not match the canonical workspace: "
                f"{self.metadata_path}"
            )


def default_projects_root() -> Path:
    return Path.home() / ".mycode" / "projects"


def project_directory_name(identity: ProjectIdentity) -> str:
    _validate_project_identity(identity)
    basename = identity.workspace_root.name or "workspace"
    safe_basename = _safe_project_basename(basename)
    return f"{safe_basename}-{identity.key[:PROJECT_HASH_LENGTH]}"


def _safe_project_basename(value: str) -> str:
    cleaned = "".join(
        "-" if character in '<>:"/\\|?*' or ord(character) < 32 else character
        for character in value
    )
    cleaned = re.sub(r"-+", "-", cleaned).strip(" .-_")
    cleaned = cleaned[:MAX_PROJECT_BASENAME_CHARS].rstrip(" .") or "workspace"
    if cleaned.partition(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"workspace-{cleaned}"
    return cleaned


def _validate_project_identity(identity: ProjectIdentity) -> None:
    if len(identity.key) != 64 or any(
        character not in "0123456789abcdef" for character in identity.key
    ):
        raise ProjectStorageError("ProjectIdentity.key must be a lowercase SHA-256.")
    canonical = ProjectIdentity.from_workspace(identity.workspace_root)
    if canonical != identity:
        raise ProjectStorageError(
            "ProjectIdentity does not match its canonical workspace path."
        )


def validate_storage_component(value: str, *, field_name: str) -> None:
    """Validate a Session or SubAgent ID using the shared portable path rules."""
    if (
        not PROJECT_COMPONENT_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or value.rstrip(" .") != value
        or value.partition(".")[0].upper() in WINDOWS_RESERVED_NAMES
        or os.path.isabs(value)
    ):
        raise ProjectStorageError(f"Invalid {field_name}: {value!r}")
