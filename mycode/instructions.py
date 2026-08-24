from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal


NATIVE_INSTRUCTION_FILE = "MYCODE.md"
SHARED_INSTRUCTION_FILE = "AGENTS.md"
COMPATIBLE_INSTRUCTION_FILE = "CLAUDE.md"
INSTRUCTION_FILE_PRIORITY = (
    NATIVE_INSTRUCTION_FILE,
    SHARED_INSTRUCTION_FILE,
    COMPATIBLE_INSTRUCTION_FILE,
)
DEFAULT_MAX_INSTRUCTION_FILE_BYTES = 32 * 1024
DEFAULT_MAX_INSTRUCTION_CHARS = 64 * 1024

InstructionScope = Literal["user", "project", "directory"]
InstructionIssueCode = Literal[
    "outside_scope",
    "broken_symlink",
    "not_a_file",
    "file_too_large",
    "invalid_utf8",
    "total_too_large",
    "read_error",
]


class InstructionPathError(ValueError):
    pass


@dataclass(frozen=True)
class InstructionLimits:
    max_file_bytes: int = DEFAULT_MAX_INSTRUCTION_FILE_BYTES
    max_total_chars: int = DEFAULT_MAX_INSTRUCTION_CHARS

    def __post_init__(self) -> None:
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1.")
        if self.max_total_chars < 1:
            raise ValueError("max_total_chars must be at least 1.")


@dataclass(frozen=True)
class InstructionSource:
    path: Path
    scope: InstructionScope
    content: str
    content_bytes: int = 0
    sha256: str = ""

    @property
    def label(self) -> str:
        return f"{self.scope}: {self.path}"


@dataclass(frozen=True)
class InstructionIssue:
    path: Path
    code: InstructionIssueCode
    message: str

    @property
    def display(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


@dataclass(frozen=True)
class InstructionBundle:
    sources: tuple[InstructionSource, ...] = ()
    issues: tuple[InstructionIssue, ...] = ()

    @property
    def total_chars(self) -> int:
        return sum(len(source.content) for source in self.sources)

    def to_prompt_text(self) -> str:
        sections: list[str] = []
        for source in self.sources:
            sections.append(
                f"### {source.label}\n\n{source.content.rstrip()}"
            )
        return "\n\n".join(sections)


def default_user_instruction_directory() -> Path:
    return Path.home() / ".mycode"


def load_instruction_bundle(
    workspace_root: str | Path,
    *,
    working_directory: str | Path | None = None,
    user_instruction_directory: str | Path | None = None,
    limits: InstructionLimits | None = None,
) -> InstructionBundle:
    resolved_workspace = _require_directory(workspace_root, label="workspace_root")
    resolved_working_directory = _require_directory(
        resolved_workspace if working_directory is None else working_directory,
        label="working_directory",
    )
    if not resolved_working_directory.is_relative_to(resolved_workspace):
        raise InstructionPathError(
            "working_directory must be inside workspace_root: "
            f"{resolved_working_directory}"
        )

    resolved_user_directory = Path(
        default_user_instruction_directory()
        if user_instruction_directory is None
        else user_instruction_directory
    ).resolve(strict=False)
    effective_limits = InstructionLimits() if limits is None else limits

    candidates: list[tuple[Path, InstructionScope, Path]] = []
    user_candidate = _select_instruction_file(resolved_user_directory)
    if user_candidate is not None:
        candidates.append(
            (user_candidate, "user", resolved_user_directory.parent)
        )

    directories = _instruction_directories(
        resolved_workspace,
        resolved_working_directory,
    )
    for index, directory in enumerate(directories):
        candidate = _select_instruction_file(directory)
        if candidate is None:
            continue
        scope: InstructionScope = "project" if index == 0 else "directory"
        candidates.append((candidate, scope, resolved_workspace))

    sources: list[InstructionSource] = []
    issues: list[InstructionIssue] = []
    seen_paths: set[Path] = set()
    total_chars = 0

    for candidate, scope, allowed_root in candidates:
        source, issue = _read_instruction_source(
            candidate,
            scope=scope,
            allowed_root=allowed_root,
            limits=effective_limits,
        )
        if issue is not None:
            issues.append(issue)
            continue
        if source is None or source.path in seen_paths:
            continue

        next_total = total_chars + len(source.content)
        if next_total > effective_limits.max_total_chars:
            issues.append(
                InstructionIssue(
                    path=source.path,
                    code="total_too_large",
                    message=(
                        "instruction total would exceed "
                        f"{effective_limits.max_total_chars} characters"
                    ),
                )
            )
            continue

        seen_paths.add(source.path)
        sources.append(source)
        total_chars = next_total

    return InstructionBundle(
        sources=tuple(sources),
        issues=tuple(issues),
    )


def _require_directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=False)
    if not resolved.exists():
        raise InstructionPathError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise InstructionPathError(f"{label} is not a directory: {resolved}")
    return resolved


def _select_instruction_file(directory: Path) -> Path | None:
    for file_name in INSTRUCTION_FILE_PRIORITY:
        candidate = directory / file_name
        if candidate.exists() or candidate.is_symlink():
            return candidate

    return None


def _instruction_directories(workspace_root: Path, working_directory: Path) -> list[Path]:
    directories = [workspace_root]
    current = workspace_root
    relative = working_directory.relative_to(workspace_root)
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return directories


def _read_instruction_source(
    path: Path,
    *,
    scope: InstructionScope,
    allowed_root: Path,
    limits: InstructionLimits,
) -> tuple[InstructionSource | None, InstructionIssue | None]:
    if path.is_symlink() and not path.exists():
        return None, InstructionIssue(
            path=path,
            code="broken_symlink",
            message="instruction symlink target does not exist",
        )

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        return None, InstructionIssue(
            path=path,
            code="read_error",
            message=str(error),
        )

    if not resolved.is_relative_to(allowed_root.resolve(strict=False)):
        return None, InstructionIssue(
            path=path,
            code="outside_scope",
            message=f"resolved path is outside {allowed_root}",
        )

    if not resolved.is_file():
        return None, InstructionIssue(
            path=resolved,
            code="not_a_file",
            message="instruction path is not a regular file",
        )

    try:
        file_size = resolved.stat().st_size
        if file_size > limits.max_file_bytes:
            return None, InstructionIssue(
                path=resolved,
                code="file_too_large",
                message=(
                    f"file is {file_size} bytes; limit is "
                    f"{limits.max_file_bytes} bytes"
                ),
            )
        raw_content = resolved.read_bytes()
        if len(raw_content) > limits.max_file_bytes:
            return None, InstructionIssue(
                path=resolved,
                code="file_too_large",
                message=(
                    f"file is {len(raw_content)} bytes; limit is "
                    f"{limits.max_file_bytes} bytes"
                ),
            )
    except OSError as error:
        return None, InstructionIssue(
            path=resolved,
            code="read_error",
            message=str(error),
        )

    try:
        content = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, InstructionIssue(
            path=resolved,
            code="invalid_utf8",
            message="instruction file must be valid UTF-8",
        )

    return InstructionSource(
        path=resolved,
        scope=scope,
        content=content,
        content_bytes=len(raw_content),
        sha256=hashlib.sha256(raw_content).hexdigest(),
    ), None
