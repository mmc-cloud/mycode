from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

from mycode.project import ProjectIdentity


MemoryScope = Literal["user", "project"]
MemoryKind = Literal["preference", "fact", "experience"]
MemoryIssueCode = Literal[
    "file_too_large",
    "invalid_utf8",
    "read_error",
    "malformed_marker",
    "missing_end_marker",
    "invalid_heading",
    "empty_content",
    "duplicate_key",
    "sensitive_content",
]
MemoryWriteAction = Literal["created", "updated"]

MEMORY_KINDS = frozenset({"preference", "fact", "experience"})
MEMORY_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
ENTRY_START_PATTERN = re.compile(
    r"^<!-- mycode-memory:entry "
    r"kind=(preference|fact|experience) "
    r"key=([a-z0-9][a-z0-9._-]{0,79}) -->[ \t]*\r?\n",
    re.MULTILINE,
)
ENTRY_MARKER_PREFIX_PATTERN = re.compile(
    r"^<!-- mycode-memory:entry\b",
    re.MULTILINE,
)
ENTRY_END_PATTERN = re.compile(
    r"^<!-- mycode-memory:end -->[ \t]*(?:\r?\n|$)",
    re.MULTILINE,
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:[a-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password)"
    r"\s*[:=]\s*[^\s]{6,}"
)
SENSITIVE_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")

DEFAULT_MAX_MEMORY_FILE_BYTES = 128 * 1024
DEFAULT_MAX_MEMORY_ENTRY_CHARS = 2000
DEFAULT_MAX_MEMORY_ENTRIES = 100


class MemoryError(RuntimeError):
    pass


class MemoryFormatError(MemoryError):
    pass


class SensitiveMemoryError(MemoryError):
    pass


@dataclass(frozen=True)
class MemoryLimits:
    max_file_bytes: int = DEFAULT_MAX_MEMORY_FILE_BYTES
    max_entry_chars: int = DEFAULT_MAX_MEMORY_ENTRY_CHARS
    max_entries: int = DEFAULT_MAX_MEMORY_ENTRIES

    def __post_init__(self) -> None:
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1.")
        if self.max_entry_chars < 1:
            raise ValueError("max_entry_chars must be at least 1.")
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least 1.")


@dataclass(frozen=True)
class MemoryEntry:
    scope: MemoryScope
    kind: MemoryKind
    key: str
    content: str


@dataclass(frozen=True)
class MemoryIssue:
    path: Path
    code: MemoryIssueCode
    message: str
    key: str | None = None

    @property
    def blocking(self) -> bool:
        return self.code != "sensitive_content"

    @property
    def display(self) -> str:
        key_text = "" if self.key is None else f" [{self.key}]"
        return f"{self.code}{key_text}: {self.path}: {self.message}"


@dataclass(frozen=True)
class _MemoryBlock:
    entry: MemoryEntry
    start: int
    end: int


@dataclass(frozen=True)
class MemoryDocument:
    path: Path
    scope: MemoryScope
    entries: tuple[MemoryEntry, ...] = ()
    issues: tuple[MemoryIssue, ...] = ()
    raw_text: str = ""
    content_bytes: int = 0
    sha256: str = ""
    _blocks: tuple[_MemoryBlock, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class MemoryWriteResult:
    action: MemoryWriteAction
    entry: MemoryEntry
    path: Path


class MemoryStore:
    def __init__(
        self,
        project: ProjectIdentity,
        *,
        base_directory: str | Path | None = None,
        limits: MemoryLimits | None = None,
    ) -> None:
        self.project = project
        self.base_directory = Path(
            Path.home() / ".mycode"
            if base_directory is None
            else base_directory
        ).resolve(strict=False)
        self.limits = MemoryLimits() if limits is None else limits

    def path_for_scope(self, scope: MemoryScope) -> Path:
        _validate_scope(scope)
        if scope == "user":
            return self.base_directory / "MEMORY.md"
        return (
            self.base_directory
            / "projects"
            / self.project.key
            / "MEMORY.md"
        )

    def read_document(self, scope: MemoryScope) -> MemoryDocument:
        path = self.path_for_scope(scope)
        if not path.exists():
            return MemoryDocument(path=path, scope=scope)
        if not path.is_file():
            return MemoryDocument(
                path=path,
                scope=scope,
                issues=(
                    MemoryIssue(
                        path=path,
                        code="read_error",
                        message="memory path is not a regular file",
                    ),
                ),
            )

        try:
            raw = path.read_bytes()
        except OSError as error:
            return MemoryDocument(
                path=path,
                scope=scope,
                issues=(
                    MemoryIssue(
                        path=path,
                        code="read_error",
                        message=str(error),
                    ),
                ),
            )
        if len(raw) > self.limits.max_file_bytes:
            return MemoryDocument(
                path=path,
                scope=scope,
                content_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                issues=(
                    MemoryIssue(
                        path=path,
                        code="file_too_large",
                        message=(
                            f"file is {len(raw)} bytes; limit is "
                            f"{self.limits.max_file_bytes} bytes"
                        ),
                    ),
                ),
            )
        try:
            raw_text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return MemoryDocument(
                path=path,
                scope=scope,
                content_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                issues=(
                    MemoryIssue(
                        path=path,
                        code="invalid_utf8",
                        message="memory file must be valid UTF-8",
                    ),
                ),
            )

        return replace(
            _parse_memory_document(path, scope, raw_text),
            content_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def list_entries(
        self,
        scope: MemoryScope | None = None,
    ) -> tuple[MemoryEntry, ...]:
        scopes: tuple[MemoryScope, ...] = (
            ("user", "project") if scope is None else (scope,)
        )
        entries: list[MemoryEntry] = []
        for selected_scope in scopes:
            entries.extend(self.read_document(selected_scope).entries)
        return tuple(entries)

    def list_issues(
        self,
        scope: MemoryScope | None = None,
    ) -> tuple[MemoryIssue, ...]:
        scopes: tuple[MemoryScope, ...] = (
            ("user", "project") if scope is None else (scope,)
        )
        issues: list[MemoryIssue] = []
        for selected_scope in scopes:
            issues.extend(self.read_document(selected_scope).issues)
        return tuple(issues)

    def save(
        self,
        *,
        scope: MemoryScope,
        kind: MemoryKind,
        key: str,
        content: str,
    ) -> MemoryWriteResult:
        normalized_scope = _validate_scope(scope)
        normalized_kind = _validate_kind(kind)
        normalized_key = validate_memory_key(key)
        normalized_content = validate_memory_content(
            content,
            max_chars=self.limits.max_entry_chars,
        )
        document = self.read_document(normalized_scope)
        _raise_for_blocking_issues(document)

        existing = next(
            (block for block in document._blocks if block.entry.key == normalized_key),
            None,
        )
        if existing is None and len(document._blocks) >= self.limits.max_entries:
            raise MemoryError(
                f"Memory file already contains {self.limits.max_entries} entries."
            )

        entry = MemoryEntry(
            scope=normalized_scope,
            kind=normalized_kind,
            key=normalized_key,
            content=normalized_content,
        )
        newline = "\r\n" if "\r\n" in document.raw_text else "\n"
        rendered_block = _render_memory_block(entry, newline=newline)
        if existing is None:
            base = document.raw_text
            if base.strip() == "":
                base = _memory_file_header(normalized_scope, newline=newline)
            updated = base.rstrip("\r\n") + newline * 2 + rendered_block + newline
            action: MemoryWriteAction = "created"
        else:
            updated = (
                document.raw_text[: existing.start]
                + rendered_block
                + newline
                + document.raw_text[existing.end :]
            )
            action = "updated"

        encoded = updated.encode("utf-8")
        if len(encoded) > self.limits.max_file_bytes:
            raise MemoryError(
                "Memory update would exceed file limit: "
                f"{len(encoded)}/{self.limits.max_file_bytes} bytes."
            )
        path = self.path_for_scope(normalized_scope)
        _atomic_write(path, encoded)

        verified = self.read_document(normalized_scope)
        verified_entry = next(
            (item for item in verified.entries if item.key == normalized_key),
            None,
        )
        if verified_entry != entry:
            raise MemoryFormatError(
                f"Memory write verification failed for key: {normalized_key}"
            )
        return MemoryWriteResult(action=action, entry=entry, path=path)

    def delete(self, *, scope: MemoryScope, key: str) -> bool:
        normalized_scope = _validate_scope(scope)
        normalized_key = validate_memory_key(key)
        document = self.read_document(normalized_scope)
        _raise_for_blocking_issues(document)
        block = next(
            (item for item in document._blocks if item.entry.key == normalized_key),
            None,
        )
        if block is None:
            return False

        updated = document.raw_text[: block.start] + document.raw_text[block.end :]
        _atomic_write(self.path_for_scope(normalized_scope), updated.encode("utf-8"))
        verified = self.read_document(normalized_scope)
        if any(item.entry.key == normalized_key for item in verified._blocks):
            raise MemoryFormatError(
                f"Memory delete verification failed for key: {normalized_key}"
            )
        return True


def validate_memory_key(key: str) -> str:
    normalized = key.strip().casefold()
    if not MEMORY_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Memory key must use 1-80 lowercase letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit."
        )
    return normalized


def validate_memory_content(content: str, *, max_chars: int) -> str:
    normalized = content.strip()
    if normalized == "":
        raise ValueError("Memory content must not be empty.")
    if len(normalized) > max_chars:
        raise ValueError(
            f"Memory content must not exceed {max_chars} characters."
        )
    if contains_sensitive_memory_content(normalized):
        raise SensitiveMemoryError(
            "Memory content appears to contain a secret, token, password or private key."
        )
    return normalized


def contains_sensitive_memory_content(content: str) -> bool:
    return bool(
        "-----BEGIN PRIVATE KEY-----" in content
        or "-----BEGIN RSA PRIVATE KEY-----" in content
        or SENSITIVE_ASSIGNMENT_PATTERN.search(content)
        or SENSITIVE_TOKEN_PATTERN.search(content)
    )


def _parse_memory_document(
    path: Path,
    scope: MemoryScope,
    raw_text: str,
) -> MemoryDocument:
    starts = list(ENTRY_START_PATTERN.finditer(raw_text))
    issues: list[MemoryIssue] = []
    blocks: list[_MemoryBlock] = []
    safe_entries: list[MemoryEntry] = []
    seen_keys: set[str] = set()

    if len(starts) != len(ENTRY_MARKER_PREFIX_PATTERN.findall(raw_text)):
        issues.append(
            MemoryIssue(
                path=path,
                code="malformed_marker",
                message="one or more memory entry markers are malformed",
            )
        )

    consumed_until = 0
    for index, start_match in enumerate(starts):
        if start_match.start() < consumed_until:
            continue
        next_start = starts[index + 1].start() if index + 1 < len(starts) else None
        end_match = ENTRY_END_PATTERN.search(raw_text, start_match.end())
        kind = start_match.group(1)
        key = start_match.group(2)
        if end_match is None or (
            next_start is not None and next_start < end_match.start()
        ):
            issues.append(
                MemoryIssue(
                    path=path,
                    code="missing_end_marker",
                    key=key,
                    message="memory entry is missing its end marker",
                )
            )
            continue

        consumed_until = end_match.end()
        payload = raw_text[start_match.end() : end_match.start()]
        lines = payload.splitlines()
        expected_heading = f"### {key}"
        heading_valid = bool(lines and lines[0].strip() == expected_heading)
        content = "\n".join(lines[1:]).strip() if lines else ""
        entry = MemoryEntry(
            scope=scope,
            kind=kind,  # type: ignore[arg-type]
            key=key,
            content=content,
        )
        block = _MemoryBlock(
            entry=entry,
            start=start_match.start(),
            end=end_match.end(),
        )
        blocks.append(block)

        if not heading_valid:
            issues.append(
                MemoryIssue(
                    path=path,
                    code="invalid_heading",
                    key=key,
                    message=f"expected heading: {expected_heading}",
                )
            )
            continue
        if content == "":
            issues.append(
                MemoryIssue(
                    path=path,
                    code="empty_content",
                    key=key,
                    message="memory entry content is empty",
                )
            )
            continue
        if key in seen_keys:
            issues.append(
                MemoryIssue(
                    path=path,
                    code="duplicate_key",
                    key=key,
                    message="memory key appears more than once",
                )
            )
            continue
        seen_keys.add(key)
        if contains_sensitive_memory_content(content):
            issues.append(
                MemoryIssue(
                    path=path,
                    code="sensitive_content",
                    key=key,
                    message="entry was withheld because it may contain sensitive content",
                )
            )
            continue
        safe_entries.append(entry)

    if len(ENTRY_END_PATTERN.findall(raw_text)) != len(blocks):
        issues.append(
            MemoryIssue(
                path=path,
                code="malformed_marker",
                message="memory end marker count does not match parsed entries",
            )
        )

    return MemoryDocument(
        path=path,
        scope=scope,
        entries=tuple(safe_entries),
        issues=tuple(issues),
        raw_text=raw_text,
        _blocks=tuple(blocks),
    )


def _raise_for_blocking_issues(document: MemoryDocument) -> None:
    blocking = [issue for issue in document.issues if issue.blocking]
    if blocking:
        details = "; ".join(issue.display for issue in blocking)
        raise MemoryFormatError(
            "Memory file contains structural errors; refusing automatic edit: "
            + details
        )


def _validate_scope(scope: MemoryScope) -> MemoryScope:
    if scope not in {"user", "project"}:
        raise ValueError(f"Unsupported memory scope: {scope}")
    return scope


def _validate_kind(kind: MemoryKind) -> MemoryKind:
    if kind not in MEMORY_KINDS:
        raise ValueError(f"Unsupported memory kind: {kind}")
    return kind


def _render_memory_block(entry: MemoryEntry, *, newline: str) -> str:
    return newline.join(
        [
            f"<!-- mycode-memory:entry kind={entry.kind} key={entry.key} -->",
            f"### {entry.key}",
            "",
            entry.content,
            "<!-- mycode-memory:end -->",
        ]
    )


def _memory_file_header(scope: MemoryScope, *, newline: str) -> str:
    title = "User" if scope == "user" else "Project"
    return newline.join(
        [
            f"# MyCode {title} Memory",
            "",
            "<!-- mycode-memory-format: 1 -->",
            "",
            "This file is the editable source of truth for long-term memory.",
            "Edit an entry body to correct it, or delete its complete marker block to forget it.",
        ]
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
