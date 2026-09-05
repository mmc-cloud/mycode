from collections.abc import Callable
import codecs
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from pydantic import Field, field_validator

from mycode.tool_result_format import (
    TOOL_RESULT_METADATA_MARKER,
    parse_tool_result_content,
    safe_tool_metadata,
)
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.session_lock import SessionLockTimeoutError
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult
from mycode.tools.bounds import clamp_positive_int_upper_bound


EXTERNALIZED_TOOL_RESULT_MARKER = "[tool result externalized]"
ARTIFACT_EXTERNALIZATION_FAILURE_MARKER = (
    "[tool result unavailable: artifact externalization failed]"
)
DEFAULT_ARTIFACT_READ_CHARS = 4000
MAX_ARTIFACT_READ_CHARS = 8000
ARTIFACT_IO_CHUNK_BYTES = 64 * 1024
ARTIFACT_IO_CHUNK_CHARS = 64 * 1024
MAX_READABLE_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_REFERENCE_METADATA_CHARS = 2400
DELETION_TRASH_DIRECTORY_NAME = "deletion-trash"

ArtifactWriteGuard = Callable[[], AbstractContextManager[None]]
ArtifactExternalizationFailureHandler = Callable[
    [str, str, str, Exception],
    str,
]


class ArtifactCleanupError(RuntimeError):
    pass


class _ArtifactTooLargeError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ArtifactTextScan:
    digest: str
    total_bytes: int
    total_chars: int
    selected_content: str
    end_offset_chars: int


@dataclass(frozen=True)
class ToolResultArtifactStore:
    """Content-addressed storage for large, model-visible tool results."""

    root: Path
    threshold_chars: int
    write_guard: ArtifactWriteGuard | None = None

    def __post_init__(self) -> None:
        if self.threshold_chars < 1:
            raise ValueError("threshold_chars must be at least 1.")
        object.__setattr__(self, "root", self.root.resolve(strict=False))

    def externalize(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        content: str,
    ) -> str:
        if len(content) <= self.threshold_chars:
            return content
        if _is_valid_artifact_reference(
            self.root,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content=content,
        ) or _is_valid_artifact_failure_reference(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content=content,
        ):
            return content

        digest = _sha256_text(content)
        artifact_path = self.root / f"{digest}.txt"
        guard = nullcontext() if self.write_guard is None else self.write_guard()
        with guard:
            self._write_once(artifact_path, content, digest=digest)

        parsed = parse_tool_result_content(content)
        reference_metadata: dict[str, object] = {
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": digest,
            "context_externalized": True,
            "original_chars": len(content),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }
        if parsed.result_preview:
            preview_key = (
                "error_preview" if parsed.status == "ERROR" else "result_preview"
            )
            reference_metadata[preview_key] = parsed.result_preview
        reference_metadata.update(
            _bounded_optional_metadata(
                reference_metadata,
                safe_tool_metadata(parsed.metadata),
            )
        )

        return (
            f"{parsed.status}\n"
            f"{EXTERNALIZED_TOOL_RESULT_MARKER}\n"
            f"tool_name: {tool_name}\n"
            f"artifact_path: {artifact_path.as_posix()}\n"
            f"original_chars: {len(content)}\n"
            f"sha256: {digest}\n"
            f"{TOOL_RESULT_METADATA_MARKER}"
            f"{json.dumps(reference_metadata, ensure_ascii=False, sort_keys=True)}"
        )

    def rehydrate(self, *, tool_name: str, tool_call_id: str, content: str) -> str:
        """Read a validated reference in this store; never trust a model-supplied path."""
        info = artifact_reference_info(
            tool_name=tool_name, tool_call_id=tool_call_id, content=content,
        )
        if info is None:
            raise ValueError("Invalid artifact reference.")
        path, digest, original_chars = info
        if not path.is_absolute() or path.parent != self.root or path.name != f"{digest}.txt":
            raise ValueError("Artifact reference is outside this store.")
        # Check the lexical path before resolving, including junctions in parents.
        if any(_is_link_or_reparse_point(part) for part in (path, *path.parents)):
            raise ValueError("Artifact reference traverses a link or reparse point.")
        if path.resolve(strict=True).parent != self.root or not path.is_file():
            raise ValueError("Artifact is not a regular file in this store.")
        if original_chars > MAX_READABLE_ARTIFACT_BYTES:
            raise _ArtifactTooLargeError("Artifact is too large to rehydrate.")
        scan = _scan_utf8_artifact(
            path, offset_chars=0, max_chars=original_chars,
            max_bytes=MAX_READABLE_ARTIFACT_BYTES,
        )
        if scan.digest != digest or scan.total_chars != original_chars:
            raise ValueError("Artifact integrity check failed.")
        return scan.selected_content

    def externalize_conversation(
        self,
        conversation: Conversation,
        *,
        on_failure: ArtifactExternalizationFailureHandler | None = None,
    ) -> Conversation:
        tool_names: dict[str, str] = {}
        externalized_messages: list[Message] = []
        for message in conversation.get_messages():
            if message.role == "assistant":
                for tool_call in message.tool_calls:
                    tool_names[tool_call.id] = tool_call.name
                externalized_messages.append(message)
                continue

            if message.role != "tool" or message.tool_call_id is None:
                externalized_messages.append(message)
                continue

            tool_name = tool_names.get(message.tool_call_id, "unknown")
            try:
                externalized_content = self.externalize(
                    tool_name=tool_name,
                    tool_call_id=message.tool_call_id,
                    content=message.content,
                )
            except Exception as error:
                if on_failure is None:
                    raise
                externalized_content = on_failure(
                    tool_name,
                    message.tool_call_id,
                    message.content,
                    error,
                )
            externalized_messages.append(
                Message(
                    role="tool",
                    content=externalized_content,
                    tool_call_id=message.tool_call_id,
                )
            )

        return Conversation.from_messages(externalized_messages)

    def _write_once(self, path: Path, content: str, *, digest: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            if _is_link_or_reparse_point(path) or not path.is_file():
                raise RuntimeError(
                    "Existing artifact path is not a regular file."
                )
            existing_digest, _ = _sha256_file(path)
            if existing_digest != digest:
                raise RuntimeError(
                    "Existing artifact content does not match its content hash."
                )
            return

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".artifact-",
            suffix=".tmp",
            dir=self.root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                for start in range(0, len(content), ARTIFACT_IO_CHUNK_CHARS):
                    stream.write(content[start : start + ARTIFACT_IO_CHUNK_CHARS])
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


class ReadArtifactArgs(ToolArgs):
    artifact_path: str
    offset_chars: int = Field(default=0, ge=0)
    max_chars: int = Field(
        default=DEFAULT_ARTIFACT_READ_CHARS,
        ge=1,
        le=MAX_ARTIFACT_READ_CHARS,
        strict=True,
    )

    @field_validator("max_chars", mode="before")
    @classmethod
    def clamp_max_chars(cls, value: object) -> object:
        return clamp_positive_int_upper_bound(
            value,
            upper_bound=MAX_ARTIFACT_READ_CHARS,
        )


class ReadArtifactTool(PydanticTool[ReadArtifactArgs]):
    name = "read_artifact"
    description = (
        "Read a bounded text slice from a tool-result artifact referenced in "
        "the current conversation."
    )
    args_model = ReadArtifactArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve(strict=False)

    def _run(self, args: ReadArtifactArgs) -> ToolResult:
        raw_path = Path(args.artifact_path)
        if not raw_path.is_absolute():
            return ToolResult.failure(
                error="Artifact path must be absolute.",
                metadata={"reason": "artifact_path_not_absolute"},
            )

        try:
            if os.path.lexists(raw_path) and _is_link_or_reparse_point(raw_path):
                return ToolResult.failure(
                    error="Artifact path must not be a link or reparse point.",
                    metadata={"reason": "artifact_path_linked"},
                )
            resolved_path = raw_path.resolve(strict=False)
        except OSError:
            return ToolResult.failure(
                error="Artifact path could not be resolved.",
                metadata={"reason": "artifact_path_unavailable"},
            )
        if not resolved_path.is_relative_to(self.artifact_root):
            return ToolResult.failure(
                error="Artifact path is outside the current session artifact root.",
                metadata={"reason": "artifact_path_outside_root"},
            )
        if (
            resolved_path.parent != self.artifact_root
            or resolved_path.suffix != ".txt"
            or len(resolved_path.stem) != 64
            or any(character not in "0123456789abcdef" for character in resolved_path.stem)
        ):
            return ToolResult.failure(
                error="Artifact path is not a valid content-addressed artifact.",
                metadata={"reason": "artifact_path_invalid"},
            )
        if not resolved_path.exists():
            return ToolResult.failure(
                error=f"Artifact not found: {args.artifact_path}",
                metadata={"reason": "artifact_not_found"},
            )
        if not resolved_path.is_file():
            return ToolResult.failure(
                error=f"Artifact path is not a file: {args.artifact_path}",
                metadata={"reason": "artifact_not_file"},
            )

        try:
            artifact_size = resolved_path.stat().st_size
        except OSError:
            return ToolResult.failure(
                error="Artifact metadata could not be read.",
                metadata={"reason": "artifact_read_failed"},
            )
        if artifact_size > MAX_READABLE_ARTIFACT_BYTES:
            return ToolResult.failure(
                error="Artifact is too large to read safely.",
                metadata={
                    "reason": "artifact_too_large",
                    "artifact_bytes": artifact_size,
                    "max_artifact_bytes": MAX_READABLE_ARTIFACT_BYTES,
                },
            )

        try:
            scan = _scan_utf8_artifact(
                resolved_path,
                offset_chars=args.offset_chars,
                max_chars=args.max_chars,
                max_bytes=MAX_READABLE_ARTIFACT_BYTES,
            )
        except _ArtifactTooLargeError:
            return ToolResult.failure(
                error="Artifact is too large to read safely.",
                metadata={
                    "reason": "artifact_too_large",
                    "max_artifact_bytes": MAX_READABLE_ARTIFACT_BYTES,
                },
            )
        except UnicodeDecodeError:
            return ToolResult.failure(
                error="Artifact is not valid UTF-8 text.",
                metadata={"reason": "artifact_encoding_invalid"},
            )
        except OSError:
            return ToolResult.failure(
                error="Artifact content could not be read.",
                metadata={"reason": "artifact_read_failed"},
            )

        if scan.digest != resolved_path.stem:
            return ToolResult.failure(
                error="Artifact content does not match its content hash.",
                metadata={"reason": "artifact_hash_mismatch"},
            )
        return ToolResult.success(
            content=scan.selected_content,
            metadata={
                "artifact_path": resolved_path.as_posix(),
                "offset_chars": args.offset_chars,
                "end_offset_chars": scan.end_offset_chars,
                "max_chars": args.max_chars,
                "total_bytes": scan.total_bytes,
                "total_chars": scan.total_chars,
                "truncated": scan.end_offset_chars < scan.total_chars,
            },
        )


def artifact_directory_for_session(
    state_directory: Path,
    *,
    project_key: str,
    session_id: str,
) -> Path:
    project_component = _validate_artifact_path_component(
        project_key,
        field_name="project_key",
    )
    session_component = _validate_artifact_path_component(
        session_id,
        field_name="session_id",
    )
    return (
        state_directory.resolve(strict=False)
        / "artifacts"
        / project_component
        / session_component
    )


def artifact_quarantine_directory(
    state_directory: Path,
    *,
    deletion_id: str,
) -> Path:
    component = _validate_artifact_path_component(
        deletion_id,
        field_name="deletion_id",
    )
    return (
        state_directory.resolve(strict=False)
        / DELETION_TRASH_DIRECTORY_NAME
        / component
    )


@dataclass(frozen=True)
class ArtifactQuarantineResult:
    had_artifacts: bool
    quarantine_directory: Path


def quarantine_session_artifacts(
    state_directory: Path,
    *,
    project_key: str,
    session_id: str,
    deletion_id: str,
) -> ArtifactQuarantineResult:
    session_directory = validate_session_artifact_cleanup_target(
        state_directory,
        project_key=project_key,
        session_id=session_id,
    )
    quarantine_directory = _validate_artifact_quarantine_target(
        state_directory,
        deletion_id=deletion_id,
    )
    source_exists = os.path.lexists(session_directory)
    quarantine_exists = os.path.lexists(quarantine_directory)
    if source_exists and quarantine_exists:
        raise ArtifactCleanupError(
            "Both the active and quarantined session artifact directories exist."
        )
    if quarantine_exists:
        return ArtifactQuarantineResult(
            had_artifacts=True,
            quarantine_directory=quarantine_directory,
        )
    if not source_exists:
        return ArtifactQuarantineResult(
            had_artifacts=False,
            quarantine_directory=quarantine_directory,
        )

    quarantine_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        session_directory.replace(quarantine_directory)
    except OSError as error:
        raise ArtifactCleanupError(
            "Failed to quarantine the session artifact directory."
        ) from error
    if os.path.lexists(session_directory) or not quarantine_directory.is_dir():
        raise ArtifactCleanupError(
            "Session artifact quarantine did not complete atomically."
        )
    _remove_empty_artifact_parents(session_directory)
    return ArtifactQuarantineResult(
        had_artifacts=True,
        quarantine_directory=quarantine_directory,
    )


def delete_quarantined_artifacts(
    state_directory: Path,
    *,
    deletion_id: str,
) -> bool:
    quarantine_directory = _validate_artifact_quarantine_target(
        state_directory,
        deletion_id=deletion_id,
    )
    if not os.path.lexists(quarantine_directory):
        return False
    try:
        shutil.rmtree(quarantine_directory)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ArtifactCleanupError(
            "Failed to delete the quarantined session artifact directory."
        ) from error
    if os.path.lexists(quarantine_directory):
        raise ArtifactCleanupError(
            "Quarantined session artifact directory still exists after deletion."
        )
    try:
        quarantine_directory.parent.rmdir()
    except OSError:
        pass
    return True


def validate_session_artifact_cleanup_target(
    state_directory: Path,
    *,
    project_key: str,
    session_id: str,
) -> Path:
    try:
        session_directory = artifact_directory_for_session(
            state_directory,
            project_key=project_key,
            session_id=session_id,
        )
    except ValueError as error:
        raise ArtifactCleanupError(
            "Session artifact cleanup target is invalid."
        ) from error

    artifacts_directory = state_directory.resolve(strict=False) / "artifacts"
    project_directory = session_directory.parent
    if project_directory.parent != artifacts_directory:
        raise ArtifactCleanupError(
            "Session artifact cleanup target escaped its project directory."
        )

    for label, path in (
        ("artifact root", artifacts_directory),
        ("project artifact directory", project_directory),
        ("session artifact directory", session_directory),
    ):
        if not os.path.lexists(path):
            continue
        try:
            linked_or_reparse = _is_link_or_reparse_point(path)
        except FileNotFoundError:
            continue
        if linked_or_reparse:
            raise ArtifactCleanupError(
                f"Refusing to delete through a linked {label}: {path}"
            )
        if not path.is_dir():
            raise ArtifactCleanupError(
                f"Expected {label} to be a directory: {path}"
            )
    return session_directory


def delete_session_artifacts(
    state_directory: Path,
    *,
    project_key: str,
    session_id: str,
) -> bool:
    session_directory = validate_session_artifact_cleanup_target(
        state_directory,
        project_key=project_key,
        session_id=session_id,
    )
    if not os.path.lexists(session_directory):
        return False

    try:
        shutil.rmtree(session_directory)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ArtifactCleanupError(
            f"Failed to delete session artifact directory: {session_directory}"
        ) from error
    if os.path.lexists(session_directory):
        raise ArtifactCleanupError(
            f"Session artifact directory still exists after deletion: "
            f"{session_directory}"
        )

    _remove_empty_artifact_parents(session_directory)
    return True


def artifact_externalization_failure_content(
    *,
    tool_name: str,
    tool_call_id: str,
    original_content: str,
    reason: str,
) -> str:
    parsed = parse_tool_result_content(original_content)
    safe_reason = _validate_reason_code(reason)
    metadata = {
        "artifact_externalization_failed": True,
        "context_externalized": False,
        "original_chars": len(original_content),
        "reason": safe_reason,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    return (
        f"{parsed.status}\n"
        f"{ARTIFACT_EXTERNALIZATION_FAILURE_MARKER}\n"
        f"tool_name: {tool_name}\n"
        f"original_chars: {len(original_content)}\n"
        f"reason: {safe_reason}\n"
        f"{TOOL_RESULT_METADATA_MARKER}"
        f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}"
    )


def artifact_failure_reason(error: Exception) -> str:
    if isinstance(error, SessionLockTimeoutError):
        return "session_lock_timeout"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, FileNotFoundError):
        return "path_unavailable"
    if isinstance(error, UnicodeError):
        return "encoding_error"
    if type(error).__name__ in {
        "SessionDeletingError",
        "SessionLeaseLostError",
        "SessionNotFoundError",
    }:
        return "session_unavailable"
    if isinstance(error, OSError):
        return "artifact_io_error"
    if isinstance(error, RuntimeError):
        return "artifact_integrity_error"
    return "artifact_externalization_error"


def _bounded_optional_metadata(
    required: dict[str, object],
    optional: dict[str, object],
) -> dict[str, object]:
    selected: dict[str, object] = {}
    omitted_count = 0
    for key in sorted(optional):
        if key in required:
            continue
        candidate = {**required, **selected, key: optional[key]}
        serialized = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if len(serialized) > MAX_ARTIFACT_REFERENCE_METADATA_CHARS:
            omitted_count += 1
            continue
        selected[key] = optional[key]

    if omitted_count:
        selected["metadata_omitted_count"] = omitted_count
    return selected


def _validate_artifact_path_component(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if (
        normalized == ""
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or "\x00" in normalized
    ):
        raise ValueError(f"{field_name} must be a single safe path component.")
    return normalized


def _is_link_or_reparse_point(path: Path) -> bool:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        return True
    file_attributes = getattr(details, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(file_attributes & reparse_attribute)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(ARTIFACT_IO_CHUNK_BYTES):
            total_bytes += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), total_bytes


def _sha256_text(content: str) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(content), ARTIFACT_IO_CHUNK_CHARS):
        digest.update(
            content[start : start + ARTIFACT_IO_CHUNK_CHARS].encode("utf-8")
        )
    return digest.hexdigest()


def _scan_utf8_artifact(
    path: Path,
    *,
    offset_chars: int,
    max_chars: int,
    max_bytes: int | None = None,
) -> _ArtifactTextScan:
    if offset_chars < 0:
        raise ValueError("offset_chars must not be negative.")
    if max_chars < 0:
        raise ValueError("max_chars must not be negative.")
    if max_bytes is not None and max_bytes < 1:
        raise ValueError("max_bytes must be positive.")

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    total_bytes = 0
    total_chars = 0
    selected_parts: list[str] = []
    requested_end = offset_chars + max_chars

    def consume_text(text: str) -> None:
        nonlocal total_chars
        if not text:
            return
        chunk_start = total_chars
        chunk_end = chunk_start + len(text)
        if (
            max_chars > 0
            and chunk_end > offset_chars
            and chunk_start < requested_end
        ):
            selected_start = max(offset_chars, chunk_start) - chunk_start
            selected_end = min(requested_end, chunk_end) - chunk_start
            selected_parts.append(text[selected_start:selected_end])
        total_chars = chunk_end

    with path.open("rb") as stream:
        if (
            max_bytes is not None
            and os.fstat(stream.fileno()).st_size > max_bytes
        ):
            raise _ArtifactTooLargeError
        while raw_chunk := stream.read(ARTIFACT_IO_CHUNK_BYTES):
            total_bytes += len(raw_chunk)
            if max_bytes is not None and total_bytes > max_bytes:
                raise _ArtifactTooLargeError
            digest.update(raw_chunk)
            consume_text(decoder.decode(raw_chunk, final=False))
        consume_text(decoder.decode(b"", final=True))

    return _ArtifactTextScan(
        digest=digest.hexdigest(),
        total_bytes=total_bytes,
        total_chars=total_chars,
        selected_content="".join(selected_parts),
        end_offset_chars=min(total_chars, requested_end),
    )


def artifact_reference_info(
    *,
    tool_name: str,
    tool_call_id: str,
    content: str,
) -> tuple[Path, str, int] | None:
    body, separator, metadata_text = content.partition(TOOL_RESULT_METADATA_MARKER)
    if (
        separator == ""
        or len(metadata_text) > MAX_ARTIFACT_REFERENCE_METADATA_CHARS
    ):
        return None
    lines = body.splitlines()
    if len(lines) != 6:
        return None
    status, marker, tool_line, path_line, chars_line, digest_line = lines
    if status not in {"OK", "ERROR"} or marker != EXTERNALIZED_TOOL_RESULT_MARKER:
        return None
    if tool_line != f"tool_name: {tool_name}":
        return None
    if not path_line.startswith("artifact_path: "):
        return None
    if not chars_line.startswith("original_chars: "):
        return None
    if not digest_line.startswith("sha256: "):
        return None

    artifact_path_text = path_line.removeprefix("artifact_path: ")
    digest = digest_line.removeprefix("sha256: ")
    try:
        original_chars = int(chars_line.removeprefix("original_chars: "))
        metadata = json.loads(metadata_text)
    except (ValueError, json.JSONDecodeError):
        return None
    if (
        original_chars < 0
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(metadata, dict)
    ):
        return None
    required_metadata = {
        "artifact_path": artifact_path_text,
        "artifact_sha256": digest,
        "context_externalized": True,
        "original_chars": original_chars,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        return None

    return Path(artifact_path_text), digest, original_chars


def _is_valid_artifact_reference(
    root: Path,
    *,
    tool_name: str,
    tool_call_id: str,
    content: str,
) -> bool:
    info = artifact_reference_info(
        tool_name=tool_name, tool_call_id=tool_call_id, content=content,
    )
    if info is None:
        return False
    artifact_path, digest, original_chars = info
    resolved_root = root.resolve(strict=False)
    resolved_path = artifact_path.resolve(strict=False)
    if (
        not artifact_path.is_absolute()
        or resolved_path.parent != resolved_root
        or resolved_path.name != f"{digest}.txt"
        or not resolved_path.is_file()
    ):
        return False
    try:
        if _is_link_or_reparse_point(resolved_path):
            return False
        scan = _scan_utf8_artifact(
            resolved_path,
            offset_chars=0,
            max_chars=0,
        )
    except (OSError, UnicodeError):
        return False
    return (
        scan.digest == digest
        and scan.total_chars == original_chars
    )


def _is_valid_artifact_failure_reference(
    *,
    tool_name: str,
    tool_call_id: str,
    content: str,
) -> bool:
    body, separator, metadata_text = content.partition(TOOL_RESULT_METADATA_MARKER)
    if (
        separator == ""
        or len(metadata_text) > MAX_ARTIFACT_REFERENCE_METADATA_CHARS
    ):
        return False
    lines = body.splitlines()
    if len(lines) != 5:
        return False
    status, marker, tool_line, chars_line, reason_line = lines
    if (
        status not in {"OK", "ERROR"}
        or marker != ARTIFACT_EXTERNALIZATION_FAILURE_MARKER
        or tool_line != f"tool_name: {tool_name}"
        or not chars_line.startswith("original_chars: ")
        or not reason_line.startswith("reason: ")
    ):
        return False
    reason = reason_line.removeprefix("reason: ")
    try:
        original_chars = int(chars_line.removeprefix("original_chars: "))
        metadata = json.loads(metadata_text)
        safe_reason = _validate_reason_code(reason)
    except (ValueError, json.JSONDecodeError):
        return False
    if original_chars < 0 or not isinstance(metadata, dict):
        return False
    required_metadata = {
        "artifact_externalization_failed": True,
        "context_externalized": False,
        "original_chars": original_chars,
        "reason": safe_reason,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    return (
        reason == safe_reason
        and len(metadata) == len(required_metadata)
        and all(
            metadata.get(key) == value
            for key, value in required_metadata.items()
        )
    )


def _validate_artifact_quarantine_target(
    state_directory: Path,
    *,
    deletion_id: str,
) -> Path:
    try:
        quarantine_directory = artifact_quarantine_directory(
            state_directory,
            deletion_id=deletion_id,
        )
    except ValueError as error:
        raise ArtifactCleanupError(
            "Session artifact quarantine target is invalid."
        ) from error
    trash_root = state_directory.resolve(strict=False) / DELETION_TRASH_DIRECTORY_NAME
    if quarantine_directory.parent != trash_root:
        raise ArtifactCleanupError(
            "Session artifact quarantine target escaped its root."
        )
    for label, path in (
        ("artifact quarantine root", trash_root),
        ("artifact quarantine directory", quarantine_directory),
    ):
        if not os.path.lexists(path):
            continue
        try:
            linked_or_reparse = _is_link_or_reparse_point(path)
        except FileNotFoundError:
            continue
        if linked_or_reparse:
            raise ArtifactCleanupError(
                f"Refusing to use a linked {label}: {path}"
            )
        if not path.is_dir():
            raise ArtifactCleanupError(f"Expected {label} to be a directory: {path}")
    return quarantine_directory


def _remove_empty_artifact_parents(session_directory: Path) -> None:
    for empty_parent in (
        session_directory.parent,
        session_directory.parent.parent,
    ):
        try:
            empty_parent.rmdir()
        except OSError:
            break


def _validate_reason_code(reason: str) -> str:
    if (
        reason == ""
        or len(reason) > 100
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in reason)
    ):
        raise ValueError("Artifact failure reason must be a short lowercase code.")
    return reason
