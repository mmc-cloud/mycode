from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Literal


TrailingRecordPolicy = Literal["error", "ignore", "truncate"]


class FilesystemStorageError(RuntimeError):
    pass


class StorageBoundaryError(FilesystemStorageError):
    pass


class JsonSnapshotError(FilesystemStorageError):
    pass


class JsonLinesError(FilesystemStorageError):
    pass


class JsonLinesCorruptionError(JsonLinesError):
    pass


def ensure_storage_directory(root: str | Path, path: str | Path) -> Path:
    resolved_root = Path(root).resolve(strict=False)
    resolved_root.mkdir(parents=True, exist_ok=True)
    raw = Path(path)
    requested = raw if raw.is_absolute() else resolved_root / raw
    if requested.resolve(strict=False) == resolved_root:
        if not resolved_root.is_dir():
            raise StorageBoundaryError(
                f"Storage root is not a directory: {resolved_root}"
            )
        return resolved_root
    candidate = _bounded_path(resolved_root, path)
    _reject_symlink_components(resolved_root, candidate)
    candidate.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(resolved_root, candidate)
    if not candidate.is_dir():
        raise StorageBoundaryError(f"Storage path is not a directory: {candidate}")
    return candidate


def write_json_snapshot(
    root: str | Path,
    path: str | Path,
    payload: Mapping[str, object],
) -> None:
    target = _prepare_file_path(root, path)
    try:
        content = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise JsonSnapshotError("JSON snapshot payload is not serializable.") from error
    _atomic_write(target, content)


def read_json_snapshot(root: str | Path, path: str | Path) -> dict[str, object]:
    target = _bounded_file_path(root, path, allow_missing=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise JsonSnapshotError(f"Invalid JSON snapshot: {target}") from error
    except OSError as error:
        raise JsonSnapshotError(f"Could not read JSON snapshot: {target}") from error
    if not isinstance(payload, dict):
        raise JsonSnapshotError(f"JSON snapshot must contain an object: {target}")
    return payload


def append_jsonl_record(
    root: str | Path,
    path: str | Path,
    record: Mapping[str, object],
) -> None:
    target = _prepare_file_path(root, path)
    try:
        content = (
            json.dumps(
                dict(record),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise JsonLinesError("JSONL record is not serializable.") from error

    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(target, flags, 0o600)
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("append made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise JsonLinesError(f"Could not append JSONL record: {target}") from error


def read_jsonl_records(
    root: str | Path,
    path: str | Path,
    *,
    trailing_record: TrailingRecordPolicy = "error",
) -> list[dict[str, object]]:
    if trailing_record not in {"error", "ignore", "truncate"}:
        raise ValueError("trailing_record must be 'error', 'ignore', or 'truncate'.")
    target = _bounded_file_path(root, path, allow_missing=True)
    try:
        content = target.read_bytes()
    except OSError as error:
        raise JsonLinesError(f"Could not read JSONL file: {target}") from error

    lines = content.splitlines(keepends=True)
    has_unterminated_tail = bool(content) and not content.endswith(b"\n")
    records: list[dict[str, object]] = []
    offset = 0
    for index, raw_line in enumerate(lines):
        line_number = index + 1
        try:
            decoded = raw_line.rstrip(b"\r\n").decode("utf-8")
            record = json.loads(decoded)
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            is_trailing = index == len(lines) - 1
            is_recoverable_tail = is_trailing and has_unterminated_tail
            if not is_recoverable_tail or trailing_record == "error":
                location = "trailing" if is_trailing else "middle"
                raise JsonLinesCorruptionError(
                    f"Invalid {location} JSONL record at line {line_number}: {target}"
                ) from error
            if trailing_record == "truncate":
                _truncate_file(target, offset)
            break
        if not isinstance(record, dict):
            raise JsonLinesCorruptionError(
                f"JSONL record at line {line_number} must be an object: {target}"
            )
        records.append(record)
        offset += len(raw_line)
    return records


def prepare_jsonl_for_append(root: str | Path, path: str | Path) -> list[dict[str, object]]:
    """Normalize under exclusive ownership and return the records already parsed."""
    records = read_jsonl_records(root, path, trailing_record="truncate")
    target = _bounded_file_path(root, path)
    try:
        with target.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
    except OSError as error:
        raise JsonLinesError("Could not prepare JSONL for append.") from error
    return records


def _prepare_file_path(root: str | Path, path: str | Path) -> Path:
    resolved_root = Path(root).resolve(strict=False)
    resolved_root.mkdir(parents=True, exist_ok=True)
    target = _bounded_path(resolved_root, path)
    ensure_storage_directory(resolved_root, target.parent)
    return _bounded_file_path(resolved_root, target, allow_missing=True)


def _bounded_file_path(
    root: str | Path,
    path: str | Path,
    *,
    allow_missing: bool = False,
) -> Path:
    resolved_root = Path(root).resolve(strict=False)
    target = _bounded_path(resolved_root, path)
    _reject_symlink_components(resolved_root, target)
    if target.is_symlink():
        raise StorageBoundaryError(f"Storage file cannot be a symlink: {target}")
    if not allow_missing and not target.is_file():
        raise JsonSnapshotError(f"Storage file does not exist: {target}")
    if target.exists() and not target.is_file():
        raise StorageBoundaryError(f"Storage path is not a file: {target}")
    return target


def _bounded_path(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise StorageBoundaryError(f"Storage path contains traversal: {raw}")
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise StorageBoundaryError(f"Storage path escapes its root: {candidate}")
    return candidate


def validate_storage_path(root: str | Path, path: str | Path) -> Path:
    """Check existing components without creating or resolving away links."""
    boundary = Path(root).resolve(strict=False)
    candidate = _bounded_path(boundary, path)
    _reject_symlink_components(boundary, candidate)
    return candidate


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise StorageBoundaryError(
            f"Storage path escapes its root: {candidate}"
        ) from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0,
        ):
            raise StorageBoundaryError(
                f"Storage path contains a symlink below its root: {current}"
            )


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as error:
        raise JsonSnapshotError(
            f"Could not atomically write JSON snapshot: {path}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _truncate_file(path: Path, size: int) -> None:
    try:
        with path.open("r+b") as stream:
            stream.truncate(size)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise JsonLinesError(
            f"Could not truncate trailing JSONL record: {path}"
        ) from error
