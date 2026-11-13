"""User-scoped credential snapshots and live reference leases.

No credentials belong in Session data, logs, UI summaries or exceptions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from uuid import uuid4
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from mycode.model_config import APIConnectionConfig
from mycode.model_selection import ModelSelection
from mycode.persistence.filesystem import read_json_snapshot, validate_storage_path, write_json_snapshot
from mycode.persistence.session_lock import SessionLifecycleLock, SessionLockTimeoutError


class ConnectionStoreError(ValueError):
    pass


class SavedConnection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    base_url: str
    api_key: str = Field(repr=False, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("id")
    @classmethod
    def reserved_id(cls, value):
        if value == "env":
            raise ValueError("env is reserved for environment configuration.")
        return value

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an HTTP(S) Base URL without credentials, query or fragment.")
        return value.rstrip("/")

    def connection(self):
        return APIConnectionConfig(self.api_key, self.base_url, self.headers)


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    default_ref: str | None = None
    default_models: ModelSelection | None = None
    connections: dict[str, SavedConnection] = Field(default_factory=dict, repr=False)
    model_windows: dict[str, dict[str, PositiveInt]] = Field(default_factory=dict)


@dataclass
class ConnectionLease:
    ownership: object = field(repr=False)
    path: Path
    lock_path: Path

    def close(self):
        if self.ownership is not None:
            ownership, self.ownership = self.ownership, None
            ownership.__exit__(None, None, None)
            for path in (self.path, self.lock_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass  # An inactive lease never blocks future management.


class ConnectionStore:
    def __init__(self, root: Path | None = None):
        self.root = Path.home() / ".mycode" if root is None else Path(root)
        self.path = self.root / "connections.json"

    @contextmanager
    def transaction(self):
        # Only acquisition failures are storage errors. Exceptions from the
        # protected operation retain their original type and explanation.
        try:
            self._check_root()
            self.root.mkdir(parents=True, exist_ok=True)
            lock = validate_storage_path(self.root, "connections.lock")
            ownership = SessionLifecycleLock(lock, timeout_seconds=2).acquire()
            ownership.__enter__()
        except ConnectionStoreError:
            raise
        except Exception:
            raise ConnectionStoreError("Could not acquire connection storage; existing data was preserved.") from None
        try:
            yield
        finally:
            ownership.__exit__(None, None, None)

    def read(self):
        self._check_root()
        if not self.path.exists() and not self.path.is_symlink():
            return _Document()
        try:
            data = read_json_snapshot(self.root, self.path)
            doc = _Document.model_validate(data)
            if any(key != value.id for key, value in doc.connections.items()):
                raise ValueError()
            if doc.default_ref is not None and doc.default_ref not in doc.connections:
                raise ValueError()
            if doc.default_models is not None and not doc.default_models.references() <= doc.connections.keys():
                raise ValueError()
            for connection in doc.connections.values():
                connection.connection()
            return doc
        except Exception:
            raise ConnectionStoreError("Saved connection data is invalid; the file was preserved.") from None

    def _check_root(self):
        try:
            info = self.root.lstat()
        except FileNotFoundError:
            return
        if self.root.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ConnectionStoreError("Connection storage root cannot be a link or reparse point.")

    def _write(self, doc):
        _secure_snapshot(self.root, self.path, doc.model_dump())

    def save(self, connection: SavedConnection, *, edit=False):
        with self.transaction():
            doc = self.read()
            if edit != (connection.id in doc.connections):
                raise ConnectionStoreError("Connection does not exist." if edit else "Connection ID already exists.")
            if edit:
                self._ensure_unused(connection.id)
            connection.connection()
            doc.connections[connection.id] = connection
            if edit:
                doc.model_windows.pop(connection.id, None)
                if doc.default_models is not None:
                    selection = doc.default_models
                    doc.default_models = selection.model_copy(update={
                        role: getattr(selection, role).model_copy(update={
                            "context_window_tokens": None, "window_binding": None})
                        for role in ("main", "subagent", "compact")
                        if (getattr(selection, role).connection_ref or selection.main.connection_ref) == connection.id})
            self._write(doc)

    def delete(self, identifier):
        with self.transaction():
            doc = self.read()
            if identifier not in doc.connections:
                raise ConnectionStoreError("Connection does not exist.")
            self._ensure_unused(identifier)
            if doc.default_ref == identifier:
                raise ConnectionStoreError("Choose or clear the default connection before deleting it.")
            if doc.default_models is not None and identifier in doc.default_models.references():
                raise ConnectionStoreError("Clear or replace default role settings before deleting this connection.")
            del doc.connections[identifier]
            doc.model_windows.pop(identifier, None)
            self._write(doc)

    def set_default(self, identifier):
        with self.transaction():
            doc = self.read()
            if identifier is not None and identifier not in doc.connections:
                raise ConnectionStoreError("Connection does not exist.")
            doc.default_ref = identifier
            self._write(doc)

    def set_model_defaults(self, selection):
        with self.transaction():
            doc = self.read()
            if selection is not None:
                selection = ModelSelection.model_validate(selection)
                if not selection.references() <= doc.connections.keys():
                    raise ConnectionStoreError("Default model settings refer to a missing connection.")
            doc.default_models = selection
            self._write(doc)

    def _pin(self, references):
        """Pin already-resolved references while the caller holds transaction()."""
        if not references:
            return None
        directory = validate_storage_path(self.root, "connection-leases")
        directory.mkdir(exist_ok=True)
        name = uuid4().hex
        path = directory / f"{name}.json"
        lock_path = directory / f"{name}.lock"
        ownership = SessionLifecycleLock(lock_path).acquire()
        ownership.__enter__()
        try:
            write_json_snapshot(self.root, path, {"refs": sorted(references)})
            return ConnectionLease(ownership, path, lock_path)
        except BaseException:
            ownership.__exit__(None, None, None)
            path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
            raise

    def _ensure_unused(self, identifier):
        directory = validate_storage_path(self.root, "connection-leases")
        if not directory.exists():
            return
        for path in directory.glob("*.json"):
            lock_path = validate_storage_path(self.root, path.with_suffix(".lock"))
            try:
                with SessionLifecycleLock(lock_path, timeout_seconds=0).acquire():
                    continue  # No live owner: crash residue cannot block management.
            except SessionLockTimeoutError:
                data = read_json_snapshot(self.root, path)
                if identifier in data.get("refs", []):
                    raise ConnectionStoreError("Connection is used by an active Session; switch away or close it first.")


def _secure_snapshot(root, target, payload):
    """Restrict an empty temp file before writing any credentials, then replace."""
    validate_storage_path(root, target)
    descriptor, name = tempfile.mkstemp(prefix=".connections-", suffix=".tmp", dir=root)
    temporary = Path(name)
    try:
        if os.name == "nt":
            result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, check=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            sid = list(csv.reader(io.StringIO(result.stdout.decode(errors="replace"))))[0][-1]
            if not re.fullmatch(r"S-1-[0-9-]+", sid):
                raise OSError("Cannot determine user identity.")
            subprocess.run(["icacls", name, "/inheritance:r", "/grant:r", f"*{sid}:F", "*S-1-5-18:F"],
                           capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        raise ConnectionStoreError("Could not securely save connections; previous data was preserved.") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
