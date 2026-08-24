from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import time
from typing import BinaryIO


DEFAULT_SESSION_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_SESSION_LOCK_POLL_SECONDS = 0.05
SESSION_LOCK_STRIPE_COUNT = 64


class SessionLockError(RuntimeError):
    pass


class SessionLockTimeoutError(SessionLockError):
    pass


@dataclass(frozen=True)
class SessionOperationLock:
    """Small cross-process exclusive lock backed by a stable control file."""

    path: Path
    timeout_seconds: float = DEFAULT_SESSION_LOCK_TIMEOUT_SECONDS
    poll_seconds: float = DEFAULT_SESSION_LOCK_POLL_SECONDS

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be at least 0.")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be above 0.")
        object.__setattr__(self, "path", self.path.resolve(strict=False))

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as stream:
            _ensure_lock_byte(stream)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    _lock_stream(stream)
                    break
                except OSError as error:
                    if not _is_lock_contention(error):
                        raise SessionLockError(
                            f"Failed to acquire session operation lock: {self.path}"
                        ) from error
                    if time.monotonic() >= deadline:
                        raise SessionLockTimeoutError(
                            "Timed out waiting for another session operation to finish."
                        ) from error
                    time.sleep(
                        min(
                            self.poll_seconds,
                            max(0.0, deadline - time.monotonic()),
                        )
                    )

            try:
                yield
            finally:
                try:
                    _unlock_stream(stream)
                except OSError as error:
                    raise SessionLockError(
                        f"Failed to release session operation lock: {self.path}"
                    ) from error


def session_operation_lock_path(
    state_directory: Path,
    *,
    project_key: str,
    session_id: str,
) -> Path:
    identity = f"{project_key}\0{session_id}".encode("utf-8")
    stripe = hashlib.sha256(identity).digest()[0] % SESSION_LOCK_STRIPE_COUNT
    return (
        state_directory.resolve(strict=False)
        / "locks"
        / f"session-{stripe:02x}.lock"
    )


def database_maintenance_lock_path(state_directory: Path) -> Path:
    return state_directory.resolve(strict=False) / "locks" / "database-maintenance.lock"


def _ensure_lock_byte(stream: BinaryIO) -> None:
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())


def _lock_stream(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_stream(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _is_lock_contention(error: OSError) -> bool:
    return error.errno in {
        errno.EACCES,
        errno.EAGAIN,
        errno.EDEADLK,
    } or getattr(error, "winerror", None) in {33, 36}
