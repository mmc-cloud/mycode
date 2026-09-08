from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import time
from typing import BinaryIO


DEFAULT_SESSION_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_SESSION_LOCK_POLL_SECONDS = 0.05


class SessionLockError(RuntimeError):
    pass


class SessionLockTimeoutError(SessionLockError):
    pass


@dataclass(frozen=True)
class SessionLifecycleLock:
    """Exclusive OS lock held for the complete lifetime of one session owner."""

    path: Path
    timeout_seconds: float = DEFAULT_SESSION_LOCK_TIMEOUT_SECONDS
    poll_seconds: float = DEFAULT_SESSION_LOCK_POLL_SECONDS

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be at least 0.")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be above 0.")
        raw_path = Path(self.path)
        if raw_path.is_symlink():
            raise SessionLockError("Session lifecycle lock file cannot be a symlink.")
        object.__setattr__(
            self,
            "path",
            raw_path.parent.resolve(strict=False) / raw_path.name,
        )

    @contextmanager
    def acquire(self) -> Iterator[None]:
        with _acquire_lifecycle_lock(
            self.path,
            timeout_seconds=self.timeout_seconds,
            poll_seconds=self.poll_seconds,
        ):
            yield


@contextmanager
def _acquire_lifecycle_lock(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SessionLockError("Session lifecycle lock file cannot be a symlink.")
    with path.open("a+b") as stream:
        _ensure_lock_byte(stream)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                _lock_stream(stream)
                break
            except OSError as error:
                if not _is_lock_contention(error):
                    raise SessionLockError(
                        f"Failed to acquire session lifecycle lock: {path}"
                    ) from error
                if time.monotonic() >= deadline:
                    raise SessionLockTimeoutError(
                        "Timed out waiting for another session lifecycle owner to finish."
                    ) from error
                time.sleep(
                    min(
                        poll_seconds,
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
                    f"Failed to release session lifecycle lock: {path}"
                ) from error


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
