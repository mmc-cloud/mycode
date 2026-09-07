from pathlib import Path
import subprocess
import sys

import pytest

from mycode.project import ProjectIdentity
from mycode.project_storage import ProjectStorage
from mycode.session_lock import (
    SessionLifecycleLock,
    SessionLockError,
)


def test_session_lifecycle_lock_rejects_symlink_file(tmp_path: Path) -> None:
    target = tmp_path / "outside.lock"
    target.touch()
    link = tmp_path / "session.lock"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    with pytest.raises(SessionLockError, match="cannot be a symlink"):
        SessionLifecycleLock(link)


def test_session_lifecycle_lock_is_exclusive_for_complete_owner_scope(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "project" / "locks" / "session-1.lock"
    script = (
        "from pathlib import Path\n"
        "from mycode.session_lock import SessionLifecycleLock, "
        "SessionLockTimeoutError\n"
        f"lock = SessionLifecycleLock(Path({str(lock_path)!r}), "
        "timeout_seconds=0.1, poll_seconds=0.01)\n"
        "try:\n"
        "    with lock.acquire():\n"
        "        print('acquired')\n"
        "except SessionLockTimeoutError:\n"
        "    print('timeout')\n"
    )

    with SessionLifecycleLock(lock_path).acquire():
        blocked = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    acquired = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert blocked.stdout.strip() == "timeout"
    assert acquired.stdout.strip() == "acquired"


def test_lifecycle_lock_after_session_deletion_does_not_recreate_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    )
    session = storage.session("deleted-session", create=True)
    session.artifacts_directory.rmdir()
    session.subagents_directory.rmdir()
    session.root.rmdir()

    with SessionLifecycleLock(session.lock_path).acquire():
        assert session.lock_path.exists()

    assert not session.root.exists()
