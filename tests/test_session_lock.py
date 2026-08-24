from pathlib import Path
import subprocess
import sys

from mycode.session_lock import (
    SessionOperationLock,
    session_operation_lock_path,
)


def test_session_operation_lock_is_cross_process_and_uses_generic_stripe_name(
    tmp_path: Path,
) -> None:
    lock_path = session_operation_lock_path(
        tmp_path,
        project_key="project-secret-name",
        session_id="session-secret-name",
    )
    assert "project-secret-name" not in lock_path.name
    assert "session-secret-name" not in lock_path.name
    script = (
        "from pathlib import Path\n"
        "from mycode.session_lock import SessionOperationLock, "
        "SessionLockTimeoutError\n"
        f"lock = SessionOperationLock(Path({str(lock_path)!r}), "
        "timeout_seconds=0.1, poll_seconds=0.01)\n"
        "try:\n"
        "    with lock.acquire():\n"
        "        print('acquired')\n"
        "except SessionLockTimeoutError:\n"
        "    print('timeout')\n"
    )

    with SessionOperationLock(lock_path).acquire():
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
