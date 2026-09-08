import json
from pathlib import Path

import pytest

from mycode.project import ProjectIdentity
from mycode.persistence.project_storage import (
    PROJECT_HASH_LENGTH,
    ProjectMetadataError,
    ProjectStorage,
    ProjectStorageError,
    project_directory_name,
)


def test_project_storage_creates_stable_layout_and_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "Work Space"
    workspace.mkdir()
    projects_root = tmp_path / "state" / "projects"
    identity = ProjectIdentity.from_workspace(workspace)

    storage = ProjectStorage.open(identity, projects_root=projects_root)

    expected_name = f"Work Space-{identity.key[:PROJECT_HASH_LENGTH]}"
    assert project_directory_name(identity) == expected_name
    assert storage.project_directory == projects_root.resolve() / expected_name
    assert json.loads(storage.metadata_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "project_key": identity.key,
        "workspace_root": str(identity.workspace_root),
    }
    assert storage.sessions_directory.is_dir()
    assert storage.locks_directory == storage.project_directory / "locks"
    assert storage.locks_directory.is_dir()


def test_project_storage_rejects_metadata_for_another_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    identity = ProjectIdentity.from_workspace(workspace)
    storage = ProjectStorage.open(identity, projects_root=tmp_path / "projects")
    storage.metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_key": "0" * 64,
                "workspace_root": str(workspace),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectMetadataError, match="does not match"):
        ProjectStorage.open(identity, projects_root=tmp_path / "projects")


def test_session_layout_creates_only_the_new_filesystem_contract(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    )

    session = storage.session("session-123", create=True)

    assert session.root == storage.sessions_directory / "session-123"
    assert session.transcript_path == session.root / "transcript.jsonl"
    assert session.meta_path == session.root / "meta.json"
    assert session.compact_path == session.root / "compact.json"
    assert session.artifacts_directory == session.root / "artifacts"
    assert session.subagents_directory == session.root / "subagents"
    assert session.subagent_log_path("run-456") == (
        session.root / "subagents" / "run-456.jsonl"
    )
    assert session.lock_path == storage.locks_directory / "session-123.lock"
    assert not session.lock_path.is_relative_to(session.root)
    assert session.artifacts_directory.is_dir()
    assert session.subagents_directory.is_dir()
    assert not session.transcript_path.exists()


def test_session_ids_map_to_distinct_project_lock_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    )

    first = storage.session("session-1")
    second = storage.session("session-2")

    assert first.lock_path == storage.locks_directory / "session-1.lock"
    assert second.lock_path == storage.locks_directory / "session-2.lock"
    assert not first.root.exists()
    assert not second.root.exists()


@pytest.mark.parametrize(
    "session_id",
    ["../outside", "nested/session", "nested\\session", ".", "..", "CON"],
)
def test_session_layout_rejects_unsafe_components(
    tmp_path: Path,
    session_id: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    )

    with pytest.raises(ProjectStorageError, match="Invalid session_id"):
        storage.session(session_id, create=True)


def test_subagent_log_path_rejects_unsafe_run_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    ).session("session-123", create=True)

    with pytest.raises(ProjectStorageError, match="Invalid run_id"):
        session.subagent_log_path("../run")


def test_session_layout_rejects_symlinked_session_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = ProjectStorage.open(
        ProjectIdentity.from_workspace(workspace),
        projects_root=tmp_path / "projects",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    link = storage.sessions_directory / "linked-session"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    with pytest.raises(ProjectStorageError, match="Could not resolve"):
        storage.session("linked-session", create=True)
