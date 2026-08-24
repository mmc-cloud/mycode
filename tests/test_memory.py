import hashlib
from pathlib import Path

import pytest

from mycode.memory import (
    MemoryError,
    MemoryFormatError,
    MemoryLimits,
    MemoryStore,
    SensitiveMemoryError,
    validate_memory_content,
    validate_memory_key,
)
from mycode.session_store import ProjectIdentity


def memory_store(tmp_path: Path, *, limits: MemoryLimits | None = None) -> MemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return MemoryStore(
        ProjectIdentity.from_workspace(workspace),
        base_directory=tmp_path / "user-state",
        limits=limits,
    )


def test_save_user_memory_creates_readable_markdown(tmp_path: Path) -> None:
    store = memory_store(tmp_path)

    result = store.save(
        scope="user",
        kind="preference",
        key="response.language",
        content="默认使用中文。",
    )

    assert result.action == "created"
    assert result.path == tmp_path / "user-state" / "MEMORY.md"
    text = result.path.read_text(encoding="utf-8")
    assert "# MyCode User Memory" in text
    assert "### response.language" in text
    assert "默认使用中文。" in text
    assert store.list_entries("user") == (result.entry,)


def test_user_and_project_memories_use_separate_files(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    store.save(
        scope="user",
        kind="preference",
        key="response.language",
        content="中文",
    )
    project_result = store.save(
        scope="project",
        kind="fact",
        key="test.command",
        content="uv run pytest",
    )

    assert project_result.path == (
        tmp_path
        / "user-state"
        / "projects"
        / store.project.key
        / "MEMORY.md"
    )
    assert [entry.key for entry in store.list_entries()] == [
        "response.language",
        "test.command",
    ]


def test_update_preserves_unknown_user_text(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    store.save(
        scope="project",
        kind="fact",
        key="test.command",
        content="pytest",
    )
    path = store.path_for_scope("project")
    original = path.read_text(encoding="utf-8")
    path.write_text("User notes stay here.\n\n" + original, encoding="utf-8")

    result = store.save(
        scope="project",
        kind="experience",
        key="test.command",
        content="Use uv run pytest.",
    )

    updated = path.read_text(encoding="utf-8")
    assert result.action == "updated"
    assert updated.startswith("User notes stay here.")
    assert "kind=experience key=test.command" in updated
    assert "Use uv run pytest." in updated
    assert "\npytest\n" not in updated


def test_manual_content_edit_becomes_source_of_truth(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    store.save(
        scope="user",
        kind="preference",
        key="response.style",
        content="Concise",
    )
    path = store.path_for_scope("user")
    text = path.read_text(encoding="utf-8").replace("Concise", "Detailed")
    path.write_text(text, encoding="utf-8")

    entries = store.list_entries("user")

    assert len(entries) == 1
    assert entries[0].content == "Detailed"


def test_delete_removes_managed_block_and_preserves_notes(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    store.save(
        scope="user",
        kind="preference",
        key="response.style",
        content="Detailed",
    )
    path = store.path_for_scope("user")
    path.write_text(
        "Personal note.\n\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert store.delete(scope="user", key="response.style") is True
    assert store.delete(scope="user", key="response.style") is False
    assert "Personal note." in path.read_text(encoding="utf-8")
    assert store.list_entries("user") == ()


def test_structural_error_blocks_automatic_edit(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    path = store.path_for_scope("user")
    path.parent.mkdir(parents=True)
    path.write_text(
        "<!-- mycode-memory:entry kind=preference key=response.style -->\n"
        "### response.style\n\nDetailed\n",
        encoding="utf-8",
    )

    document = store.read_document("user")
    assert any(issue.code == "missing_end_marker" for issue in document.issues)
    with pytest.raises(MemoryFormatError, match="structural errors"):
        store.save(
            scope="user",
            kind="preference",
            key="another.key",
            content="value",
        )


def test_manual_sensitive_entry_is_withheld_but_can_be_replaced(
    tmp_path: Path,
) -> None:
    store = memory_store(tmp_path)
    path = store.path_for_scope("user")
    path.parent.mkdir(parents=True)
    path.write_text(
        "<!-- mycode-memory:entry kind=fact key=provider.secret -->\n"
        "### provider.secret\n\napi_key=super-secret-value\n"
        "<!-- mycode-memory:end -->\n",
        encoding="utf-8",
    )

    document = store.read_document("user")
    assert document.entries == ()
    assert any(issue.code == "sensitive_content" for issue in document.issues)

    store.save(
        scope="user",
        kind="fact",
        key="provider.secret",
        content="Provider credentials stay outside memory.",
    )

    assert store.list_entries("user")[0].content == (
        "Provider credentials stay outside memory."
    )


def test_invalid_utf8_and_large_files_are_reported(tmp_path: Path) -> None:
    store = memory_store(
        tmp_path,
        limits=MemoryLimits(max_file_bytes=8, max_entry_chars=20, max_entries=2),
    )
    path = store.path_for_scope("user")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff")
    assert store.read_document("user").issues[0].code == "invalid_utf8"

    path.write_bytes(b"x" * 9)
    assert store.read_document("user").issues[0].code == "file_too_large"


def test_entry_count_and_content_limits_are_enforced(tmp_path: Path) -> None:
    store = memory_store(
        tmp_path,
        limits=MemoryLimits(
            max_file_bytes=10_000,
            max_entry_chars=10,
            max_entries=1,
        ),
    )
    store.save(scope="user", kind="fact", key="first", content="12345")

    with pytest.raises(ValueError, match="10 characters"):
        store.save(scope="user", kind="fact", key="first", content="x" * 11)
    with pytest.raises(MemoryError, match="already contains 1"):
        store.save(scope="user", kind="fact", key="second", content="value")


def test_memory_validation_normalizes_key_and_rejects_secrets() -> None:
    assert validate_memory_key(" Response.Language ") == "response.language"
    with pytest.raises(ValueError, match="Memory key"):
        validate_memory_key("bad key")
    with pytest.raises(SensitiveMemoryError, match="secret"):
        validate_memory_content("OPENAI_API_KEY=super-secret-value", max_chars=100)


def test_memory_document_records_raw_file_fingerprint(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    raw_content = (
        b"\xef\xbb\xbf<!-- mycode-memory:entry kind=fact key=test.command -->\r\n"
        b"### test.command\r\npytest\r\n<!-- mycode-memory:end -->\r\n"
    )
    path = store.path_for_scope("project")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_content)

    document = store.read_document("project")

    assert document.content_bytes == len(raw_content)
    assert document.sha256 == hashlib.sha256(raw_content).hexdigest()
    assert document.entries[0].content == "pytest"


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    store = memory_store(tmp_path)
    store.save(
        scope="project",
        kind="fact",
        key="project.name",
        content="demo",
    )

    memory_directory = store.path_for_scope("project").parent
    assert [path.name for path in memory_directory.iterdir()] == ["MEMORY.md"]
