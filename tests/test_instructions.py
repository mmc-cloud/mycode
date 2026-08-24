import hashlib
from pathlib import Path

import pytest

from mycode.instructions import (
    InstructionLimits,
    InstructionPathError,
    load_instruction_bundle,
)


def test_load_instruction_bundle_orders_user_project_and_directory_sources(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    working_directory = workspace / "src" / "feature"
    user_directory.mkdir()
    working_directory.mkdir(parents=True)
    (user_directory / "AGENTS.md").write_text("user rules", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("project rules", encoding="utf-8")
    (workspace / "src" / "AGENTS.md").write_text("src rules", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("feature rules", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        working_directory=working_directory,
        user_instruction_directory=user_directory,
    )

    assert [source.scope for source in bundle.sources] == [
        "user",
        "project",
        "directory",
        "directory",
    ]
    assert [source.content for source in bundle.sources] == [
        "user rules",
        "project rules",
        "src rules",
        "feature rules",
    ]
    assert bundle.issues == ()


def test_load_instruction_bundle_prefers_agents_over_claude(tmp_path: Path) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("agents rules", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("claude rules", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert [source.path.name for source in bundle.sources] == ["AGENTS.md"]
    assert bundle.sources[0].content == "agents rules"


def test_load_instruction_bundle_prefers_mycode_over_compatible_files(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "MYCODE.md").write_text("mycode rules", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("agents rules", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("claude rules", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert [source.path.name for source in bundle.sources] == ["MYCODE.md"]
    assert bundle.sources[0].content == "mycode rules"


def test_load_instruction_bundle_selects_priority_independently_per_scope(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    working_directory = workspace / "src" / "feature"
    user_directory.mkdir()
    working_directory.mkdir(parents=True)

    (user_directory / "MYCODE.md").write_text("user mycode", encoding="utf-8")
    (user_directory / "AGENTS.md").write_text("user agents", encoding="utf-8")
    (user_directory / "CLAUDE.md").write_text("user claude", encoding="utf-8")
    (workspace / "MYCODE.md").write_text("project mycode", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("project agents", encoding="utf-8")
    (workspace / "src" / "AGENTS.md").write_text("src agents", encoding="utf-8")
    (workspace / "src" / "CLAUDE.md").write_text("src claude", encoding="utf-8")
    (working_directory / "CLAUDE.md").write_text(
        "feature claude",
        encoding="utf-8",
    )

    bundle = load_instruction_bundle(
        workspace,
        working_directory=working_directory,
        user_instruction_directory=user_directory,
    )

    assert [source.scope for source in bundle.sources] == [
        "user",
        "project",
        "directory",
        "directory",
    ]
    assert [source.path.name for source in bundle.sources] == [
        "MYCODE.md",
        "MYCODE.md",
        "AGENTS.md",
        "CLAUDE.md",
    ]
    assert [source.content for source in bundle.sources] == [
        "user mycode",
        "project mycode",
        "src agents",
        "feature claude",
    ]


def test_load_instruction_bundle_falls_back_to_claude(tmp_path: Path) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (user_directory / "CLAUDE.md").write_text("user fallback", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("project fallback", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert [source.path.name for source in bundle.sources] == [
        "CLAUDE.md",
        "CLAUDE.md",
    ]
    assert [source.content for source in bundle.sources] == [
        "user fallback",
        "project fallback",
    ]


def test_load_instruction_bundle_rejects_working_directory_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()

    with pytest.raises(InstructionPathError, match="inside workspace_root"):
        load_instruction_bundle(workspace, working_directory=outside)


def test_load_instruction_bundle_reports_large_file(tmp_path: Path) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("x" * 11, encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
        limits=InstructionLimits(max_file_bytes=10, max_total_chars=100),
    )

    assert bundle.sources == ()
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "file_too_large"


def test_load_instruction_bundle_reports_invalid_utf8(tmp_path: Path) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "AGENTS.md").write_bytes(b"\xff\xfe")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert bundle.sources == ()
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "invalid_utf8"


def test_load_instruction_bundle_reports_total_limit_without_truncating(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (user_directory / "AGENTS.md").write_text("12345", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("67890", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
        limits=InstructionLimits(max_file_bytes=100, max_total_chars=7),
    )

    assert [source.content for source in bundle.sources] == ["12345"]
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "total_too_large"


def test_load_instruction_bundle_rejects_symlink_outside_scope(tmp_path: Path) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.md"
    user_directory.mkdir()
    workspace.mkdir()
    outside.write_text("outside rules", encoding="utf-8")
    instruction_path = workspace / "AGENTS.md"
    try:
        instruction_path.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are not available in this environment.")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert bundle.sources == ()
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "outside_scope"


def test_user_instruction_symlink_can_target_sibling_dotfiles_directory(
    tmp_path: Path,
) -> None:
    home_directory = tmp_path / "home"
    user_directory = home_directory / ".mycode"
    dotfiles_directory = home_directory / "dotfiles"
    workspace = tmp_path / "workspace"
    user_directory.mkdir(parents=True)
    dotfiles_directory.mkdir()
    workspace.mkdir()
    target = dotfiles_directory / "MYCODE.md"
    target.write_text("dotfiles user rules", encoding="utf-8")
    link = user_directory / "MYCODE.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are not available in this environment.")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert bundle.issues == ()
    assert len(bundle.sources) == 1
    assert bundle.sources[0].scope == "user"
    assert bundle.sources[0].path == target.resolve()
    assert bundle.sources[0].content == "dotfiles user rules"


def test_user_instruction_symlink_cannot_leave_user_home_boundary(
    tmp_path: Path,
) -> None:
    home_directory = tmp_path / "home"
    user_directory = home_directory / ".mycode"
    outside_directory = tmp_path / "outside"
    workspace = tmp_path / "workspace"
    user_directory.mkdir(parents=True)
    outside_directory.mkdir()
    workspace.mkdir()
    target = outside_directory / "MYCODE.md"
    target.write_text("outside user rules", encoding="utf-8")
    link = user_directory / "MYCODE.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are not available in this environment.")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert bundle.sources == ()
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "outside_scope"


def test_broken_native_symlink_reports_specific_issue_without_fallback(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("fallback rules", encoding="utf-8")
    native_link = workspace / "MYCODE.md"
    try:
        native_link.symlink_to(workspace / "missing-MYCODE.md")
    except OSError:
        pytest.skip("Symlinks are not available in this environment.")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    assert bundle.sources == ()
    assert len(bundle.issues) == 1
    assert bundle.issues[0].code == "broken_symlink"
    assert bundle.issues[0].path == native_link


def test_instruction_bundle_prompt_text_includes_scope_path_and_content(
    tmp_path: Path,
) -> None:
    user_directory = tmp_path / "user"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project rules\n", encoding="utf-8")

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=user_directory,
    )

    prompt_text = bundle.to_prompt_text()
    assert "### project:" in prompt_text
    assert str((workspace / "AGENTS.md").resolve()) in prompt_text
    assert prompt_text.endswith("project rules")


def test_loaded_instruction_source_records_raw_file_fingerprint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_content = b"\xef\xbb\xbfproject rules\r\n"
    (workspace / "AGENTS.md").write_bytes(raw_content)

    bundle = load_instruction_bundle(
        workspace,
        user_instruction_directory=tmp_path / "user",
    )

    source = bundle.sources[0]
    assert source.content == "project rules\r\n"
    assert source.content_bytes == len(raw_content)
    assert source.sha256 == hashlib.sha256(raw_content).hexdigest()
