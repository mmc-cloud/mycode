from pathlib import Path

import pytest

from mycode.skills import (
    DuplicateSkillError,
    Skill,
    SkillNotFoundError,
    SkillPathError,
    SkillRegistry,
)


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Use this test Skill.",
    body: str = "Follow the test procedure.",
    extra: str = "",
) -> Path:
    skill_root = root / name
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{extra}"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_root


def discover(
    workspace: Path, *, builtin: Path, user: Path
) -> SkillRegistry:
    return SkillRegistry.discover(
        workspace, builtin_root=builtin, user_root=user
    )


def test_discover_no_skills(tmp_path: Path) -> None:
    registry = discover(
        tmp_path / "workspace", builtin=tmp_path / "builtin", user=tmp_path / "user"
    )

    assert registry.list_skills() == []
    assert registry.get_catalog() == ()
    assert registry.warnings == []


def test_discover_valid_skill_and_preserve_unknown_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = write_skill(
        workspace / ".mycode" / "skills",
        "test-skill",
        extra="allowed-tools: Bash(*) Write\nfuture-field:\n  nested: true\n",
    )

    registry = discover(workspace, builtin=tmp_path / "builtin", user=tmp_path / "user")
    skill = registry.require("test-skill")

    assert skill.root == root.resolve()
    assert skill.skill_file == (root / "SKILL.md").resolve()
    assert skill.source == "project"
    assert not hasattr(skill, "body")
    assert skill.allowed_tools == "Bash(*) Write"
    assert skill.frontmatter["future-field"] == {"nested": True}
    assert registry.get_catalog() == (("test-skill", "Use this test Skill."),)


def test_discovery_reads_only_frontmatter_and_load_reads_body_on_demand(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    skill_root = workspace / ".mycode" / "skills" / "lazy-skill"
    skill_root.mkdir(parents=True)
    skill_file = skill_root / "SKILL.md"
    skill_file.write_bytes(
        b"---\nname: lazy-skill\ndescription: Lazy metadata.\n---\n\xffbody"
    )

    registry = discover(workspace, builtin=tmp_path / "builtin", user=tmp_path / "user")

    assert registry.get_catalog() == (("lazy-skill", "Lazy metadata."),)
    assert not hasattr(registry.require("lazy-skill"), "body")
    with pytest.raises(ValueError, match="valid UTF-8"):
        registry.load_instructions("lazy-skill")

    skill_file.write_text(
        "---\nname: lazy-skill\ndescription: Lazy metadata.\n---\nLOADED LATER\n",
        encoding="utf-8",
    )
    assert registry.load_instructions("lazy-skill") == "LOADED LATER\n"


def test_load_instructions_reports_routing_metadata_drift(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    skill_root = write_skill(
        workspace / ".mycode" / "skills",
        "drift-skill",
        description="Original description.",
    )
    registry = discover(
        workspace, builtin=tmp_path / "builtin", user=tmp_path / "user"
    )
    (skill_root / "SKILL.md").write_text(
        "---\nname: drift-skill\ndescription: Changed description.\n---\nBODY\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "SKILL.md name or description changed after discovery; "
            "restart MyCode to rediscover Skills\\."
        ),
    ):
        registry.load_instructions("drift-skill")


@pytest.mark.parametrize(
    ("name_length", "valid"),
    [(64, True), (65, False)],
)
def test_skill_name_length_boundary(
    tmp_path: Path, name_length: int, valid: bool
) -> None:
    workspace = tmp_path / "workspace"
    name = "a" * name_length
    write_skill(workspace / ".mycode" / "skills", name)

    registry = discover(workspace, builtin=tmp_path / "builtin", user=tmp_path / "user")

    assert (registry.get(name) is not None) is valid
    assert bool(registry.warnings) is (not valid)


@pytest.mark.parametrize(
    ("description_length", "valid"),
    [(1024, True), (1025, False)],
)
def test_skill_description_length_boundary(
    tmp_path: Path, description_length: int, valid: bool
) -> None:
    workspace = tmp_path / "workspace"
    write_skill(
        workspace / ".mycode" / "skills",
        "description-skill",
        description="d" * description_length,
    )

    registry = discover(workspace, builtin=tmp_path / "builtin", user=tmp_path / "user")

    assert (registry.get("description-skill") is not None) is valid
    assert bool(registry.warnings) is (not valid)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("---\nname: [\n---\nbody", "Invalid SKILL.md YAML"),
        ("---\ndescription: present\n---\nbody", "requires a non-empty string name"),
        ("---\nname: broken\n---\nbody", "requires a non-empty string description"),
        (
            "---\nname: other-name\ndescription: present\n---\nbody",
            "does not match directory",
        ),
    ],
)
def test_invalid_skill_is_skipped_with_warning(
    tmp_path: Path, contents: str, message: str
) -> None:
    workspace = tmp_path / "workspace"
    skill_root = workspace / ".mycode" / "skills" / "broken"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(contents, encoding="utf-8")

    registry = discover(workspace, builtin=tmp_path / "builtin", user=tmp_path / "user")

    assert registry.list_skills() == []
    assert len(registry.warnings) == 1
    assert message in registry.warnings[0].message


def test_project_overrides_user_and_builtin(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    write_skill(builtin, "shared-skill", description="builtin")
    write_skill(user, "shared-skill", description="user")
    write_skill(
        workspace / ".mycode" / "skills", "shared-skill", description="project"
    )

    registry = discover(workspace, builtin=builtin, user=user)

    assert registry.require("shared-skill").source == "project"
    assert registry.require("shared-skill").description == "project"
    assert registry.get_catalog() == (("shared-skill", "project"),)


def test_invalid_high_priority_skill_keeps_valid_fallback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    write_skill(builtin, "shared-skill", description="fallback")
    invalid = workspace / ".mycode" / "skills" / "shared-skill"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text(
        "---\nname: wrong\ndescription: invalid\n---\n", encoding="utf-8"
    )

    registry = discover(workspace, builtin=builtin, user=user)

    assert registry.require("shared-skill").description == "fallback"
    assert registry.require("shared-skill").source == "builtin"
    assert len(registry.warnings) == 1
    assert registry.warnings[0].source == "project"


def test_register_rejects_duplicate_from_same_source(tmp_path: Path) -> None:
    first = Skill("same", "first", tmp_path, tmp_path / "SKILL.md", "project", {})
    second = Skill("same", "second", tmp_path, tmp_path / "SKILL.md", "project", {})
    registry = SkillRegistry()
    registry.register(first)

    with pytest.raises(DuplicateSkillError):
        registry.register(second)


def test_require_unknown_skill(tmp_path: Path) -> None:
    with pytest.raises(SkillNotFoundError, match="Skill not found"):
        SkillRegistry().require("missing")


@pytest.mark.parametrize("path", ["../outside.txt", "sub/../../outside.txt"])
def test_resource_path_rejects_traversal(tmp_path: Path, path: str) -> None:
    root = tmp_path / "skill"
    skill = Skill("test", "desc", root, root / "SKILL.md", "project", {})

    with pytest.raises(SkillPathError, match="traversal"):
        SkillRegistry().resolve_resource(skill, path)


def test_resource_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "references"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    skill = Skill("test", "desc", root, root / "SKILL.md", "project", {})

    with pytest.raises(SkillPathError, match="escapes"):
        SkillRegistry().resolve_resource(skill, "references/data.txt")
