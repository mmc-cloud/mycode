from importlib.resources import files
from pathlib import Path

from mycode.skills import ActiveSkillState, SkillRegistry
from mycode.tools import LoadSkillTool, ReadSkillResourceTool, ToolRegistry


DATABASE_RECOVERY_DESCRIPTION = (
    "用于数据库损坏、异常中断、文件截断、transaction log/WAL/journal 异常、"
    "corruption、salvage、restore 或 recover 等需要恢复或抢救数据的任务。"
    "提供原始证据保护、工作副本、数据库专用恢复、结果验证和不确定性处理流程。"
)


def discover_builtin(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry.discover(
        tmp_path / "workspace", user_root=tmp_path / "user-skills"
    )


def test_database_recovery_builtin_discovery_exposes_only_metadata(
    tmp_path: Path,
) -> None:
    registry = discover_builtin(tmp_path)
    skill = registry.require("database-recovery")

    assert skill.source == "builtin"
    assert skill.description == DATABASE_RECOVERY_DESCRIPTION
    assert (skill.name, skill.description) in registry.get_catalog()
    assert "# Database Recovery" not in str(registry.get_catalog())
    assert not hasattr(skill, "body")


def test_database_recovery_body_is_loaded_only_on_activation(
    tmp_path: Path, monkeypatch
) -> None:
    read_paths: list[Path] = []
    original_read_text = Path.read_text

    def record_read_text(path: Path, *args, **kwargs) -> str:
        read_paths.append(path.resolve())
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read_text)
    registry = discover_builtin(tmp_path)
    skill = registry.require("database-recovery")

    assert skill.skill_file not in read_paths

    state = ActiveSkillState()
    result = ToolRegistry.from_tools(
        [LoadSkillTool(registry, state)]
    ).run_tool("load_skill", {"name": skill.name})

    assert result.ok
    assert read_paths == [skill.skill_file]
    assert "# Database Recovery" in state.get_active()[0].instructions
    assert "先保护证据，再进行恢复" in state.to_system_contexts()[0]


def test_database_recovery_sqlite_reference_is_read_separately(
    tmp_path: Path,
) -> None:
    registry = discover_builtin(tmp_path)
    state = ActiveSkillState()
    tools = ToolRegistry.from_tools(
        [
            LoadSkillTool(registry, state),
            ReadSkillResourceTool(registry, state),
        ]
    )

    loaded = tools.run_tool("load_skill", {"name": "database-recovery"})
    reference = tools.run_tool(
        "read_skill_resource",
        {
            "skill": "database-recovery",
            "path": "references/sqlite.md",
            "max_chars": 20000,
        },
    )

    assert loaded.ok
    assert reference.ok
    assert "# SQLite Recovery Reference" in reference.content
    assert "<database>-wal" in reference.content
    assert "# SQLite Recovery Reference" not in state.get_active()[0].instructions


def test_database_recovery_markdown_files_are_package_resources() -> None:
    skill_root = (
        files("mycode.skills")
        .joinpath("builtin")
        .joinpath("database-recovery")
    )
    skill_file = skill_root.joinpath("SKILL.md")
    sqlite_reference = skill_root.joinpath("references").joinpath("sqlite.md")

    assert skill_file.is_file()
    assert sqlite_reference.is_file()
    assert "# Database Recovery" in skill_file.read_text(encoding="utf-8")
    assert "# SQLite Recovery Reference" in sqlite_reference.read_text(
        encoding="utf-8"
    )
