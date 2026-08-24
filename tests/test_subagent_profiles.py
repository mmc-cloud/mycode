from dataclasses import replace
from pathlib import Path

import pytest

from mycode.subagents.profiles import (
    BUILTIN_AGENT_PROFILES,
    EXPLORER_PROFILE,
    REVIEWER_PROFILE,
    TESTER_PROFILE,
    ProfileToolConfigurationError,
    create_subagent_tool_registry,
)
from mycode.subagents.prompts import build_subagent_system_prompt
from mycode.tools import Workspace


def test_builtin_profiles_define_three_distinct_roles() -> None:
    assert tuple(BUILTIN_AGENT_PROFILES) == ("explorer", "tester", "reviewer")
    assert EXPLORER_PROFILE.submission_model.__name__ == "ExplorerResult"
    assert TESTER_PROFILE.submission_model.__name__ == "TesterReport"
    assert REVIEWER_PROFILE.submission_model.__name__ == "ReviewerResult"
    assert "停止扩大搜索范围" in EXPLORER_PROFILE.convergence_prompt
    assert "最关键且尚未执行的验证" in TESTER_PROFILE.convergence_prompt
    assert "高优先级 findings" in REVIEWER_PROFILE.convergence_prompt
    assert all(
        "submit_result" in profile.convergence_prompt
        for profile in BUILTIN_AGENT_PROFILES.values()
    )


@pytest.mark.parametrize(
    ("profile", "expected_names"),
    [
        (EXPLORER_PROFILE, ("read_file", "glob", "grep", "submit_result")),
        (
            TESTER_PROFILE,
            ("read_file", "glob", "grep", "run_validation", "submit_result"),
        ),
        (
            REVIEWER_PROFILE,
            ("read_file", "glob", "grep", "inspect_changes", "submit_result"),
        ),
    ],
)
def test_profile_registry_matches_declared_tools(
    tmp_path: Path,
    profile,
    expected_names: tuple[str, ...],
) -> None:
    registry = create_subagent_tool_registry(profile, Workspace(tmp_path))

    assert tuple(tool.name for tool in registry.list_tools()) == expected_names
    assert tuple(schema["name"] for schema in registry.get_schemas()) == expected_names


def test_profile_registries_apply_distinct_command_boundaries(tmp_path: Path) -> None:
    tester = create_subagent_tool_registry(TESTER_PROFILE, Workspace(tmp_path))
    reviewer = create_subagent_tool_registry(REVIEWER_PROFILE, Workspace(tmp_path))

    assert tester.require("run_validation").get_permission_profile().capability == "command"
    assert reviewer.require("inspect_changes").get_permission_profile().capability == "read"
    assert tester.get("run_command") is None
    assert reviewer.get("run_command") is None
    assert tester.require("submit_result").get_permission_profile().capability == "control"


def test_profile_registry_rejects_declared_tool_drift(tmp_path: Path) -> None:
    mismatched = replace(
        EXPLORER_PROFILE,
        tool_names=("read_file", "submit_result"),
    )

    with pytest.raises(ProfileToolConfigurationError, match="contract mismatch"):
        create_subagent_tool_registry(mismatched, Workspace(tmp_path))


def test_profile_rejects_submission_model_for_another_role() -> None:
    with pytest.raises(ValueError, match="submission_model does not match"):
        replace(EXPLORER_PROFILE, submission_model=REVIEWER_PROFILE.submission_model)


@pytest.mark.parametrize(
    "profile",
    [EXPLORER_PROFILE, TESTER_PROFILE, REVIEWER_PROFILE],
)
def test_subagent_prompt_lists_registry_tools_and_success_contract(
    tmp_path: Path,
    profile,
) -> None:
    registry = create_subagent_tool_registry(profile, Workspace(tmp_path))

    prompt = build_subagent_system_prompt(profile, registry)

    for tool_name in profile.tool_names:
        assert f"- {tool_name}：" in prompt
    assert "必须调用 submit_result" in prompt
    assert "必须单独出现在一次工具调用响应中" in prompt
    assert "不能创建其他 SubAgent" in prompt


def test_reviewer_prompt_does_not_advertise_command_execution(tmp_path: Path) -> None:
    registry = create_subagent_tool_registry(REVIEWER_PROFILE, Workspace(tmp_path))

    prompt = build_subagent_system_prompt(REVIEWER_PROFILE, registry)

    assert "- inspect_changes：" in prompt
    assert "- run_validation：" not in prompt
    assert "- run_command：" not in prompt


def test_tester_prompt_does_not_advertise_change_inspection(tmp_path: Path) -> None:
    registry = create_subagent_tool_registry(TESTER_PROFILE, Workspace(tmp_path))

    prompt = build_subagent_system_prompt(TESTER_PROFILE, registry)

    assert "- run_validation：" in prompt
    assert "- inspect_changes：" not in prompt


def test_subagent_prompt_embeds_frozen_project_instruction_snapshot(
    tmp_path: Path,
) -> None:
    registry = create_subagent_tool_registry(EXPLORER_PROFILE, Workspace(tmp_path))

    prompt = build_subagent_system_prompt(
        EXPLORER_PROFILE,
        registry,
        project_instructions="Use current project rules.",
    )

    assert "本次 SubAgent 启动时生成的只读快照" in prompt
    assert "<project_instructions>" in prompt
    assert "Use current project rules." in prompt


def test_subagent_prompt_rejects_profile_registry_mismatch(tmp_path: Path) -> None:
    tester_registry = create_subagent_tool_registry(
        TESTER_PROFILE,
        Workspace(tmp_path),
    )

    with pytest.raises(ProfileToolConfigurationError, match="Prompt tool contract mismatch"):
        build_subagent_system_prompt(EXPLORER_PROFILE, tester_registry)
