from pathlib import Path

import pytest

from mycode.confirmers import TerminalConfirmer
from mycode.permissions import ConfirmationResult
from mycode.skills import MAX_ACTIVE_SKILLS, ActiveSkillState, Skill, SkillRegistry
from mycode.tools import (
    LoadSkillTool,
    ReadSkillResourceTool,
    RunSkillScriptTool,
    ToolRegistry,
    Workspace,
)


class ApprovingConfirmer:
    def __init__(self) -> None:
        self.requests = []

    def confirm(self, request):
        self.requests.append(request)
        return ConfirmationResult.approved("approved")


class RejectingConfirmer:
    def __init__(self) -> None:
        self.requests = []

    def confirm(self, request):
        self.requests.append(request)
        return ConfirmationResult.rejected("rejected")


def skill_setup(tmp_path: Path, *, allowed_tools: object | None = None):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    skill_root = tmp_path / "test-skill"
    (skill_root / "references").mkdir(parents=True)
    (skill_root / "scripts").mkdir()
    frontmatter = {} if allowed_tools is None else {"allowed-tools": allowed_tools}
    skill_file = skill_root / "SKILL.md"
    skill_file.write_text(
        "---\nname: test-skill\ndescription: test\n---\nSECRET BODY\n",
        encoding="utf-8",
    )
    skill = Skill(
        "test-skill",
        "test",
        skill_root.resolve(),
        skill_file.resolve(),
        "project",
        frontmatter,
    )
    registry = SkillRegistry()
    registry.register(skill)
    state = ActiveSkillState()
    return Workspace(workspace_root), skill, registry, state


def test_load_skill_activation_unknown_and_idempotence(tmp_path: Path) -> None:
    _workspace, skill, registry, state = skill_setup(tmp_path)
    tools = ToolRegistry.from_tools([LoadSkillTool(registry, state)])

    missing = tools.run_tool("load_skill", {"name": "missing"})
    first = tools.run_tool("load_skill", {"name": skill.name})
    second = tools.run_tool("load_skill", {"name": skill.name})

    assert not missing.ok
    assert first.ok and second.ok
    assert "SECRET BODY" not in first.content
    assert first.metadata == {
        "skill_name": "test-skill",
        "skill_source": "project",
        "already_active": False,
    }
    assert second.metadata["already_active"] is True
    assert len(state.get_active()) == 1
    assert state.get_active()[0].instructions == "SECRET BODY\n"
    assert tools.is_concurrency_safe("load_skill") is False


def test_load_skill_limits_new_activations_but_keeps_duplicates_idempotent(
    tmp_path: Path,
) -> None:
    registry = SkillRegistry()
    state = ActiveSkillState()
    for index in range(MAX_ACTIVE_SKILLS + 1):
        name = f"skill-{index}"
        root = tmp_path / name
        root.mkdir()
        skill_file = root / "SKILL.md"
        skill_file.write_text(
            f"---\nname: {name}\ndescription: test\n---\nBODY {index}\n",
            encoding="utf-8",
        )
        registry.register(
            Skill(name, "test", root, skill_file, "project", {})
        )
    tools = ToolRegistry.from_tools([LoadSkillTool(registry, state)])

    first_results = [
        tools.run_tool("load_skill", {"name": f"skill-{index}"})
        for index in range(MAX_ACTIVE_SKILLS)
    ]
    duplicate = tools.run_tool("load_skill", {"name": "skill-0"})
    sixth_skill = registry.require(f"skill-{MAX_ACTIVE_SKILLS}")
    sixth_skill.skill_file.write_bytes(
        f"---\nname: {sixth_skill.name}\ndescription: test\n---\n".encode()
        + b"\xffbroken body"
    )
    sixth = tools.run_tool("load_skill", {"name": f"skill-{MAX_ACTIVE_SKILLS}"})

    assert all(result.ok for result in first_results)
    assert duplicate.ok and duplicate.metadata["already_active"] is True
    assert not sixth.ok
    assert sixth.error == f"最多只能同时激活 {MAX_ACTIVE_SKILLS} 个 Skill。"
    assert sixth.metadata["active_skill_count"] == MAX_ACTIVE_SKILLS
    assert sixth.metadata["max_active_skills"] == MAX_ACTIVE_SKILLS
    assert len(state.get_active()) == MAX_ACTIVE_SKILLS


def test_read_active_skill_resource_is_bounded(tmp_path: Path) -> None:
    _workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "references" / "guide.md").write_text("0123456789", encoding="utf-8")
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools([ReadSkillResourceTool(registry, state)])

    result = tools.run_tool(
        "read_skill_resource",
        {"skill": skill.name, "path": "references/guide.md", "offset_chars": 3, "max_chars": 4},
    )

    assert result.ok
    assert result.content == "3456"
    assert result.metadata["has_more"] is True
    assert result.metadata["next_offset_chars"] == 7
    assert result.metadata["skill_source"] == "project"


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("missing.md", "not found"),
        ("references", "not a file"),
        ("../outside.md", "traversal"),
    ],
)
def test_read_skill_resource_rejects_invalid_paths(
    tmp_path: Path, path: str, error: str
) -> None:
    _workspace, skill, registry, state = skill_setup(tmp_path)
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools([ReadSkillResourceTool(registry, state)])

    result = tools.run_tool(
        "read_skill_resource", {"skill": skill.name, "path": path}
    )

    assert not result.ok
    assert error in (result.error or "")


def test_read_skill_resource_rejects_inactive_absolute_and_binary(tmp_path: Path) -> None:
    _workspace, skill, registry, state = skill_setup(tmp_path)
    binary = skill.root / "references" / "binary.bin"
    binary.write_bytes(b"abc\x00def")
    tools = ToolRegistry.from_tools([ReadSkillResourceTool(registry, state)])

    inactive = tools.run_tool(
        "read_skill_resource", {"skill": skill.name, "path": "references/binary.bin"}
    )
    state.activate(skill, "SECRET BODY")
    absolute = tools.run_tool(
        "read_skill_resource", {"skill": skill.name, "path": str(binary.resolve())}
    )
    binary_result = tools.run_tool(
        "read_skill_resource", {"skill": skill.name, "path": "references/binary.bin"}
    )

    assert not inactive.ok and "not active" in (inactive.error or "")
    assert not absolute.ok and "Absolute" in (absolute.error or "")
    assert not binary_result.ok and "UTF-8 text" in (binary_result.error or "")


def test_run_skill_script_requires_activation_and_confirmation(tmp_path: Path) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    script = skill.root / "scripts" / "show.py"
    script.write_text(
        "from pathlib import Path\nimport sys\nprint(Path.cwd())\nprint(sys.argv[1])\n",
        encoding="utf-8",
    )
    rejecting = RejectingConfirmer()
    tool = RunSkillScriptTool(workspace, registry, state)
    tools = ToolRegistry.from_tools([tool], confirmer=rejecting)

    inactive = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "show.py", "args": ["ok"]}
    )
    state.activate(skill, "SECRET BODY")
    rejected = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "show.py", "args": ["ok"]}
    )

    assert not inactive.ok and "not active" in (inactive.error or "")
    assert len(rejecting.requests) == 1
    assert not rejected.ok
    assert rejected.metadata["confirmation_status"] == "rejected"
    request = rejecting.requests[0].permission_request
    assert request.target == "test-skill:scripts/show.py"
    assert 'Skill "test-skill" 请求在 workspace 中执行' in request.description
    assert 'Skill "test-skill" 请求执行' in rejecting.requests[0].prompt
    confirmation = rejecting.requests[0]
    assert confirmation.metadata["skill_name"] == "test-skill"
    assert confirmation.metadata["skill_source"] == "project"
    assert confirmation.metadata["script"] == "scripts/show.py"
    assert confirmation.metadata["arguments"] == ["ok"]
    assert confirmation.metadata["cwd"] == str(workspace.root)


def test_run_skill_script_terminal_confirmation_displays_skill_details(
    tmp_path: Path,
) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "scripts" / "show.py").write_text("print('ran')", encoding="utf-8")
    state.activate(skill, "SECRET BODY")
    outputs: list[str] = []
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=TerminalConfirmer(
            input_func=lambda prompt: "n", output_func=outputs.append
        ),
    )

    result = tools.run_tool(
        "run_skill_script",
        {"skill": skill.name, "script": "show.py", "args": ["database.db", "--safe"]},
    )

    assert not result.ok
    assert "skill_name> test-skill" in outputs
    assert "skill_source> project" in outputs
    assert "script> scripts/show.py" in outputs
    assert "arguments> ['database.db', '--safe']" in outputs
    assert f"cwd> {workspace.root}" in outputs


def test_run_skill_script_executes_in_workspace_and_captures_output(tmp_path: Path) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "scripts" / "show.py").write_text(
        "from pathlib import Path\nimport sys\nprint(Path.cwd())\nprint(sys.argv[1])\nprint('warning', file=sys.stderr)\n",
        encoding="utf-8",
    )
    state.activate(skill, "SECRET BODY")
    confirmer = ApprovingConfirmer()
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)], confirmer=confirmer
    )

    result = tools.run_tool(
        "run_skill_script",
        {"skill": skill.name, "script": "scripts/show.py", "args": ["hello"]},
    )

    assert result.ok
    assert str(workspace.root) in result.content
    assert "hello" in result.content
    assert "warning" in result.content
    assert result.metadata["exit_code"] == 0
    assert result.metadata["skill_name"] == skill.name
    assert result.metadata["script"] == "scripts/show.py"
    assert result.metadata["permission_status"] == "allow"


def test_run_skill_script_nonzero_timeout_and_truncation(tmp_path: Path) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "scripts" / "fail.py").write_text(
        "import sys\nprint('failure output')\nsys.exit(3)\n", encoding="utf-8"
    )
    (skill.root / "scripts" / "slow.py").write_text(
        "import time\nprint('before timeout', flush=True)\ntime.sleep(2)\n", encoding="utf-8"
    )
    (skill.root / "scripts" / "long.py").write_text(
        "print('x' * 1000)\n", encoding="utf-8"
    )
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=ApprovingConfirmer(),
    )

    failed = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "fail.py"}
    )
    timed_out = tools.run_tool(
        "run_skill_script",
        {"skill": skill.name, "script": "slow.py", "timeout_seconds": 0.05},
    )
    truncated = tools.run_tool(
        "run_skill_script",
        {"skill": skill.name, "script": "long.py", "max_output_chars": 20},
    )

    assert not failed.ok and failed.metadata["exit_code"] == 3
    assert "failure output" in (failed.error or "")
    assert not timed_out.ok and timed_out.metadata["timed_out"] is True
    assert truncated.ok and truncated.metadata["stdout_truncated"] is True


@pytest.mark.parametrize(
    "script", ["../outside.py", "references/no.py", "C:/outside.py"]
)
def test_run_skill_script_rejects_outside_scripts(tmp_path: Path, script: str) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "references" / "no.py").write_text("print('no')", encoding="utf-8")
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=ApprovingConfirmer(),
    )

    result = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": script}
    )

    assert not result.ok


def test_run_skill_script_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')", encoding="utf-8")
    link = skill.root / "scripts" / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=ApprovingConfirmer(),
    )

    result = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "linked.py"}
    )

    assert not result.ok
    assert "escapes" in (result.error or "")


def test_run_skill_script_rejects_unsupported_or_missing_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    workspace, skill, registry, state = skill_setup(tmp_path)
    (skill.root / "scripts" / "unknown.rb").write_text("puts 'no'", encoding="utf-8")
    (skill.root / "scripts" / "known.js").write_text("console.log('no')", encoding="utf-8")
    state.activate(skill, "SECRET BODY")
    monkeypatch.setattr("mycode.tools.run_skill_script.shutil.which", lambda name: None)
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=ApprovingConfirmer(),
    )

    unsupported = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "unknown.rb"}
    )
    missing = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "known.js"}
    )

    assert not unsupported.ok and "Unsupported" in (unsupported.error or "")
    assert not missing.ok and "runtime is unavailable" in (missing.error or "")


def test_allowed_tools_metadata_does_not_bypass_permission(tmp_path: Path) -> None:
    workspace, skill, registry, state = skill_setup(
        tmp_path, allowed_tools="Bash(*) Write Edit"
    )
    (skill.root / "scripts" / "show.py").write_text("print('ran')", encoding="utf-8")
    state.activate(skill, "SECRET BODY")
    tools = ToolRegistry.from_tools(
        [RunSkillScriptTool(workspace, registry, state)],
        confirmer=RejectingConfirmer(),
    )

    result = tools.run_tool(
        "run_skill_script", {"skill": skill.name, "script": "show.py"}
    )

    assert not result.ok
    assert result.metadata["confirmation_status"] == "rejected"
    assert "ran" not in (result.content or "")
