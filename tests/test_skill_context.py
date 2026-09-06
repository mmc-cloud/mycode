import json
from pathlib import Path

from mycode.agent import AgentModelResponse, AgentToolCall
from mycode.context_budget import ContextBudget, TokenEstimator
from mycode.context_builder import ContextBuilder
from mycode.context_compact import CompactPolicy, ConversationCompactor
from mycode.application import build_agent_runner
from mycode.cli import run_agent_loop
from mycode.config import LLMConfig
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.messages import Message
from mycode.prompts import build_agent_system_prompt
from mycode.runner import AgentRunner
from mycode.skills import ActiveSkillState, Skill, SkillRegistry
from mycode.tools import LoadSkillTool, ToolRegistry
from mycode.tool_result_retention import ToolResultRetentionPolicy


def make_skill(tmp_path: Path, *, body: str = "PRIVATE SKILL PROCEDURE") -> Skill:
    root = tmp_path / "test-skill"
    root.mkdir()
    skill_file = root / "SKILL.md"
    skill_file.write_text(
        "---\nname: test-skill\ndescription: Use for tests.\n---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return Skill(
        "test-skill", "Use for tests.", root, skill_file, "project", {}
    )


def budget(max_input_tokens: int = 4000) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )


def test_catalog_contains_only_name_and_description(tmp_path: Path) -> None:
    prompt = build_agent_system_prompt(
        skill_catalog=(("test-skill", "Use for tests."),)
    )
    empty_prompt = build_agent_system_prompt()

    assert "<available_skills>" in prompt
    assert '"test-skill": "Use for tests."' in prompt
    assert "应先调用 load_skill 加载对应专项指导" in prompt
    assert "不要在同一轮继续执行依赖该 Skill 指导的操作" in prompt
    assert "PRIVATE SKILL PROCEDURE" not in prompt
    assert "<available_skills>" not in empty_prompt


def test_persistent_system_context_is_budgeted_without_mutating_canonical(
    tmp_path: Path,
) -> None:
    skill = make_skill(tmp_path, body="x" * 400)
    state = ActiveSkillState()
    state.activate(skill, "x" * 400)
    canonical = Conversation.from_messages(
        [Message(role="system", content="core"), Message(role="user", content="task")]
    )
    builder = ContextBuilder(budget=budget(), token_estimator=TokenEstimator())

    without_skill = builder.build(canonical).context
    with_skill = builder.build(
        canonical,
        persistent_system_messages=tuple(
            Message(role="system", content=content)
            for content in state.to_system_contexts()
        ),
    ).context

    assert with_skill.estimate.estimated_input_tokens > without_skill.estimate.estimated_input_tokens
    assert any("PRIVATE" not in message.content and "x" * 100 in message.content for message in with_skill.messages)
    assert canonical.get_messages() == [
        Message(role="system", content="core"),
        Message(role="user", content="task"),
    ]


def test_tool_result_retention_does_not_downgrade_skill_context(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, body="UNCOMPRESSED SKILL BODY " * 300)
    state = ActiveSkillState()
    state.activate(skill, "UNCOMPRESSED SKILL BODY " * 300)
    context_budget = budget(10000)
    result = ContextBuilder(
        budget=context_budget,
        token_estimator=TokenEstimator(),
        retention_policy=ToolResultRetentionPolicy(context_budget, None),
    ).build(
        Conversation.from_messages([Message(role="user", content="task")]),
        persistent_system_messages=tuple(
            Message(role="system", content=content)
            for content in state.to_system_contexts()
        ),
    )

    skill_messages = [
        message for message in result.context.messages if "UNCOMPRESSED SKILL BODY" in message.content
    ]
    assert len(skill_messages) == 1
    assert skill_messages[0].content.count("UNCOMPRESSED SKILL BODY") == 300


class RecordingCompactClient(FakeLLMClient):
    def __init__(self, response: str) -> None:
        super().__init__(responses=[response])
        self.seen_messages = []

    def complete(self, conversation):
        self.seen_messages.append(conversation.get_messages())
        return super().complete(conversation)


def test_active_skill_survives_compact_but_is_not_summarized(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, body="SKILL BODY " * 50)
    state = ActiveSkillState()
    state.activate(skill, "SKILL BODY " * 50)
    messages = [Message(role="system", content="core")]
    for index in range(4):
        messages.extend(
            [
                Message(role="user", content=f"old request {index} " * 20),
                Message(role="assistant", content=f"old answer {index} " * 20),
            ]
        )
    canonical = Conversation.from_messages(messages)
    summary = json.dumps(
        {
            "objective": "test",
            "progress": ["history summarized"],
            "decisions": [],
            "constraints": [],
            "open_items": [],
            "references": [],
        }
    )
    compact_client = RecordingCompactClient(summary)
    compactor = ConversationCompactor(
        llm_client=compact_client,
        policy=CompactPolicy(trigger_ratio=0.1, recent_turns_to_keep=1),
    )

    result = ContextBuilder(
        budget=budget(2200),
        token_estimator=TokenEstimator(),
        compactor=compactor,
    ).build(
        canonical,
        persistent_system_messages=tuple(
            Message(role="system", content=content)
            for content in state.to_system_contexts()
        ),
    )

    assert result.context.compact_stats is not None
    assert result.context.compact_stats.status == "compacted"
    assert any("SKILL BODY" in message.content for message in result.context.messages)
    assert all(
        "SKILL BODY" not in message.content
        for message in compact_client.seen_messages[0]
    )
    assert all("SKILL BODY" not in message.content for message in canonical.get_messages())


class RecordingSkillClient(FakeLLMClient):
    def __init__(self, tool_responses):
        super().__init__(responses=[], tool_responses=tool_responses)
        self.seen_messages = []

    def stream_with_tools(self, conversation, tools):
        self.seen_messages.append(conversation.get_messages())
        yield from super().stream_with_tools(conversation, tools)


def test_runner_skill_lifecycle_and_observability_are_task_scoped(tmp_path: Path) -> None:
    skill = make_skill(tmp_path)
    registry = SkillRegistry()
    registry.register(skill)
    state = ActiveSkillState()
    tools = ToolRegistry.from_tools([LoadSkillTool(registry, state)])
    client = RecordingSkillClient(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        id="load-1", name="load_skill", arguments={"name": skill.name}
                    ),
                ),
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="first done"),
            AgentModelResponse(content="second done"),
        ]
    )
    observations = []
    runner = AgentRunner(
        llm_client=client,
        tool_registry=tools,
        active_skill_state=state,
        observability_sink=observations.append,
    )

    list(runner.run("first task"))
    assert state.get_active() == ()
    list(runner.run("second task"))

    assert not any("PRIVATE SKILL PROCEDURE" in m.content for m in client.seen_messages[0])
    assert any("PRIVATE SKILL PROCEDURE" in m.content for m in client.seen_messages[1])
    assert not any("PRIVATE SKILL PROCEDURE" in m.content for m in client.seen_messages[2])
    assert all(
        "PRIVATE SKILL PROCEDURE" not in m.content
        for m in runner.conversation.get_messages()
    )
    snapshots = [record for record in observations if record["event_type"] == "context_snapshot"]
    assert [(s["active_skill_count"], s["active_skill_names"]) for s in snapshots] == [
        (0, []),
        (1, ["test-skill"]),
        (0, []),
    ]
    assert all("PRIVATE SKILL PROCEDURE" not in str(snapshot) for snapshot in snapshots)


def test_build_agent_runner_wires_catalog_tools_and_shared_state(
    tmp_path: Path, monkeypatch
) -> None:
    skill = make_skill(tmp_path)
    registry = SkillRegistry()
    registry.register(skill)
    monkeypatch.setattr(
        "mycode.application.SkillRegistry.discover", lambda workspace_root: registry
    )
    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
        ),
    )

    tools = {tool.name: tool for tool in runner.tool_registry.list_tools()}
    assert {"load_skill", "read_skill_resource", "run_skill_script"} <= tools.keys()
    assert runner.active_skill_state is tools["load_skill"].state
    assert runner.active_skill_state is tools["read_skill_resource"].state
    assert runner.active_skill_state is tools["run_skill_script"].state
    assert "完整 Skill 指导会从下一轮模型调用开始可见" in tools["load_skill"].description
    system_prompt = runner.conversation.get_messages()[0].content
    assert '"test-skill": "Use for tests."' in system_prompt
    assert "PRIVATE SKILL PROCEDURE" not in system_prompt


def test_build_agent_runner_omits_catalog_and_tools_without_skills(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mycode.application.SkillRegistry.discover",
        lambda workspace_root: SkillRegistry(),
    )
    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
        ),
    )

    tool_names = {tool.name for tool in runner.tool_registry.list_tools()}
    assert "load_skill" not in tool_names
    assert "read_skill_resource" not in tool_names
    assert "run_skill_script" not in tool_names
    assert "<available_skills>" not in runner.conversation.get_messages()[0].content


def test_cli_displays_skill_discovery_warning() -> None:
    class WarningRunner:
        instruction_sources = ()
        instruction_warnings = ()
        skill_warnings = ("project:broken: invalid YAML",)

        def run(self, user_message):
            raise AssertionError("runner must not start")
            yield

    outputs = []
    run_agent_loop(
        WarningRunner(),
        input_func=lambda prompt: "/quit",
        output_func=outputs.append,
    )

    assert outputs == [
        "skills> 警告：project:broken: invalid YAML",
        "输入 /exit 或 /quit 退出。",
    ]


def test_runner_clears_skill_state_when_generator_is_closed(tmp_path: Path) -> None:
    skill = make_skill(tmp_path)
    state = ActiveSkillState()
    registry = SkillRegistry()
    registry.register(skill)
    client = RecordingSkillClient(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        id="load-1", name="load_skill", arguments={"name": skill.name}
                    ),
                ),
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="unused"),
        ]
    )
    runner = AgentRunner(
        llm_client=client,
        tool_registry=ToolRegistry.from_tools([LoadSkillTool(registry, state)]),
        active_skill_state=state,
    )
    run = runner.run("task")
    for event in run:
        if state.is_active(skill.name):
            break
    assert state.is_active(skill.name)

    run.close()

    assert state.get_active() == ()
