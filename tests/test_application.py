from uuid import uuid4
from types import SimpleNamespace

import pytest

from mycode.agent.events import AgentEvent
from mycode.application import build_agent_runner, run_agent_turn
from mycode.config import LLMConfig
from mycode.model_config import APIConnectionConfig, ModelConfiguration
from mycode.model_selection import ModelSelection, RoleSelection
from mycode.agent.outcome import AgentRunOutcome
from mycode.agent.runner import AgentRunner


class EventRunner:
    def __init__(self, events: list[AgentEvent]) -> None:
        self.events = events
        self.seen_content: list[str] = []

    def run(self, content: str):
        self.seen_content.append(content)
        yield from self.events


def test_build_agent_runner_returns_runtime_for_workspace(tmp_path) -> None:
    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
        ),
    )

    assert isinstance(runner, AgentRunner)
    assert runner.tool_registry is not None


def _runner_llm_session_ids(runner: AgentRunner) -> list[str | None]:
    delegate_tool = runner.tool_registry.require("delegate_task")
    subagent_client = delegate_tool.runtime.llm_client_factory()
    return [
        runner.llm_client.session_id,
        runner.compactor.llm_client.session_id,
        subagent_client.session_id,
    ]


def test_build_agent_runner_shares_explicit_llm_session_id(tmp_path) -> None:
    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(
            api_key="test-key",
            base_url="https://opencode.ai/zen/go/v1",
            model="test-model",
        ),
        llm_session_id="session-123",
    )

    assert _runner_llm_session_ids(runner) == ["session-123"] * 3


def test_build_agent_runner_generates_one_shared_fallback_session_id(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def counted_uuid4():
        nonlocal calls
        calls += 1
        return uuid4()

    monkeypatch.setattr("mycode.application.runtime.uuid4", counted_uuid4)

    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(
            api_key="test-key",
            base_url="https://opencode.ai/zen/go/v1",
            model="test-model",
        ),
    )

    session_ids = _runner_llm_session_ids(runner)
    assert calls == 1
    assert session_ids[0]
    assert session_ids == [session_ids[0]] * 3


def test_run_agent_turn_forwards_events_and_returns_outcome() -> None:
    events = [
        AgentEvent(type="text_delta", content="done"),
        AgentEvent(type="stop", stop_reason="final_answer"),
    ]
    runner = EventRunner(events)
    handled: list[AgentEvent] = []

    outcome = run_agent_turn(runner, "inspect", event_handler=handled.append)

    assert runner.seen_content == ["inspect"]
    assert handled == events
    assert outcome == AgentRunOutcome.from_stop_reason("final_answer")


def test_run_agent_turn_without_exactly_one_stop_is_runtime_failure() -> None:
    no_stop = EventRunner([AgentEvent(type="text_delta", content="partial")])
    repeated_stop = EventRunner(
        [
            AgentEvent(type="stop", stop_reason="final_answer"),
            AgentEvent(type="stop", stop_reason="max_turns"),
        ]
    )

    assert run_agent_turn(no_stop, "inspect") == AgentRunOutcome.from_stop_reason(None)
    assert run_agent_turn(repeated_stop, "inspect") == AgentRunOutcome.from_stop_reason(
        None
    )


def test_factory_keeps_eager_main_compact_and_fresh_deferred_subagents(tmp_path, monkeypatch):
    created = []

    def client(**kwargs):
        result = SimpleNamespace(**kwargs, model=kwargs["config"].model)
        created.append(result)
        return result

    monkeypatch.setattr("mycode.providers.factory.OpenAICompatibleLLMClient", client)
    runner = build_agent_runner(
        workspace_path=tmp_path,
        llm_config=LLMConfig(api_key="test", base_url="https://example.com", model="main",
                             compact_model="compact", subagent_model="subagent"),
        llm_session_id="shared-session",
    )
    assert [item.model for item in created] == ["main", "compact"]
    runtime = runner.tool_registry.require("delegate_task").runtime
    first = runtime.llm_client_factory()
    second = runtime.llm_client_factory()
    assert first is not second
    assert [item.model for item in created] == ["main", "compact", "subagent", "subagent"]
    assert {item.session_id for item in created} == {"shared-session"}


def test_runner_shares_connection_with_independent_models_and_budgets(tmp_path):
    models = ModelConfiguration(connections={"main": APIConnectionConfig("main-key", "https://main.example/v1", {"X-Shared": "shared-header"})},
        selection=ModelSelection(main=RoleSelection(connection_ref="main", model="main", thinking="on", effort="max"),
            subagent=RoleSelection(model="sub", thinking="on", effort="high", context_window_tokens=16000,
                reserved_output_tokens=2000, context_safety_margin_tokens=1000),
            compact=RoleSelection(model="summary", context_window_tokens=32000,
                reserved_output_tokens=4000, context_safety_margin_tokens=2000)))
    runner = build_agent_runner(workspace_path=tmp_path, model_configuration=models)
    runtime = runner.tool_registry.require("delegate_task").runtime
    assert runner.context_budget.context_window_tokens == 128000
    assert runtime.context_budget.max_input_tokens == 13000
    assert runner.compactor.summary_budget.max_input_tokens == 26000
    subagent = runtime.llm_client_factory()
    for client in (runner.llm_client, subagent, runner.compactor.llm_client):
        assert client.config.base_url == "https://main.example/v1"
        assert client.config.api_key == "main-key"
        assert client.config.headers == {"X-Shared": "shared-header"}
        assert client.session_id == runner.llm_client.session_id
    assert runner.llm_client.reasoning_effort == "max"
    assert subagent.model == "sub"
    assert subagent.reasoning_effort == "high"
    assert runner.compactor.llm_client.model == "summary"
    assert runner.compactor.llm_client.thinking_enabled is False


def test_invalid_role_config_fails_before_any_client_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mycode.providers.factory.OpenAICompatibleLLMClient",
        lambda **kwargs: pytest.fail("No client should be created for invalid role config"),
    )
    models = ModelConfiguration(connections={"main": APIConnectionConfig("test", "https://example.com")},
        selection=ModelSelection(main=RoleSelection(connection_ref="main", model="main"),
            subagent=RoleSelection(model="different", context_window_tokens=10)))
    with pytest.raises(ValueError, match="reserves"):
        build_agent_runner(workspace_path=tmp_path, model_configuration=models)
