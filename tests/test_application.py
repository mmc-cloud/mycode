from uuid import uuid4

from mycode.agent.events import AgentEvent
from mycode.application import build_agent_runner, run_agent_turn
from mycode.config import LLMConfig
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
