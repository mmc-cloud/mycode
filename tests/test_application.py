from mycode.agent import AgentEvent
from mycode.application import build_agent_runner, run_agent_turn
from mycode.config import LLMConfig
from mycode.run_outcome import AgentRunOutcome
from mycode.runner import AgentRunner


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
