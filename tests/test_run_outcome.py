import pytest

from mycode.run_outcome import AgentRunOutcome


@pytest.mark.parametrize("stop_reason", ["final_answer", "control_tool"])
def test_completed_stop_reasons_use_exit_zero(stop_reason) -> None:
    outcome = AgentRunOutcome.from_stop_reason(stop_reason)
    assert outcome.status == "completed"
    assert outcome.exit_code == 0


@pytest.mark.parametrize(
    "stop_reason",
    ["max_turns", "repeated_tool_call"],
)
def test_incomplete_stop_reasons_use_exit_two(stop_reason) -> None:
    outcome = AgentRunOutcome.from_stop_reason(stop_reason)
    assert outcome.status == "task_incomplete"
    assert outcome.exit_code == 2


@pytest.mark.parametrize(
    "stop_reason",
    [None, "model_error", "tool_error", "context_overflow", "tool_calls"],
)
def test_runtime_failure_stop_reasons_use_exit_one(stop_reason) -> None:
    outcome = AgentRunOutcome.from_stop_reason(stop_reason)
    assert outcome.status == "runtime_failure"
    assert outcome.exit_code == 1


def test_run_outcome_json_round_trip() -> None:
    outcome = AgentRunOutcome.from_stop_reason("max_turns")
    assert AgentRunOutcome.from_json(outcome.to_json()) == outcome


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"completed","stop_reason":"final_answer","extra":true}',
        '{"status":"runtime_failure","stop_reason":"unknown"}',
        '{"status":"completed","stop_reason":"max_turns"}',
        "[]",
        "not-json",
    ],
)
def test_run_outcome_rejects_invalid_json_contract(payload: str) -> None:
    with pytest.raises(ValueError):
        AgentRunOutcome.from_json(payload)
