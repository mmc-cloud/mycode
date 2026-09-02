import pytest

from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.run_progress import (
    RESUME_PROMPT,
    RuntimeObservation,
    RuntimePolicy,
    RuntimeState,
    classify_mutation,
    classify_resource,
    classify_validation,
    decide_runtime_policy,
    normalize_run_checkpoint,
    observe_tool_result,
    resume_guidance,
)
from mycode.tools import Workspace


def observation(
    *,
    tool: str = "read_file",
    result: str = "result-a",
    resource: str | None = "file:a.py",
    mutation: str = "no",
    validation: str = "none",
) -> RuntimeObservation:
    return RuntimeObservation(
        tool_result="success",
        tool_signature=tool,
        resource=resource,
        mutation=mutation,
        validation=validation,
        result_signature=result,
    )


def test_runtime_state_retains_mutation_and_validation_as_observation_facts() -> None:
    state = RuntimeState()
    observed = observation(
        tool="run_command",
        mutation="unknown",
        validation="pass",
    )

    state.observe_tool_turn((observed,))

    assert state.last_observation is observed
    assert state.last_tool_observation is observed
    assert state.last_tool_observation.mutation == "unknown"
    assert state.last_tool_observation.validation == "pass"


def test_first_repeated_turn_only_increments_stagnation() -> None:
    state = RuntimeState()
    repeated = observation()
    state.observe_tool_turn((repeated,))
    state.observe_tool_turn((repeated,))

    decision = decide_runtime_policy(state)

    assert state.stagnation_turns == 1
    assert decision.policy == RuntimePolicy.NO_INTERVENTION
    assert state.convergence_guided is False


def test_sustained_repetition_triggers_one_convergence_guidance_per_event() -> None:
    state = RuntimeState()
    repeated = observation()
    state.observe_tool_turn((repeated,))
    state.observe_tool_turn((repeated,))
    state.observe_tool_turn((repeated,))

    first = decide_runtime_policy(state)
    second = decide_runtime_policy(state)

    assert first.policy == RuntimePolicy.CONVERGENCE_GUIDANCE
    assert state.stagnation_turns == 2
    assert "same_tool_repeat=3" in first.guidance[0]
    assert second.policy == RuntimePolicy.NO_INTERVENTION
    assert state.convergence_guided is True


def test_continued_stagnation_does_not_repeat_guidance() -> None:
    state = RuntimeState()
    repeated = observation()
    state.observe_tool_turn((repeated,))
    state.observe_tool_turn((repeated,))
    state.observe_tool_turn((repeated,))
    decide_runtime_policy(state)
    state.observe_tool_turn((repeated,))

    assert state.stagnation_turns == 3
    assert (
        decide_runtime_policy(state).policy
        == RuntimePolicy.NO_INTERVENTION
    )


def test_nonrepeated_turns_decay_before_new_convergence_episode() -> None:
    state = RuntimeState()
    first = observation()
    state.observe_tool_turn((first,))
    state.observe_tool_turn((first,))
    state.observe_tool_turn((first,))
    decide_runtime_policy(state)

    different_a = observation(tool="grep", result="result-b", resource="file:b.py")
    state.observe_tool_turn((different_a,))
    assert state.stagnation_turns == 1
    assert state.convergence_guided is True

    different_b = observation(tool="glob", result="result-c", resource="file:c.py")
    state.observe_tool_turn((different_b,))
    assert state.stagnation_turns == 0
    assert state.convergence_guided is False

    state.observe_tool_turn((different_b,))
    assert state.stagnation_turns == 1
    assert (
        decide_runtime_policy(state).policy
        == RuntimePolicy.NO_INTERVENTION
    )
    state.observe_tool_turn((different_b,))
    assert (
        decide_runtime_policy(state).policy
        == RuntimePolicy.CONVERGENCE_GUIDANCE
    )


def test_same_result_or_resource_can_form_stagnation_without_same_tool() -> None:
    state = RuntimeState()
    state.observe_tool_turn((observation(tool="read_file"),))
    state.observe_tool_turn((observation(tool="grep"),))
    state.observe_tool_turn((observation(tool="glob"),))

    assert state.same_tool_repeat == 1
    assert state.same_result_repeat == 3
    assert state.resource_repeat == 3
    assert state.stagnation_turns == 2
    assert (
        decide_runtime_policy(state).policy
        == RuntimePolicy.CONVERGENCE_GUIDANCE
    )


def test_observe_tool_result_records_all_factual_fields(tmp_path) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 1", encoding="utf-8")
    workspace = Workspace(tmp_path)

    observed = observe_tool_result(
        tool_name="read_file",
        capability="read",
        arguments={"path": "a.py"},
        ok=True,
        content="x = 1",
        error=None,
        metadata={"path": "a.py", "start_line": 1, "end_line": 1},
        workspace=workspace,
    )

    assert observed.tool_result == "success"
    assert observed.tool_signature is not None
    assert observed.resource == "file:a.py:1-1"
    assert observed.mutation == "no"
    assert observed.validation == "none"
    assert observed.result_signature is not None


def test_run_command_mutation_is_unknown_even_when_it_succeeds() -> None:
    assert classify_mutation(
        tool_name="run_command",
        capability="command",
        arguments={"command": ["git", "status"]},
        ok=True,
    ) == "unknown"


@pytest.mark.parametrize(
    ("tool_name", "command", "metadata", "expected"),
    [
        ("run_command", ["pytest", "-q"], {"exit_code": 0}, "pass"),
        ("run_command", ["pytest", "-q"], {"exit_code": 1}, "fail"),
        ("run_validation", ["pytest", "-q"], {"exit_code": 0}, "pass"),
        ("run_command", ["pytest", "--help"], {"exit_code": 0}, "none"),
        ("run_validation", ["pytest", "--help"], {"exit_code": 0}, "none"),
        ("run_command", ["pytest", "--version"], {"exit_code": 0}, "none"),
        ("run_command", ["pytest", "--collect-only"], {"exit_code": 0}, "none"),
        ("run_validation", ["project-check"], {"exit_code": 0}, "unknown"),
        ("run_command", ["python", "-c", "print(1)"], {"exit_code": 0}, "unknown"),
        ("run_command", ["python", "test_fix.py"], {"exit_code": 0}, "pass"),
        ("run_command", ["python", "test_fix.py"], {"exit_code": 2}, "fail"),
        ("run_command", ["python", "arbitrary_script.py"], {"exit_code": 0}, "unknown"),
        ("run_command", ["python", "-m", "build"], {"exit_code": 0}, "pass"),
        ("run_command", ["pip", "install", "-e", "."], {"exit_code": 0}, "none"),
        ("run_command", ["ls"], {"exit_code": 0}, "none"),
        ("run_command", ["cat", "README.md"], {"exit_code": 0}, "none"),
        ("run_command", ["git", "status"], {"exit_code": 0}, "none"),
        ("run_command", ["project-check"], {"exit_code": 0}, "unknown"),
        ("run_command", ["pytest", "-q"], {"timed_out": True}, "unknown"),
    ],
)
def test_validation_uses_command_identity_and_confirmed_result(
    tool_name, command, metadata, expected
) -> None:
    assert classify_validation(
        tool_name=tool_name,
        arguments={"command": command},
        metadata=metadata,
    ) == expected


def test_read_file_resource_identity_uses_actual_returned_interval(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    first = classify_resource(
        "read_file",
        {"path": "a.py", "start_line": 1, "max_lines": 200},
        workspace,
        metadata={"path": "a.py", "start_line": 1, "end_line": 120},
    )
    second = classify_resource(
        "read_file",
        {"path": "a.py", "start_line": 121, "max_lines": 200},
        workspace,
        metadata={"path": "a.py", "start_line": 121, "end_line": 240},
    )

    assert first == "file:a.py:1-120"
    assert second == "file:a.py:121-240"
    assert first != second


def test_same_actual_read_file_interval_increases_resource_repeat(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    metadata = {"path": "a.py", "start_line": 1, "end_line": 120}
    observed = observe_tool_result(
        tool_name="read_file",
        capability="read",
        arguments={"path": "a.py", "start_line": 1, "max_lines": 200},
        ok=True,
        content="same",
        error=None,
        metadata=metadata,
        workspace=workspace,
    )
    state = RuntimeState()
    state.observe_tool_turn((observed,))
    state.observe_tool_turn((observed,))

    assert state.resource_repeat == 2


def test_different_read_file_intervals_do_not_increase_resource_repeat(tmp_path) -> None:
    workspace = Workspace(tmp_path)

    def observed(start_line: int, end_line: int) -> RuntimeObservation:
        return observe_tool_result(
            tool_name="read_file",
            capability="read",
            arguments={"path": "a.py", "start_line": start_line, "max_lines": 200},
            ok=True,
            content=f"{start_line}-{end_line}",
            error=None,
            metadata={
                "path": "a.py",
                "start_line": start_line,
                "end_line": end_line,
            },
            workspace=workspace,
        )

    first = observed(1, 120)
    second = observed(121, 240)
    state = RuntimeState()
    state.observe_tool_turn((first,))
    state.observe_tool_turn((second,))

    assert state.resource_repeat == 1
    state.observe_tool_turn((second,))
    assert state.resource_repeat == 2


def test_temporary_write_is_not_a_tracked_mutation() -> None:
    assert classify_mutation(
        tool_name="write_file",
        capability="write",
        arguments={"path": ".tmp/repro.py"},
        ok=True,
    ) == "no"


def test_debug_repro_and_temp_filenames_are_tracked_mutations() -> None:
    common = {
        "tool_name": "write_file",
        "capability": "write",
        "ok": True,
    }

    for path in (
        "debug.py",
        "repro.py",
        "temp.py",
        "scratch.py",
        "repro_bug.py",
        "test_repro_bug.py",
        "tests/test_reproduction.py",
    ):
        assert classify_mutation(**common, arguments={"path": path}) == "yes"


def test_deterministic_cache_build_and_temp_paths_are_not_tracked() -> None:
    common = {
        "tool_name": "write_file",
        "capability": "write",
        "ok": True,
    }

    for path in (
        ".pytest_cache/state.json",
        "__pycache__/module.pyc",
        "build/generated.py",
        "dist/package.txt",
        ".tmp/repro.py",
        "report.tmp",
    ):
        assert classify_mutation(**common, arguments={"path": path}) == "no"


def test_resume_guidance_and_checkpoint_normalization() -> None:
    conversation = Conversation.from_messages(
        [Message(role="assistant", content="## 续跑检查点\n剩余动作")]
    )

    assert resume_guidance(conversation, "继续") == RESUME_PROMPT
    assert normalize_run_checkpoint("状态") == "## 续跑检查点\n\n状态"
