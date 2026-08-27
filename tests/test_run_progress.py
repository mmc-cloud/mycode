import pytest

from mycode.run_progress import (
    COMPLETION_CORRECTION_PROMPT,
    DEFAULT_READONLY_TURN_LIMIT,
    DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT,
    DEFAULT_READY_ACTION_TURN_LIMIT,
    MAIN_CONVERGENCE_PROMPT,
    READINESS_ACTION_PROMPT,
    REPLAN_PROMPT,
    RESOURCE_STAGNATION_REPEAT,
    RESOURCE_STAGNATION_WINDOW,
    RESUME_PROMPT,
    RunProgress,
    TASK_PHASE_PROMPTS,
    ToolBehavior,
    ToolBehaviorObservation,
    ToolObservation,
    blocks_investigation_for_policy,
    classify_tool_behavior,
    classify_tool_effect,
    classify_resource_key,
    decide_completion_request,
    observe_tool_result,
    resolve_runtime_decision,
    turn_guidance,
)
from mycode.tools.command_risk import analyze_command_risk
from mycode.tools import Workspace


def _observations(
    *behaviors: ToolBehavior,
    succeeded: bool = True,
) -> tuple[ToolBehaviorObservation, ...]:
    return tuple(
        ToolBehaviorObservation(behavior=behavior, succeeded=succeeded)
        for behavior in behaviors
    )


def _reach_readonly_threshold(progress: RunProgress) -> None:
    for turn in range(1, DEFAULT_READONLY_TURN_LIMIT + 1):
        progress.observe_tool_turn(
            turn_number=turn,
            observations=_observations("investigate"),
        )


def test_readonly_threshold_enters_soft_convergence() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)

    first_notice = progress.take_readiness_notice(9)
    first_decision = resolve_runtime_decision(
        progress=progress,
        turn_index=8,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=first_notice,
        resume_guidance=None,
        completion_correction_guidance=None,
    )
    second_notice = progress.take_readiness_notice(10)
    second_decision = resolve_runtime_decision(
        progress=progress,
        turn_index=9,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=second_notice,
        resume_guidance=None,
        completion_correction_guidance=None,
    )

    assert progress.readiness_mode == "soft_convergence"
    assert len(first_decision.guidance) == 1
    assert "请整合现有证据并决定下一步" in first_decision.guidance[0]
    assert first_notice == (
        "已连续只读 8 轮，进入短暂收敛提醒，优先实施和验证"
    )
    assert second_decision.guidance == first_decision.guidance
    assert second_notice is None
    assert first_decision.tool_policy == "open"
    assert progress.readiness_turn == 9


def test_soft_convergence_uses_two_investigation_turns_before_action_window() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)
    for turn in range(DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=9 + turn,
            observations=_observations("investigate"),
        )

    assert progress.readiness_mode == "ready_action"
    notice = progress.take_readiness_notice(11)
    decision = resolve_runtime_decision(
        progress=progress,
        turn_index=10,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=notice,
        resume_guidance=None,
        completion_correction_guidance=None,
    )
    assert "开放式调查暂时停止" in decision.guidance[0]
    assert notice is not None and "有限动作窗口" in notice
    assert decision.tool_policy == "block_investigate"


def test_soft_convergence_successful_edit_resets_readiness_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)

    progress.observe_tool_turn(
        turn_number=9,
        observations=_observations("act"),
    )

    assert progress.first_edit_turn == 9
    assert progress.readiness_mode == "open"
    assert progress.readiness_active is False


def test_soft_convergence_successful_verify_resets_readiness_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)

    progress.observe_tool_turn(
        turn_number=9,
        observations=_observations("verify"),
    )

    assert progress.first_key_test_turn == 9
    assert progress.readiness_mode == "open"


def test_failed_verify_does_not_reset_readiness_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)

    progress.observe_tool_turn(
        turn_number=9,
        observations=_observations("verify", succeeded=False),
    )

    assert progress.readiness_mode == "soft_convergence"
    assert progress.soft_convergence_turn_count == 0
    assert progress.consecutive_readonly_turns == DEFAULT_READONLY_TURN_LIMIT
    assert progress.last_verification_succeeded is False


def test_action_window_successful_verify_resets_readiness_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)
    for turn in range(DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=9 + turn,
            observations=_observations("investigate"),
        )

    progress.observe_tool_turn(
        turn_number=11,
        observations=_observations("verify"),
    )

    assert progress.readiness_mode == "open"


def test_action_window_exhaustion_reopens_and_requests_replan() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)
    for turn in range(DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=9 + turn,
            observations=_observations("investigate"),
        )

    for attempt in range(DEFAULT_READY_ACTION_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=11 + attempt,
            observations=_observations("investigate", succeeded=False),
        )

    assert progress.readiness_mode == "open"
    assert progress.readiness_active is False
    assert progress.last_progress_reason == "readiness_action_window_exhausted"
    reason = progress.take_replan_reason()
    assert reason == "有限收敛窗口没有产生明确进展或完成结果"
    assert "修改或验证" not in reason


def test_replan_prompt_supports_new_evidence_coding_or_analysis_completion() -> None:
    assert "一个能够产生新信息的直接动作" in REPLAN_PROMPT
    assert "进行必要修改、验证" in REPLAN_PROMPT
    assert "形成结论并结束" in REPLAN_PROMPT
    assert "无信息增益的调查路径" in REPLAN_PROMPT
    assert "立即进入修改和关键验证" not in REPLAN_PROMPT


def _resource_visit(
    resource_key: str,
    *,
    succeeded: bool = True,
) -> ToolObservation:
    return ToolObservation(
        effect="investigate",
        succeeded=succeeded,
        tool_name="read_file",
        resource_key=resource_key,
    )


def test_resource_stagnation_triggers_at_five_of_seven_and_clears_window() -> None:
    progress = RunProgress()
    resources = [
        "file:src/query.py",
        "file:src/other.py",
        "file:src/query.py",
        "file:src/query.py",
        "file:src/third.py",
        "file:src/query.py",
        "file:src/query.py",
    ]

    for turn_number, resource_key in enumerate(resources, start=1):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(_resource_visit(resource_key),),
        )

    assert RESOURCE_STAGNATION_WINDOW == 7
    assert RESOURCE_STAGNATION_REPEAT == 5
    assert progress.pending_replan_reason is not None
    assert "src/query.py" in progress.pending_replan_reason
    assert progress.last_stagnant_resource == "file:src/query.py"
    assert list(progress.recent_investigation_resources) == []

    decision = resolve_runtime_decision(
        progress=progress,
        turn_index=7,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=None,
        resume_guidance=None,
        completion_correction_guidance=None,
    )
    assert len(decision.guidance) == 1
    assert "src/query.py" in decision.guidance[0]
    assert decision.tool_policy == "open"

    progress.take_replan_reason()
    progress.observe_tool_turn(
        turn_number=8,
        observations=(_resource_visit("file:src/query.py"),),
    )
    assert progress.pending_replan_reason is None
    assert list(progress.recent_investigation_resources) == ["file:src/query.py"]


def test_read_ranges_and_grep_for_same_file_share_stagnation_history(
    tmp_path,
) -> None:
    workspace = Workspace(tmp_path)
    source = tmp_path / "src" / "query.py"
    source.parent.mkdir()
    source.write_text("needle = True\n", encoding="utf-8")
    (source.parent / "other.py").write_text("other\n", encoding="utf-8")
    (source.parent / "third.py").write_text("third\n", encoding="utf-8")
    calls = [
        ("read_file", {"path": "src/query.py", "start_line": 1}),
        ("read_file", {"path": "src/other.py", "start_line": 1}),
        ("grep", {"query": "needle", "path_pattern": "src/query.py"}),
        ("read_file", {"path": "./src/query.py", "start_line": 200}),
        ("read_file", {"path": "src/third.py", "start_line": 1}),
        ("grep", {"query": "value", "path_pattern": "src/query.py"}),
        ("read_file", {"path": "src\\query.py", "start_line": 400}),
    ]
    progress = RunProgress()

    for turn_number, (tool_name, arguments) in enumerate(calls, start=1):
        resource_key = classify_resource_key(
            tool_name=tool_name,
            arguments=arguments,
            workspace=workspace,
        )
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(
                ToolObservation(
                    effect="investigate",
                    succeeded=True,
                    resource_key=resource_key,
                ),
            ),
        )

    assert progress.last_stagnant_resource == "file:src/query.py"
    assert progress.pending_replan_reason is not None


@pytest.mark.parametrize(
    "resources",
    [
        ["file:a.py"] * 4 + ["file:b.py"] * 3,
        [f"file:{index}.py" for index in range(7)],
    ],
)
def test_resource_stagnation_does_not_trigger_without_five_of_seven(
    resources: list[str],
) -> None:
    progress = RunProgress()

    for turn_number, resource_key in enumerate(resources, start=1):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(_resource_visit(resource_key),),
        )

    assert progress.pending_replan_reason is None
    assert list(progress.recent_investigation_resources) == resources


@pytest.mark.parametrize("validation_passed", [True, False])
def test_mutation_and_effective_validation_clear_resource_history(
    validation_passed: bool,
) -> None:
    validation_progress = RunProgress()
    validation_progress.recent_investigation_resources.extend(["file:a.py"] * 4)
    validation_progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolObservation(
                effect="validate",
                succeeded=validation_passed,
                validation_passed=validation_passed,
            ),
        ),
    )
    mutation_progress = RunProgress()
    mutation_progress.recent_investigation_resources.extend(["file:a.py"] * 4)
    mutation_progress.observe_tool_turn(
        turn_number=1,
        observations=(ToolObservation(effect="mutate", succeeded=True),),
    )

    assert list(validation_progress.recent_investigation_resources) == []
    assert list(mutation_progress.recent_investigation_resources) == []


def test_validation_without_result_does_not_clear_resource_history() -> None:
    progress = RunProgress()
    progress.recent_investigation_resources.extend(["file:a.py"] * 4)

    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolObservation(
                effect="validate",
                succeeded=False,
                validation_passed=None,
            ),
        ),
    )

    assert list(progress.recent_investigation_resources) == ["file:a.py"] * 4


def test_resource_history_excludes_rehydrate_failed_and_ready_action_reads() -> None:
    progress = RunProgress(readiness_mode="ready_action")

    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolObservation(
                effect="rehydrate",
                succeeded=True,
                resource_key="file:artifact.txt",
            ),
            _resource_visit("file:blocked.py", succeeded=False),
            _resource_visit("file:unexpected.py"),
        ),
    )

    assert list(progress.recent_investigation_resources) == []


@pytest.mark.parametrize("phase", ["ACT", "VERIFY", "VALIDATED"])
def test_resource_history_is_disabled_outside_investigate(phase: str) -> None:
    progress = RunProgress(task_phase=phase)

    for turn_number in range(1, 8):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(_resource_visit("file:src/query.py"),),
        )

    assert list(progress.recent_investigation_resources) == []
    assert progress.pending_replan_reason is None


def test_resource_stagnation_preserves_existing_replan_and_subagent_behavior() -> None:
    progress = RunProgress()
    progress.request_replan("先到达的精确重复原因")
    for turn_number in range(1, 8):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(_resource_visit("file:src/query.py"),),
        )

    subagent_progress = RunProgress(readonly_turn_limit=None)
    for turn_number in range(1, 8):
        subagent_progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(_resource_visit("file:src/query.py"),),
        )

    assert progress.pending_replan_reason == "先到达的精确重复原因"
    assert list(progress.recent_investigation_resources) == []
    assert subagent_progress.pending_replan_reason is None
    assert list(subagent_progress.recent_investigation_resources) == []


def test_key_test_records_turn_and_resets_readonly_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)

    progress.observe_tool_turn(
        turn_number=9,
        observations=_observations("verify"),
    )

    assert progress.first_key_test_turn == 9
    assert progress.consecutive_readonly_turns == 0
    assert progress.readiness_active is False
    assert progress.take_readiness_notice(11) is None


def test_final_answer_resets_readonly_counter() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )

    progress.observe_final_answer()

    assert progress.consecutive_readonly_turns == 0
    assert progress.readiness_active is False


def test_behavior_counts_record_mixed_turn_without_calling_it_investigation_only() -> None:
    progress = RunProgress(readonly_turn_limit=1)

    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate", "other"),
    )

    assert progress.last_tool_effects == frozenset({"investigate", "neutral"})
    assert progress.investigation_turn_count == 1
    assert progress.other_turn_count == 1
    assert progress.consecutive_readonly_turns == 0
    assert progress.readiness_mode == "open"


def test_classify_tool_behavior_uses_command_metadata() -> None:
    inspect_behavior = classify_tool_behavior(
        tool_name="run_command",
        capability="command",
        arguments={"command": ["git", "diff"]},
        ok=True,
        metadata={"command_risk_category": "inspect", "exit_code": 0},
    )
    failed_test_behavior = classify_tool_behavior(
        tool_name="run_command",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"command_risk_category": "test", "exit_code": 1},
    )

    assert inspect_behavior == "investigate"
    assert failed_test_behavior == "verify"


def test_read_artifact_is_rehydration_not_investigation() -> None:
    behavior = classify_tool_behavior(
        tool_name="read_artifact",
        capability="read",
        arguments={"artifact_path": "artifacts/result.txt"},
        ok=True,
        metadata={},
    )
    progress = RunProgress(readonly_turn_limit=1)

    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(
                behavior=behavior,
                succeeded=True,
                tool_name="read_artifact",
            ),
        ),
    )

    assert behavior == "rehydrate"
    assert progress.rehydration_turn_count == 1
    assert progress.investigation_turn_count == 0
    assert progress.consecutive_readonly_turns == 0
    assert progress.readiness_mode == "open"


def test_validation_without_a_result_keeps_validation_passed_unknown() -> None:
    permission_rejected = observe_tool_result(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"permission_status": "ask"},
    )
    start_failed = observe_tool_result(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["missing-validator"]},
        ok=False,
        metadata={"exit_code": None, "timed_out": False},
    )
    timed_out = observe_tool_result(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"exit_code": None, "timed_out": True},
    )

    assert permission_rejected.effect == "validate"
    assert start_failed.effect == "validate"
    assert timed_out.effect == "validate"
    assert permission_rejected.validation_passed is None
    assert start_failed.validation_passed is None
    assert timed_out.validation_passed is None


def test_classify_tool_behavior_distinguishes_temporary_and_formal_writes() -> None:
    temporary = classify_tool_behavior(
        tool_name="write_file",
        capability="write",
        arguments={"path": "test_repro_skip_pdb.py"},
        ok=True,
        metadata={},
    )
    formal_test = classify_tool_behavior(
        tool_name="write_file",
        capability="write",
        arguments={"path": "tests/test_repro_skip_pdb.py"},
        ok=True,
        metadata={},
    )
    formal_source = classify_tool_behavior(
        tool_name="edit_file",
        capability="write",
        arguments={"path": "src/debug_tools.py"},
        ok=True,
        metadata={},
    )

    assert temporary == "investigate"
    assert formal_test == "act"
    assert formal_source == "act"


def test_core_tools_have_explicit_runtime_effects() -> None:
    for tool_name in ("read_file", "grep", "glob", "inspect_changes"):
        assert classify_tool_effect(
            tool_name=tool_name,
            capability="read",
            arguments={},
        ) == "investigate"

    assert classify_tool_effect(
        tool_name="read_artifact",
        capability="read",
        arguments={},
    ) == "rehydrate"
    assert classify_tool_effect(
        tool_name="edit_file",
        capability="write",
        arguments={"path": "src/app.py"},
    ) == "mutate"
    assert classify_tool_effect(
        tool_name="memory_search",
        capability=None,
        arguments={},
    ) == "neutral"


def test_delegate_roles_have_policy_only_investigation_semantics() -> None:
    assert blocks_investigation_for_policy(
        tool_name="delegate_task",
        capability="control",
        arguments={"role": "explorer"},
    ) is True
    assert blocks_investigation_for_policy(
        tool_name="delegate_task",
        capability="control",
        arguments={"role": "reviewer"},
    ) is True
    assert blocks_investigation_for_policy(
        tool_name="delegate_task",
        capability="control",
        arguments={"role": "tester"},
    ) is False

    tester = observe_tool_result(
        tool_name="delegate_task",
        capability="control",
        arguments={"role": "tester"},
        ok=True,
        metadata={"child_status": "completed"},
    )
    progress = RunProgress(mutation_revision=1, task_phase="VERIFY")
    progress.observe_tool_turn(turn_number=1, observations=(tester,))

    assert tester.effect == "neutral"
    assert progress.validated_revision == 0
    assert progress.task_phase == "VERIFY"


def test_resource_key_unifies_read_ranges_grep_and_path_spellings(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir()
    source.write_text("needle = True\n", encoding="utf-8")

    keys = {
        classify_resource_key(
            tool_name="read_file",
            arguments={"path": path, "start_line": start_line},
            workspace=workspace,
        )
        for path, start_line in (
            ("src/foo.py", 1),
            ("./src/foo.py", 20),
            ("src\\foo.py", 40),
            (str(source), 60),
        )
    }
    keys.add(
        classify_resource_key(
            tool_name="grep",
            arguments={"query": "needle", "path_pattern": "src/foo.py"},
            workspace=workspace,
        )
    )

    assert keys == {"file:src/foo.py"}


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("glob", {"pattern": "**/*.py"}),
        ("grep", {"query": "needle", "path_pattern": "**/*.py"}),
        ("grep", {"query": "needle", "path_pattern": "src"}),
        ("run_command", {"command": ["rg", "needle", "."]}),
        ("run_command", {"command": ["bash", "-c", "cat src/foo.py"]}),
        ("delegate_task", {"text": "inspect"}),
    ],
)
def test_resource_key_is_none_without_one_explicit_file(
    tmp_path,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / "src").mkdir()

    assert classify_resource_key(
        tool_name=tool_name,
        arguments=arguments,
        workspace=workspace,
    ) is None


def test_run_command_effect_is_classified_from_argv_not_permission_risk() -> None:
    cases = (
        (["sed", "-n", "1,20p", "app.py"], "investigate"),
        (["sed", "-i", "s/a/b/", "app.py"], "mutate"),
        (["cp", "source.py", "target.py"], "mutate"),
        (["mv", "old.py", "new.py"], "mutate"),
        (["rm", "obsolete.py"], "mutate"),
        (["rmdir", "obsolete"], "mutate"),
        (["mkdir", "generated"], "mutate"),
        (["touch", "generated.txt"], "mutate"),
        (["Set-Content", "app.py", "updated"], "mutate"),
        (["Remove-Item", "obsolete.py"], "mutate"),
        (["Copy-Item", "source.py", "target.py"], "mutate"),
        (["Move-Item", "old.py", "new.py"], "mutate"),
        (["Add-Content", "app.py", "updated"], "mutate"),
        (["Out-File", "result.txt"], "mutate"),
        (["del", "obsolete.py"], "mutate"),
        (["copy", "source.py", "target.py"], "mutate"),
        (["move", "old.py", "new.py"], "mutate"),
        (["bash", "-c", "rm obsolete.py"], "neutral"),
        (["python", "-c", "open('app.py', 'w').close()"], "neutral"),
        (["pytest", "-q"], "validate"),
        (["python", "script.py"], "neutral"),
        (["pwsh", "-Command", "Set-Content", "app.py", "updated"], "mutate"),
        (["powershell", "-Command", "Copy-Item", "a.py", "b.py"], "mutate"),
        (["powershell.exe", "-Command", "Remove-Item", "old.py"], "mutate"),
        (["cmd", "/c", "copy", "a.py", "b.py"], "mutate"),
        (["cmd.exe", "/c", "del", "old.py"], "mutate"),
        (["pwsh", "-Command", "Get-Content", "app.py"], "neutral"),
        (["pwsh", "-Command", "if ($x) { Set-Content app.py x }"], "neutral"),
        (["pwsh", "-Command", "$x='Set-Content';", "&", "$x"], "neutral"),
        (["cmd", "/c", "copy", "a", "b", "&&", "del", "a"], "neutral"),
        (["cmd", "/c", "copy a b"], "neutral"),
    )

    for command, expected_effect in cases:
        permission_category = analyze_command_risk(command).category
        observation = observe_tool_result(
            tool_name="run_command",
            capability="command",
            arguments={"command": command},
            ok=True,
            metadata={
                "command_risk_category": permission_category,
                "exit_code": 0,
            },
        )
        assert observation.effect == expected_effect

    assert analyze_command_risk(["sed", "-i", "s/a/b/", "app.py"]).category == (
        "inspect"
    )
    assert analyze_command_risk(["Set-Content", "app.py", "updated"]).category == (
        "unknown"
    )


@pytest.mark.parametrize(
    "command",
    [
        ["pwsh", "-Command", "Set-Content", "app.py", "updated"],
        ["cmd.exe", "/c", "copy", "source.py", "target.py"],
    ],
)
def test_wrapped_command_mutation_reaches_completion_guard(
    command: list[str],
) -> None:
    observation = observe_tool_result(
        tool_name="run_command",
        capability="command",
        arguments={"command": command},
        ok=True,
        metadata={
            "command_risk_category": analyze_command_risk(command).category,
            "exit_code": 0,
        },
    )
    progress = RunProgress()

    progress.observe_tool_turn(turn_number=1, observations=(observation,))

    assert observation.effect == "mutate"
    assert progress.mutation_revision == 1
    assert progress.task_phase == "VERIFY"
    assert decide_completion_request(progress) == "correct"


def test_run_command_mutation_reaches_revision_and_completion_guard() -> None:
    observation = observe_tool_result(
        tool_name="run_command",
        capability="command",
        arguments={"command": ["cp", "source.py", "target.py"]},
        ok=True,
        metadata={
            "command_risk_category": "unknown",
            "exit_code": 0,
        },
    )
    progress = RunProgress()

    progress.observe_tool_turn(
        turn_number=1,
        observations=(observation,),
    )

    assert observation.effect == "mutate"
    assert progress.mutation_revision == 1
    assert progress.validated_revision == 0
    assert progress.task_phase == "VERIFY"
    assert decide_completion_request(progress) == "correct"
    progress.record_completion_correction()
    assert progress.completion_correction_revision == 1


def test_validation_observation_distinguishes_pass_fail_and_no_result() -> None:
    passed = observe_tool_result(
        tool_name="run_command",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=True,
        metadata={"exit_code": 0},
    )
    failed = observe_tool_result(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"exit_code": 1},
    )
    unavailable = observe_tool_result(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"exit_code": None, "timed_out": True},
    )

    assert (passed.effect, passed.validation_passed) == ("validate", True)
    assert (failed.effect, failed.validation_passed) == ("validate", False)
    assert (unavailable.effect, unavailable.validation_passed) == ("validate", None)


def test_failed_mutation_does_not_advance_revision() -> None:
    progress = RunProgress()

    progress.observe_tool_turn(
        turn_number=1,
        observations=(ToolBehaviorObservation(behavior="act", succeeded=False),),
    )

    assert progress.mutation_revision == 0
    assert progress.validated_revision == 0
    assert progress.task_phase == "INVESTIGATE"


def test_completion_does_not_depend_on_validation_tool_name() -> None:
    states = []
    for tool_name in ("run_validation", "run_command"):
        progress = RunProgress()
        progress.observe_tool_turn(
            turn_number=1,
            observations=(
                ToolBehaviorObservation(behavior="act", succeeded=True),
                ToolObservation(
                    effect="validate",
                    succeeded=True,
                    validation_passed=True,
                    tool_name=tool_name,
                ),
            ),
        )
        assert decide_completion_request(progress) == "accept"
        progress.observe_final_answer()
        states.append(
            (
                progress.task_phase,
                progress.mutation_revision,
                progress.validated_revision,
            )
        )

    assert states == [("VALIDATED", 1, 1), ("VALIDATED", 1, 1)]


def test_task_phase_closes_act_verify_loop_in_execution_order() -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )
    assert progress.task_phase == "INVESTIGATE"

    progress.observe_tool_turn(
        turn_number=3,
        observations=_observations("act"),
    )
    assert progress.task_phase == "VERIFY"

    progress.observe_tool_turn(
        turn_number=4,
        observations=_observations("verify", succeeded=False),
    )
    assert progress.task_phase == "ACT"
    assert progress.last_verification_succeeded is False

    progress.observe_tool_turn(
        turn_number=5,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )

    assert progress.task_phase == "VALIDATED"
    assert progress.last_verification_succeeded is True
    assert progress.task_phase_history == [
        "INVESTIGATE",
        "VERIFY",
        "ACT",
        "VERIFY",
        "VALIDATED",
    ]


def test_validation_cannot_finish_before_a_formal_action() -> None:
    progress = RunProgress(readonly_turn_limit=1)

    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("verify"),
    )

    assert progress.task_phase == "INVESTIGATE"
    assert progress.last_verification_succeeded is True
    assert progress.task_phase_history == ["INVESTIGATE"]

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("investigate"),
    )

    assert progress.task_phase == "INVESTIGATE"
    assert progress.last_verification_succeeded is True


def test_task_phase_respects_order_when_validation_precedes_edit() -> None:
    progress = RunProgress()

    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="verify", succeeded=True),
            ToolBehaviorObservation(behavior="act", succeeded=True),
        ),
    )

    assert progress.task_phase == "VERIFY"
    assert progress.task_phase_history == ["INVESTIGATE", "VERIFY"]


def test_final_answer_does_not_change_progress_state() -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act"),
    )
    before = (
        progress.task_phase,
        progress.mutation_revision,
        progress.validated_revision,
    )

    progress.observe_final_answer()

    assert (
        progress.task_phase,
        progress.mutation_revision,
        progress.validated_revision,
    ) == before


def test_turn_limit_resets_readiness_without_marking_task_done() -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )

    progress.observe_turn_limit()

    assert progress.readiness_mode == "open"
    assert progress.task_phase == "INVESTIGATE"


def test_failed_followup_validation_invalidates_current_revision() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )
    assert progress.task_phase == "VALIDATED"

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("verify", succeeded=False),
    )

    assert progress.task_phase == "ACT"
    assert progress.mutation_revision == 1
    assert progress.validated_revision == 0
    assert progress.last_verification_succeeded is False
    assert decide_completion_request(progress) == "correct"


def test_failed_followup_validation_reopens_correction_then_repass_recovers() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act", succeeded=True),
    )
    progress.record_completion_correction()
    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("verify", succeeded=True),
    )
    assert progress.task_phase == "VALIDATED"
    assert progress.completion_correction_revision == 1

    progress.observe_tool_turn(
        turn_number=3,
        observations=_observations("verify", succeeded=False),
    )

    assert progress.task_phase == "ACT"
    assert progress.validated_revision == 0
    assert progress.completion_correction_revision is None
    assert decide_completion_request(progress) == "correct"

    progress.observe_tool_turn(
        turn_number=4,
        observations=_observations("verify", succeeded=True),
    )

    assert progress.task_phase == "VALIDATED"
    assert progress.validated_revision == 1
    assert progress.last_verification_succeeded is True
    assert decide_completion_request(progress) == "accept"


def test_new_mutation_after_invalidated_validation_advances_revision() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )
    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("verify", succeeded=False),
    )

    progress.observe_tool_turn(
        turn_number=3,
        observations=_observations("act", succeeded=True),
    )

    assert progress.task_phase == "VERIFY"
    assert progress.mutation_revision == 2
    assert progress.validated_revision == 0


@pytest.mark.parametrize("tool_name", ["run_validation", "run_command"])
def test_validation_tools_share_pass_then_fail_invalidation(tool_name: str) -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act", succeeded=True),
    )
    passed = observe_tool_result(
        tool_name=tool_name,
        capability="command",
        arguments={"command": ["pytest", "-q"]},
        ok=True,
        metadata={"exit_code": 0},
    )
    failed = observe_tool_result(
        tool_name=tool_name,
        capability="command",
        arguments={"command": ["pytest", "-q"]},
        ok=False,
        metadata={"exit_code": 1},
    )

    progress.observe_tool_turn(turn_number=2, observations=(passed,))
    progress.observe_tool_turn(turn_number=3, observations=(failed,))

    assert progress.task_phase == "ACT"
    assert progress.validated_revision == 0
    assert progress.last_verification_succeeded is False
    assert decide_completion_request(progress) == "correct"


def test_post_validation_tool_turn_is_observed_without_restricting_it() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("investigate"),
    )

    assert progress.task_phase == "VALIDATED"
    assert progress.post_validation_tool_turn_count == 1
    assert progress.last_progress_reason == "post_validation_tool_observed"


def test_completion_correction_is_once_per_mutation_revision() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act", succeeded=True),
    )

    assert decide_completion_request(progress) == "correct"
    state_before_correction = (
        progress.task_phase,
        progress.mutation_revision,
        progress.validated_revision,
    )
    progress.record_completion_correction()
    assert progress.completion_correction_revision == 1
    assert progress.last_progress_reason == "completion_correction_required"
    assert (
        progress.task_phase,
        progress.mutation_revision,
        progress.validated_revision,
    ) == state_before_correction
    assert decide_completion_request(progress) == "accept"

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("act", succeeded=True),
    )

    assert progress.mutation_revision == 2
    assert progress.completion_correction_revision == 1
    assert decide_completion_request(progress) == "correct"


def test_successful_validation_of_current_revision_avoids_correction() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(
                behavior="verify",
                succeeded=True,
                tool_name="run_validation",
            ),
        ),
    )

    assert progress.mutation_revision == 1
    assert progress.validated_revision == 1
    assert decide_completion_request(progress) == "accept"


def test_new_mutation_invalidates_previous_successful_validation() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(
                behavior="verify",
                succeeded=True,
                tool_name="run_validation",
            ),
            ToolBehaviorObservation(behavior="act", succeeded=True),
        ),
    )

    assert progress.mutation_revision == 2
    assert progress.validated_revision == 1
    assert decide_completion_request(progress) == "correct"


def test_no_formal_mutation_does_not_trigger_completion_correction() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("verify", succeeded=False),
    )

    assert decide_completion_request(progress) == "accept"


def test_run_command_validation_satisfies_completion_guard() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(
                behavior="verify",
                succeeded=True,
                tool_name="run_command",
            ),
        ),
    )

    assert progress.validated_revision == 1
    assert decide_completion_request(progress) == "accept"


def test_validated_phase_survives_non_mutating_tool_effects() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )

    for turn_number, effect in enumerate(
        ("investigate", "rehydrate", "neutral"),
        start=2,
    ):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=(ToolObservation(effect=effect, succeeded=True),),
        )
        assert progress.task_phase == "VALIDATED"
        assert progress.mutation_revision == 1
        assert progress.validated_revision == 1


def test_validated_phase_returns_to_verify_after_new_mutation() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("act", succeeded=True),
    )

    assert progress.task_phase == "VERIFY"
    assert progress.mutation_revision == 2
    assert progress.validated_revision == 1


def test_task_phase_guidance_tracks_phase_without_forcing_long_investigation() -> None:
    progress = RunProgress(readonly_turn_limit=1)

    def decision():
        return resolve_runtime_decision(
            progress=progress,
            turn_index=0,
            max_turns=50,
            convergence_remaining_turns=None,
            convergence_prompt=None,
            readiness_notice=None,
            resume_guidance=None,
            completion_correction_guidance=None,
        )

    assert decision().guidance == ()

    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )

    progress.observe_tool_turn(
        turn_number=3,
        observations=_observations("act"),
    )
    assert TASK_PHASE_PROMPTS["VERIFY"] in decision().guidance[0]

    progress.observe_tool_turn(
        turn_number=4,
        observations=_observations("verify", succeeded=False),
    )
    assert TASK_PHASE_PROMPTS["ACT"] in decision().guidance[0]

    progress.observe_tool_turn(
        turn_number=5,
        observations=_observations("act"),
    )
    assert TASK_PHASE_PROMPTS["VERIFY"] in decision().guidance[0]

    progress.observe_tool_turn(
        turn_number=6,
        observations=_observations("verify"),
    )
    validated = decision().guidance[0]
    assert TASK_PHASE_PROMPTS["VALIDATED"] in validated
    assert "只有已经识别出具体剩余问题时才继续调用工具" in validated


def test_task_phase_guidance_is_disabled_with_main_readiness_for_subagents() -> None:
    progress = RunProgress(readonly_turn_limit=None)

    decision = resolve_runtime_decision(
        progress=progress,
        turn_index=0,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=None,
        resume_guidance=None,
        completion_correction_guidance=None,
    )

    assert decision.guidance == ()
    assert decision.tool_policy == "open"


@pytest.mark.parametrize("phase", ["VERIFY", "ACT", "VALIDATED"])
def test_non_investigate_phases_do_not_advance_readiness(phase: str) -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act"),
    )
    if phase == "ACT":
        progress.observe_tool_turn(
            turn_number=2,
            observations=_observations("verify", succeeded=False),
        )
    elif phase == "VALIDATED":
        progress.observe_tool_turn(
            turn_number=2,
            observations=_observations("verify"),
        )

    for turn_number in range(3, 8):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=_observations("investigate"),
        )

    assert progress.task_phase == phase
    assert progress.readiness_mode == "open"
    assert progress.consecutive_readonly_turns == 0


def test_ready_action_rehydrate_pauses_but_neutral_consumes_window() -> None:
    progress = RunProgress(readiness_mode="ready_action")

    for turn_number in range(1, 4):
        progress.observe_tool_turn(
            turn_number=turn_number,
            observations=_observations("rehydrate"),
        )

    assert progress.readiness_mode == "ready_action"
    assert progress.ready_action_turn_count == 0

    progress.observe_tool_turn(
        turn_number=4,
        observations=_observations("other"),
    )

    assert progress.readiness_mode == "ready_action"
    assert progress.ready_action_turn_count == 1


def test_completion_correction_keeps_readiness_state_but_opens_policy() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act"),
    )
    progress.readiness_mode = "ready_action"

    progress.record_completion_correction()
    decision = resolve_runtime_decision(
        progress=progress,
        turn_index=1,
        max_turns=50,
        convergence_remaining_turns=None,
        convergence_prompt=None,
        readiness_notice=None,
        resume_guidance=None,
        completion_correction_guidance=COMPLETION_CORRECTION_PROMPT,
    )

    assert progress.readiness_mode == "ready_action"
    assert decision.tool_policy == "open"
    assert "target_phase: VERIFY" in decision.guidance[0]


def test_completion_correction_prefers_formal_validation_without_requiring_it() -> None:
    assert "`run_validation`" in COMPLETION_CORRECTION_PROMPT
    assert "`run_command`" in COMPLETION_CORRECTION_PROMPT
    assert "请优先使用" in COMPLETION_CORRECTION_PROMPT
    assert "无需执行验证（例如分析或 Review）" in COMPLETION_CORRECTION_PROMPT
    assert "客观上无法执行验证" in COMPLETION_CORRECTION_PROMPT
    assert "可基于已有证据说明情况并完成任务" in COMPLETION_CORRECTION_PROMPT
    assert "必须验证" not in COMPLETION_CORRECTION_PROMPT


def test_turn_guidance_emits_only_completion_correction_when_signals_compete() -> None:
    guidance, notice = turn_guidance(
        turn_index=48,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="ACT",
        task_phase_guidance=TASK_PHASE_PROMPTS["ACT"],
        readiness_guidance=READINESS_ACTION_PROMPT,
        readiness_notice="动作窗口已开启",
        resume_guidance=RESUME_PROMPT,
        replan_reason="缺少新增证据",
        completion_correction_guidance=COMPLETION_CORRECTION_PROMPT,
    )

    assert len(guidance) == 1
    directive = guidance[0]
    assert "target_phase: VERIFY" in directive
    assert "reason: completion_correction_required_with_replan" in directive
    assert COMPLETION_CORRECTION_PROMPT in directive
    assert "上一动作没有推进（缺少新增证据）" in directive
    assert "调整方案后继续完成对当前修改的验证" in directive
    assert READINESS_ACTION_PROMPT not in directive
    assert MAIN_CONVERGENCE_PROMPT not in directive
    assert RESUME_PROMPT not in directive
    assert "调整方案" in directive
    assert "纠偏机会" in notice


def test_turn_guidance_keeps_act_primary_over_replan_and_convergence() -> None:
    guidance, _ = turn_guidance(
        turn_index=47,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="ACT",
        task_phase_guidance=TASK_PHASE_PROMPTS["ACT"],
        readiness_guidance=None,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason="最近调用没有新增证据",
        completion_correction_guidance=None,
    )

    assert len(guidance) == 1
    directive = guidance[0]
    assert "target_phase: ACT" in directive
    assert "reason: task_phase_act_with_replan" in directive
    assert TASK_PHASE_PROMPTS["ACT"] in directive
    assert "不要重复相同的失败动作" in directive
    assert "继续完成必要的修改" in directive
    assert "剩余轮次有限" in directive
    assert "调整方案" in directive
    assert MAIN_CONVERGENCE_PROMPT not in directive


def test_turn_guidance_promotes_ready_action_to_single_converge_directive() -> None:
    guidance, _ = turn_guidance(
        turn_index=47,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="INVESTIGATE",
        task_phase_guidance=None,
        readiness_guidance=READINESS_ACTION_PROMPT,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason="最近调用没有新增证据",
        completion_correction_guidance=None,
    )

    assert len(guidance) == 1
    directive = guidance[0]
    assert "target_phase: CONVERGE" in directive
    assert "reason: investigation_budget_exhausted_with_replan" in directive
    assert READINESS_ACTION_PROMPT in directive
    assert "剩余轮次有限" in directive
    assert "调整方案" in directive
    assert MAIN_CONVERGENCE_PROMPT not in directive


def test_turn_guidance_uses_resume_or_convergence_as_single_fallback() -> None:
    resumed, _ = turn_guidance(
        turn_index=0,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="INVESTIGATE",
        task_phase_guidance=None,
        readiness_guidance=None,
        readiness_notice=None,
        resume_guidance=RESUME_PROMPT,
        replan_reason=None,
        completion_correction_guidance=None,
    )
    converging, _ = turn_guidance(
        turn_index=46,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="INVESTIGATE",
        task_phase_guidance=None,
        readiness_guidance=None,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason=None,
        completion_correction_guidance=None,
    )

    assert len(resumed) == 1
    assert "reason: session_resumed" in resumed[0]
    assert RESUME_PROMPT in resumed[0]
    assert len(converging) == 1
    assert "reason: turn_budget_converging" in converging[0]
    assert MAIN_CONVERGENCE_PROMPT in converging[0]


def test_turn_guidance_keeps_verify_target_and_incorporates_pending_replan() -> None:
    guidance, notice = turn_guidance(
        turn_index=10,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="VERIFY",
        task_phase_guidance=TASK_PHASE_PROMPTS["VERIFY"],
        readiness_guidance=None,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason="验证命令重复且没有新进展",
        completion_correction_guidance=None,
    )

    assert len(guidance) == 1
    assert "target_phase: VERIFY" in guidance[0]
    assert "reason: task_phase_verify_with_replan" in guidance[0]
    assert "验证命令重复且没有新进展" in guidance[0]
    assert "继续完成对当前修改的验证" in guidance[0]
    assert "已要求 Agent 调整方案" in notice


def test_pending_replan_is_consumed_only_after_it_is_visible_in_directive() -> None:
    progress = RunProgress()
    progress.request_replan("重复动作没有推进")

    assert progress.pending_replan_reason == "重复动作没有推进"
    replan_reason = progress.pending_replan_reason
    guidance, _ = turn_guidance(
        turn_index=3,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="ACT",
        task_phase_guidance=TASK_PHASE_PROMPTS["ACT"],
        readiness_guidance=None,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason=replan_reason,
        completion_correction_guidance=None,
    )

    assert progress.pending_replan_reason == "重复动作没有推进"
    assert len(guidance) == 1
    assert "重复动作没有推进" in guidance[0]
    assert "reason: task_phase_act_with_replan" in guidance[0]
    assert progress.take_replan_reason() == replan_reason
    assert progress.pending_replan_reason is None


def test_completion_correction_incorporates_replan_before_consuming_it() -> None:
    progress = RunProgress()
    progress.request_replan("结束前的验证动作没有推进")
    replan_reason = progress.pending_replan_reason

    guidance, _ = turn_guidance(
        turn_index=50,
        max_turns=50,
        convergence_remaining_turns=5,
        convergence_prompt=MAIN_CONVERGENCE_PROMPT,
        task_phase="ACT",
        task_phase_guidance=TASK_PHASE_PROMPTS["ACT"],
        readiness_guidance=READINESS_ACTION_PROMPT,
        readiness_notice=None,
        resume_guidance=None,
        replan_reason=replan_reason,
        completion_correction_guidance=COMPLETION_CORRECTION_PROMPT,
    )

    assert progress.pending_replan_reason == replan_reason
    assert len(guidance) == 1
    assert "target_phase: VERIFY" in guidance[0]
    assert "reason: completion_correction_required_with_replan" in guidance[0]
    assert "结束前的验证动作没有推进" in guidance[0]
    assert progress.take_replan_reason() == replan_reason
    assert progress.pending_replan_reason is None
