from mycode.run_progress import (
    DEFAULT_READONLY_TURN_LIMIT,
    DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT,
    DEFAULT_READY_ACTION_TURN_LIMIT,
    RunProgress,
    ToolBehavior,
    ToolBehaviorObservation,
    classify_tool_behavior,
)


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

    first_guidance, first_notice = progress.readiness_guidance_for_turn(9)
    second_guidance, second_notice = progress.readiness_guidance_for_turn(10)

    assert progress.readiness_mode == "soft_convergence"
    assert first_guidance is not None
    assert "优先根据现有证据实施最小修改并验证" in first_guidance
    assert first_notice == (
        "已连续只读 8 轮，进入短暂收敛提醒，优先实施和验证"
    )
    assert second_guidance == first_guidance
    assert second_notice is None
    assert progress.readiness_turn == 9


def test_soft_convergence_uses_two_normal_tool_turns_before_action_window() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)
    for turn in range(DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=9 + turn,
            observations=_observations("other"),
        )

    assert progress.readiness_mode == "ready_action"
    guidance, notice = progress.readiness_guidance_for_turn(11)
    assert guidance is not None and "调查预算已经用完" in guidance
    assert notice is not None and "有限动作窗口" in notice


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
    assert progress.soft_convergence_turn_count == 1
    assert progress.consecutive_readonly_turns == DEFAULT_READONLY_TURN_LIMIT
    assert progress.last_verification_succeeded is False


def test_action_window_successful_verify_resets_readiness_cycle() -> None:
    progress = RunProgress()
    _reach_readonly_threshold(progress)
    for turn in range(DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT):
        progress.observe_tool_turn(
            turn_number=9 + turn,
            observations=_observations("other"),
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
            observations=_observations("other"),
        )

    for _ in range(DEFAULT_READY_ACTION_TURN_LIMIT):
        progress.observe_restricted_tool_turn(attempted_investigation=True)

    assert progress.readiness_mode == "open"
    assert progress.readiness_active is False
    assert progress.last_progress_reason == "readiness_action_window_exhausted"
    assert progress.take_replan_reason() == "有限动作窗口没有产生有效修改或验证"


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
    assert progress.readiness_guidance_for_turn(11) == (None, None)


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

    assert progress.last_tool_behaviors == frozenset({"investigate", "other"})
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


def test_validation_requires_a_started_process_before_it_counts_as_verify() -> None:
    permission_rejected = classify_tool_behavior(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"permission_status": "ask"},
    )
    start_failed = classify_tool_behavior(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["missing-validator"]},
        ok=False,
        metadata={"exit_code": None, "timed_out": False},
    )
    timed_out = classify_tool_behavior(
        tool_name="run_validation",
        capability="command",
        arguments={"command": ["pytest"]},
        ok=False,
        metadata={"exit_code": None, "timed_out": True},
    )

    assert permission_rejected == "other"
    assert start_failed == "other"
    assert timed_out == "verify"


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

    assert progress.task_phase == "DONE"
    assert progress.last_verification_succeeded is True
    assert progress.task_phase_history == [
        "INVESTIGATE",
        "VERIFY",
        "ACT",
        "VERIFY",
        "DONE",
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


def test_not_ready_returns_phase_to_investigation_and_final_answer_finishes() -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )
    assert progress.task_phase == "INVESTIGATE"

    progress.observe_final_answer()

    assert progress.task_phase == "DONE"


def test_turn_limit_resets_readiness_without_marking_task_done() -> None:
    progress = RunProgress(readonly_turn_limit=1)
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )

    progress.observe_turn_limit()

    assert progress.readiness_mode == "open"
    assert progress.task_phase == "INVESTIGATE"


def test_failed_followup_verification_reopens_done_phase() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=(
            ToolBehaviorObservation(behavior="act", succeeded=True),
            ToolBehaviorObservation(behavior="verify", succeeded=True),
        ),
    )
    assert progress.task_phase == "DONE"

    progress.observe_tool_turn(
        turn_number=2,
        observations=_observations("verify", succeeded=False),
    )

    assert progress.task_phase == "ACT"
    assert progress.last_verification_succeeded is False


def test_done_extra_tool_turn_is_observed_without_restricting_it() -> None:
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

    assert progress.task_phase == "DONE"
    assert progress.done_extra_tool_turn_count == 1
    assert progress.last_progress_reason == "done_extra_tool_observed"


def test_completion_correction_tracks_validation_after_latest_mutation() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("act", succeeded=True),
    )

    assert progress.intercept_unverified_completion_once() is True
    assert progress.completion_correction_used is True
    assert progress.last_progress_reason == "completion_correction_required"
    assert progress.take_completion_correction_guidance() is not None
    assert progress.take_completion_correction_guidance() is None
    assert progress.intercept_unverified_completion_once() is False


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
    assert progress.intercept_unverified_completion_once() is False


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
    assert progress.intercept_unverified_completion_once() is True


def test_no_formal_mutation_does_not_trigger_completion_correction() -> None:
    progress = RunProgress()
    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("verify", succeeded=False),
    )

    assert progress.intercept_unverified_completion_once() is False


def test_legacy_run_command_test_does_not_satisfy_completion_guard() -> None:
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

    assert progress.validated_revision == 0
    assert progress.intercept_unverified_completion_once() is True


def test_task_phase_guidance_tracks_phase_without_forcing_long_investigation() -> None:
    progress = RunProgress(readonly_turn_limit=1)

    investigate = progress.task_phase_guidance_for_turn()
    assert investigate is None

    progress.observe_tool_turn(
        turn_number=1,
        observations=_observations("investigate"),
    )

    progress.observe_tool_turn(
        turn_number=3,
        observations=_observations("act"),
    )
    assert "阶段=VERIFY" in progress.task_phase_guidance_for_turn()

    progress.observe_tool_turn(
        turn_number=4,
        observations=_observations("verify", succeeded=False),
    )
    assert "阶段=ACT" in progress.task_phase_guidance_for_turn()

    progress.observe_tool_turn(
        turn_number=5,
        observations=_observations("act"),
    )
    assert "阶段=VERIFY" in progress.task_phase_guidance_for_turn()

    progress.observe_tool_turn(
        turn_number=6,
        observations=_observations("verify"),
    )
    done = progress.task_phase_guidance_for_turn()
    assert done is not None
    assert "阶段=DONE" in done
    assert "当前修改已有成功验证结果" in done


def test_task_phase_guidance_is_disabled_with_main_readiness_for_subagents() -> None:
    progress = RunProgress(readonly_turn_limit=None)

    assert progress.task_phase_guidance_for_turn() is None
