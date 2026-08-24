import pytest
from pydantic import ValidationError

from mycode.subagents.contracts import (
    ExplorerFinding,
    ExplorerResult,
    ReviewFinding,
    ReviewerResult,
    SubAgentResult,
    SubAgentTask,
    TesterReport as TesterReportModel,
    TesterResult as TesterResultModel,
    ValidationExecution,
)


def test_subagent_task_normalizes_text_and_scope() -> None:
    task = SubAgentTask(
        role="explorer",
        objective="  locate the runner  ",
        context="  focus on tool calls  ",
        scope_paths=["  mycode/runner.py  "],
    )

    assert task.objective == "locate the runner"
    assert task.context == "focus on tool calls"
    assert task.scope_paths == ["mycode/runner.py"]


def test_subagent_task_rejects_blank_objective() -> None:
    with pytest.raises(ValidationError, match="objective must not be blank"):
        SubAgentTask(role="explorer", objective="   ")


def test_explorer_completed_result_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="at least one finding"):
        ExplorerResult(
            status="completed",
            summary="found it",
            searched_scope=["mycode"],
        )


def test_explorer_no_match_records_search_scope() -> None:
    result = ExplorerResult(
        status="no_match",
        summary="No matching implementation was found.",
        searched_scope=["mycode", "tests"],
        uncertainties=["Generated files were not searched."],
    )

    assert result.findings == []
    assert result.searched_scope == ["mycode", "tests"]


def test_explorer_finding_requires_nonblank_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must not be blank"):
        ExplorerFinding(
            path="mycode/runner.py",
            line=93,
            claim="The runner executes tool calls in order.",
            evidence="   ",
        )


def test_tester_report_requires_failure_or_blocked_details() -> None:
    with pytest.raises(ValidationError, match="failure_summary"):
        TesterReportModel(status="failed", summary="Tests failed.")
    with pytest.raises(ValidationError, match="blocked_reason"):
        TesterReportModel(status="blocked", summary="Tests did not run.")


def test_tester_passed_result_requires_real_successful_execution() -> None:
    with pytest.raises(ValidationError, match="successful execution"):
        TesterResultModel(status="passed", summary="Everything passed.")

    result = TesterResultModel(
        status="passed",
        summary="Everything passed.",
        executions=[
            ValidationExecution(
                command=["uv", "run", "pytest"],
                cwd=".",
                exit_code=0,
                duration_ms=125,
            )
        ],
    )

    assert result.executions[0].exit_code == 0


def test_validation_execution_rejects_oversized_command_evidence() -> None:
    with pytest.raises(ValidationError, match="total characters"):
        ValidationExecution(
            command=["python", "-m", "pytest", *["x" * 900 for _ in range(5)]],
            cwd="C:/workspace",
            exit_code=0,
            duration_ms=1,
        )


def test_tester_failed_result_requires_failed_execution() -> None:
    with pytest.raises(ValidationError, match="failed or timed-out execution"):
        TesterResultModel(
            status="failed",
            summary="Failure reported.",
            failure_summary="A test failed.",
            executions=[
                ValidationExecution(
                    command=["pytest"],
                    cwd=".",
                    exit_code=0,
                    duration_ms=10,
                )
            ],
        )


def test_reviewer_changes_requested_requires_finding() -> None:
    with pytest.raises(ValidationError, match="at least one finding"):
        ReviewerResult(
            recommendation="changes_requested",
            summary="Changes are needed.",
            reviewed_scope=["mycode/runner.py"],
        )


def test_reviewer_approve_rejects_high_severity_finding() -> None:
    finding = ReviewFinding(
        severity="high",
        path="mycode/runner.py",
        line=93,
        problem="A stale call may run.",
        evidence="All calls are generated before any result is returned.",
        suggestion="Add a control-flow barrier.",
    )

    with pytest.raises(ValidationError, match="approve cannot include"):
        ReviewerResult(
            recommendation="approve",
            summary="Looks good.",
            reviewed_scope=["mycode/runner.py"],
            findings=[finding],
        )


def test_bounded_results_require_consistent_truncation_metadata() -> None:
    with pytest.raises(ValidationError, match="truncated must be true"):
        ExplorerResult(
            status="no_match",
            summary="No match.",
            searched_scope=["mycode"],
            omitted_count=2,
        )


def test_subagent_result_requires_payload_matching_role() -> None:
    reviewer_payload = ReviewerResult(
        recommendation="approve",
        summary="No issues found.",
        reviewed_scope=["mycode/runner.py"],
    )

    with pytest.raises(ValidationError, match="payload does not match"):
        SubAgentResult(
            run_id="run-1",
            role="explorer",
            status="completed",
            stop_reason="submitted",
            summary="Done.",
            payload=reviewer_payload,
        )


def test_subagent_failed_result_cannot_include_payload() -> None:
    explorer_payload = ExplorerResult(
        status="completed",
        summary="Found the runner.",
        searched_scope=["mycode/runner.py"],
        findings=[
            ExplorerFinding(
                path="mycode/runner.py",
                line=93,
                claim="Tool calls run sequentially.",
                evidence="The loop calls registry.run_tool for every tool call.",
            )
        ],
    )

    with pytest.raises(ValidationError, match="cannot include payload"):
        SubAgentResult(
            run_id="run-2",
            role="explorer",
            status="failed",
            stop_reason="model_error",
            summary="Failed.",
            payload=explorer_payload,
        )
