import pytest
from pydantic import ValidationError

from mycode.subagents.contracts import (
    ExplorerFinding,
    ExplorerResult,
    ReviewFinding,
    ReviewerResult,
    TesterReport,
    TesterResult,
    ValidationExecution,
)
from mycode.subagents.results import (
    ResultCompressionError,
    compress_subagent_payload,
    finalize_submitted_payload,
)


def test_result_compression_drops_uncertainties_before_findings() -> None:
    payload = ExplorerResult(
        status="completed",
        summary="Located the implementation.",
        searched_scope=["mycode"],
        findings=[
            ExplorerFinding(
                path="mycode/runner.py",
                line=1,
                claim="The runner owns the loop.",
                evidence="evidence " * 60,
            )
        ],
        uncertainties=["uncertain " * 100, "another " * 100],
    )
    protected_size = len(
        payload.model_copy(
            update={"uncertainties": [], "truncated": True, "omitted_count": 2}
        ).model_dump_json()
    )

    compressed = compress_subagent_payload(payload, max_chars=protected_size)

    assert compressed.findings == payload.findings
    assert compressed.uncertainties == []
    assert compressed.truncated is True
    assert compressed.omitted_count == 2


def test_reviewer_compression_preserves_higher_severity_finding() -> None:
    high = _review_finding("high", "critical behavior " * 50)
    low = _review_finding("low", "minor behavior " * 50)
    payload = ReviewerResult(
        recommendation="changes_requested",
        summary="Changes are required.",
        reviewed_scope=["mycode"],
        findings=[low, high],
    )
    one_finding_budget = len(
        ReviewerResult(
            recommendation="changes_requested",
            summary=payload.summary,
            reviewed_scope=payload.reviewed_scope,
            findings=[high],
            truncated=True,
            omitted_count=1,
        ).model_dump_json()
    )

    compressed = compress_subagent_payload(payload, max_chars=one_finding_budget)

    assert [finding.severity for finding in compressed.findings] == ["high"]
    assert compressed.omitted_count == 1


def test_tester_compression_keeps_failed_execution_before_successful_evidence() -> None:
    executions = [
        ValidationExecution(
            command=["python", "-m", "py_compile", f"file-{index}.py", "x" * 300],
            cwd="C:/workspace",
            exit_code=0,
            duration_ms=1,
        )
        for index in range(3)
    ]
    failed = ValidationExecution(
        command=["python", "-m", "pytest", "tests/test_failure.py", "y" * 300],
        cwd="C:/workspace",
        exit_code=1,
        duration_ms=2,
    )
    payload = TesterResult(
        status="failed",
        summary="One validation failed.",
        executions=[*executions, failed],
        failure_summary="The failure is reproducible.",
    )
    one_execution_budget = len(
        TesterResult(
            status="failed",
            summary=payload.summary,
            executions=[failed],
            failure_summary=payload.failure_summary,
            truncated=True,
            omitted_count=3,
        ).model_dump_json()
    )

    compressed = compress_subagent_payload(payload, max_chars=one_execution_budget)

    assert compressed.executions == [failed]
    assert compressed.omitted_count == 3


def test_tester_finalization_rejects_passed_report_without_real_execution() -> None:
    report = TesterReport(status="passed", summary="Everything passed.")

    with pytest.raises(ValidationError, match="successful execution"):
        finalize_submitted_payload("tester", report)


def test_result_compression_fails_instead_of_cutting_protected_fields() -> None:
    payload = ExplorerResult(
        status="no_match",
        summary="No implementation matched the requested behavior.",
        searched_scope=["mycode"],
    )

    with pytest.raises(ResultCompressionError, match="protected"):
        compress_subagent_payload(payload, max_chars=10)


def _review_finding(severity, evidence: str) -> ReviewFinding:
    return ReviewFinding(
        severity=severity,
        path="mycode/runtime.py",
        line=10,
        problem=f"{severity} problem",
        evidence=evidence,
        suggestion=f"Fix the {severity} problem.",
    )
