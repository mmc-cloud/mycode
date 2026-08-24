from collections.abc import Sequence

from mycode.subagents.contracts import (
    BoundedResultArgs,
    ExplorerResult,
    ReviewerResult,
    SubAgentPayload,
    SubAgentRole,
    TesterReport,
    TesterResult,
    ValidationExecution,
)


DEFAULT_MAX_FINAL_PAYLOAD_CHARS = 12000


class ResultCompressionError(ValueError):
    pass


def finalize_submitted_payload(
    role: SubAgentRole,
    submitted: BoundedResultArgs,
    *,
    validation_executions: Sequence[ValidationExecution] = (),
    max_chars: int = DEFAULT_MAX_FINAL_PAYLOAD_CHARS,
) -> SubAgentPayload:
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")

    if role == "explorer" and isinstance(submitted, ExplorerResult):
        payload: SubAgentPayload = submitted
    elif role == "tester" and isinstance(submitted, TesterReport):
        payload = TesterResult(
            status=submitted.status,
            summary=submitted.summary,
            executions=list(validation_executions),
            failure_summary=submitted.failure_summary,
            blocked_reason=submitted.blocked_reason,
            uncertainties=submitted.uncertainties,
            truncated=submitted.truncated,
            omitted_count=submitted.omitted_count,
        )
    elif role == "reviewer" and isinstance(submitted, ReviewerResult):
        payload = submitted
    else:
        raise ValueError(f"Submitted result does not match SubAgent role: {role}")

    return compress_subagent_payload(payload, max_chars=max_chars)


def compress_subagent_payload(
    payload: SubAgentPayload,
    *,
    max_chars: int = DEFAULT_MAX_FINAL_PAYLOAD_CHARS,
) -> SubAgentPayload:
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if _serialized_chars(payload) <= max_chars:
        return payload

    candidate = _drop_uncertainties(payload, max_chars=max_chars)
    if _serialized_chars(candidate) <= max_chars:
        return candidate

    if isinstance(candidate, ExplorerResult):
        candidate = _compress_explorer(candidate, max_chars=max_chars)
    elif isinstance(candidate, ReviewerResult):
        candidate = _compress_reviewer(candidate, max_chars=max_chars)
    else:
        candidate = _compress_tester(candidate, max_chars=max_chars)

    if _serialized_chars(candidate) > max_chars:
        raise ResultCompressionError(
            "Structured result cannot fit the final payload budget without "
            "removing protected summary, location, command, exit-status, or "
            "error fields."
        )
    return candidate


def _drop_uncertainties(
    payload: SubAgentPayload,
    *,
    max_chars: int,
) -> SubAgentPayload:
    candidate = payload
    while candidate.uncertainties and _serialized_chars(candidate) > max_chars:
        candidate = _copy_with_omissions(
            candidate,
            uncertainties=candidate.uncertainties[:-1],
        )
    return candidate


def _compress_explorer(
    payload: ExplorerResult,
    *,
    max_chars: int,
) -> ExplorerResult:
    candidate = payload
    minimum_findings = 1 if candidate.findings else 0
    while (
        len(candidate.findings) > minimum_findings
        and _serialized_chars(candidate) > max_chars
    ):
        candidate = _copy_with_omissions(
            candidate,
            findings=candidate.findings[:-1],
        )
    return candidate


def _compress_reviewer(
    payload: ReviewerResult,
    *,
    max_chars: int,
) -> ReviewerResult:
    candidate = payload
    minimum_findings = 1 if candidate.findings else 0
    severity_priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    while (
        len(candidate.findings) > minimum_findings
        and _serialized_chars(candidate) > max_chars
    ):
        removable_index = max(
            range(len(candidate.findings)),
            key=lambda index: (severity_priority[candidate.findings[index].severity], index),
        )
        findings = list(candidate.findings)
        del findings[removable_index]
        candidate = _copy_with_omissions(candidate, findings=findings)
    return candidate


def _compress_tester(
    payload: TesterResult,
    *,
    max_chars: int,
) -> TesterResult:
    candidate = payload
    while len(candidate.executions) > 1 and _serialized_chars(candidate) > max_chars:
        removable_index = _tester_execution_to_remove(candidate)
        executions = list(candidate.executions)
        del executions[removable_index]
        candidate = _copy_with_omissions(candidate, executions=executions)
    return candidate


def _tester_execution_to_remove(payload: TesterResult) -> int:
    failed_indexes = [
        index
        for index, execution in enumerate(payload.executions)
        if execution.timed_out
        or execution.exit_code is None
        or execution.exit_code != 0
    ]
    successful_indexes = [
        index for index in range(len(payload.executions)) if index not in failed_indexes
    ]
    if payload.status in {"failed", "blocked"} and successful_indexes:
        return successful_indexes[-1]
    if payload.status == "passed":
        return successful_indexes[-1]
    return len(payload.executions) - 1


def _copy_with_omissions(payload, **updates):
    data = payload.model_dump()
    data.update(updates)
    data["truncated"] = True
    data["omitted_count"] = payload.omitted_count + 1
    return type(payload).model_validate(data)


def _serialized_chars(payload: SubAgentPayload) -> int:
    return len(payload.model_dump_json())
