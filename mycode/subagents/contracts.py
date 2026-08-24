from typing import ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator

from mycode.subagents.limits import (
    MAX_VALIDATION_COMMAND_CHARS,
    MAX_VALIDATION_COMMAND_PART_CHARS,
    MAX_VALIDATION_COMMAND_PARTS,
    MAX_VALIDATION_EXECUTIONS,
)
from mycode.tools.base import ToolArgs


MAX_TASK_CHARS = 4000
MAX_CONTEXT_CHARS = 4000
MAX_SUMMARY_CHARS = 2000
MAX_DETAIL_CHARS = 2000
MAX_PATH_CHARS = 500
MAX_SCOPE_ITEMS = 50
MAX_FINDINGS = 40
MAX_UNCERTAINTIES = 20

SubAgentRole = Literal["explorer", "tester", "reviewer"]


class SubAgentTask(ToolArgs):
    role: SubAgentRole
    objective: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    context: str = Field(default="", max_length=MAX_CONTEXT_CHARS)
    scope_paths: list[str] = Field(default_factory=list, max_length=MAX_SCOPE_ITEMS)

    @field_validator("objective")
    @classmethod
    def objective_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="objective")

    @field_validator("context")
    @classmethod
    def normalize_context(cls, value: str) -> str:
        return value.strip()

    @field_validator("scope_paths")
    @classmethod
    def validate_scope_paths(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(value, field_name="scope_paths", max_chars=MAX_PATH_CHARS)


class BoundedResultArgs(ToolArgs):
    truncated: bool = False
    omitted_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def truncation_metadata_must_match(self) -> Self:
        if self.truncated != (self.omitted_count > 0):
            raise ValueError(
                "truncated must be true exactly when omitted_count is above 0."
            )
        return self


class ExplorerFinding(ToolArgs):
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS)
    line: int | None = Field(default=None, ge=1)
    symbol: str | None = Field(default=None, max_length=200)
    claim: str = Field(min_length=1, max_length=MAX_DETAIL_CHARS)
    evidence: str = Field(min_length=1, max_length=MAX_DETAIL_CHARS)

    @field_validator("path", "claim", "evidence")
    @classmethod
    def required_fields_must_not_be_blank(cls, value: str, info) -> str:
        return _required_text(value, field_name=info.field_name)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        return _optional_text(value)


ExplorerStatus = Literal["completed", "partial", "blocked", "no_match"]


class ExplorerResult(BoundedResultArgs):
    status: ExplorerStatus
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    searched_scope: list[str] = Field(min_length=1, max_length=MAX_SCOPE_ITEMS)
    findings: list[ExplorerFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    uncertainties: list[str] = Field(default_factory=list, max_length=MAX_UNCERTAINTIES)
    blocked_reason: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="summary")

    @field_validator("searched_scope")
    @classmethod
    def validate_searched_scope(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="searched_scope",
            max_chars=MAX_PATH_CHARS,
        )

    @field_validator("uncertainties")
    @classmethod
    def validate_uncertainties(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="uncertainties",
            max_chars=MAX_DETAIL_CHARS,
        )

    @field_validator("blocked_reason")
    @classmethod
    def normalize_blocked_reason(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def status_must_match_findings(self) -> Self:
        if self.status == "completed" and not self.findings:
            raise ValueError("completed Explorer results require at least one finding.")
        if self.status == "no_match" and self.findings:
            raise ValueError("no_match Explorer results must not contain findings.")
        if self.status == "blocked" and self.blocked_reason is None:
            raise ValueError("blocked Explorer results require blocked_reason.")
        if self.status == "partial" and not self.findings and self.blocked_reason is None:
            raise ValueError(
                "partial Explorer results require a finding or blocked_reason."
            )
        return self


TesterStatus = Literal["passed", "failed", "blocked"]


class TesterReport(BoundedResultArgs):
    __test__: ClassVar[bool] = False

    status: TesterStatus
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    failure_summary: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)
    blocked_reason: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)
    uncertainties: list[str] = Field(default_factory=list, max_length=MAX_UNCERTAINTIES)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="summary")

    @field_validator("failure_summary", "blocked_reason")
    @classmethod
    def normalize_optional_details(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("uncertainties")
    @classmethod
    def validate_uncertainties(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="uncertainties",
            max_chars=MAX_DETAIL_CHARS,
        )

    @model_validator(mode="after")
    def status_must_match_details(self) -> Self:
        if self.status == "failed" and self.failure_summary is None:
            raise ValueError("failed Tester reports require failure_summary.")
        if self.status == "blocked" and self.blocked_reason is None:
            raise ValueError("blocked Tester reports require blocked_reason.")
        if self.status == "passed" and (
            self.failure_summary is not None or self.blocked_reason is not None
        ):
            raise ValueError(
                "passed Tester reports cannot include failure_summary or blocked_reason."
            )
        return self


class ValidationExecution(ToolArgs):
    command: list[str] = Field(
        min_length=1,
        max_length=MAX_VALIDATION_COMMAND_PARTS,
    )
    cwd: str = Field(min_length=1, max_length=MAX_PATH_CHARS)
    exit_code: int | None = None
    duration_ms: int = Field(ge=0)
    timed_out: bool = False

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        normalized = _bounded_text_list(
            value,
            field_name="command",
            max_chars=MAX_VALIDATION_COMMAND_PART_CHARS,
        )
        if sum(len(part) for part in normalized) > MAX_VALIDATION_COMMAND_CHARS:
            raise ValueError(
                "command must not exceed "
                f"{MAX_VALIDATION_COMMAND_CHARS} total characters."
            )
        return normalized

    @field_validator("cwd")
    @classmethod
    def cwd_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="cwd")


class TesterResult(BoundedResultArgs):
    __test__: ClassVar[bool] = False

    status: TesterStatus
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    executions: list[ValidationExecution] = Field(
        default_factory=list,
        max_length=MAX_VALIDATION_EXECUTIONS,
    )
    failure_summary: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)
    blocked_reason: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)
    uncertainties: list[str] = Field(default_factory=list, max_length=MAX_UNCERTAINTIES)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="summary")

    @field_validator("failure_summary", "blocked_reason")
    @classmethod
    def normalize_optional_details(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("uncertainties")
    @classmethod
    def validate_uncertainties(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="uncertainties",
            max_chars=MAX_DETAIL_CHARS,
        )

    @model_validator(mode="after")
    def status_must_match_executions(self) -> Self:
        failed_executions = [
            item
            for item in self.executions
            if item.timed_out or item.exit_code is None or item.exit_code != 0
        ]
        if self.status == "passed":
            if not self.executions or failed_executions:
                raise ValueError(
                    "passed Tester results require at least one successful execution."
                )
            if self.failure_summary is not None or self.blocked_reason is not None:
                raise ValueError(
                    "passed Tester results cannot include failure_summary or blocked_reason."
                )
        if self.status == "failed":
            if not failed_executions:
                raise ValueError(
                    "failed Tester results require a failed or timed-out execution."
                )
            if self.failure_summary is None:
                raise ValueError("failed Tester results require failure_summary.")
        if self.status == "blocked" and self.blocked_reason is None:
            raise ValueError("blocked Tester results require blocked_reason.")
        return self


ReviewSeverity = Literal["critical", "high", "medium", "low"]
ReviewRecommendation = Literal["approve", "changes_requested", "blocked"]


class ReviewFinding(ToolArgs):
    severity: ReviewSeverity
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS)
    line: int | None = Field(default=None, ge=1)
    problem: str = Field(min_length=1, max_length=MAX_DETAIL_CHARS)
    evidence: str = Field(min_length=1, max_length=MAX_DETAIL_CHARS)
    suggestion: str = Field(min_length=1, max_length=MAX_DETAIL_CHARS)

    @field_validator("path", "problem", "evidence", "suggestion")
    @classmethod
    def required_fields_must_not_be_blank(cls, value: str, info) -> str:
        return _required_text(value, field_name=info.field_name)


class ReviewerResult(BoundedResultArgs):
    recommendation: ReviewRecommendation
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    reviewed_scope: list[str] = Field(min_length=1, max_length=MAX_SCOPE_ITEMS)
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    uncertainties: list[str] = Field(default_factory=list, max_length=MAX_UNCERTAINTIES)
    blocked_reason: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        return _required_text(value, field_name="summary")

    @field_validator("reviewed_scope")
    @classmethod
    def validate_reviewed_scope(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="reviewed_scope",
            max_chars=MAX_PATH_CHARS,
        )

    @field_validator("uncertainties")
    @classmethod
    def validate_uncertainties(cls, value: list[str]) -> list[str]:
        return _bounded_text_list(
            value,
            field_name="uncertainties",
            max_chars=MAX_DETAIL_CHARS,
        )

    @field_validator("blocked_reason")
    @classmethod
    def normalize_blocked_reason(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def recommendation_must_match_findings(self) -> Self:
        if self.recommendation == "changes_requested" and not self.findings:
            raise ValueError("changes_requested requires at least one finding.")
        if self.recommendation == "blocked" and self.blocked_reason is None:
            raise ValueError("blocked Reviewer results require blocked_reason.")
        if self.recommendation == "approve" and any(
            finding.severity in {"critical", "high"}
            for finding in self.findings
        ):
            raise ValueError(
                "approve cannot include critical or high severity findings."
            )
        return self


SubAgentEnvelopeStatus = Literal["completed", "failed", "interrupted"]
SubAgentStopReason = Literal[
    "submitted",
    "max_turns",
    "invalid_result",
    "context_overflow",
    "model_error",
    "runtime_error",
    "repeated_tool_call",
    "interrupted",
]
SubAgentPayload = ExplorerResult | TesterResult | ReviewerResult


class SubAgentResult(ToolArgs):
    run_id: str = Field(min_length=1, max_length=100)
    role: SubAgentRole
    status: SubAgentEnvelopeStatus
    stop_reason: SubAgentStopReason
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    payload: SubAgentPayload | None = None
    error: str | None = Field(default=None, max_length=MAX_DETAIL_CHARS)

    @field_validator("run_id", "summary")
    @classmethod
    def required_fields_must_not_be_blank(cls, value: str, info) -> str:
        return _required_text(value, field_name=info.field_name)

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def envelope_must_match_status_and_role(self) -> Self:
        if self.status == "completed":
            if self.stop_reason != "submitted" or self.payload is None:
                raise ValueError(
                    "completed SubAgent results require submitted payload."
                )
        elif self.payload is not None:
            raise ValueError("failed or interrupted SubAgent results cannot include payload.")

        expected_type = {
            "explorer": ExplorerResult,
            "tester": TesterResult,
            "reviewer": ReviewerResult,
        }[self.role]
        if self.payload is not None and not isinstance(self.payload, expected_type):
            raise ValueError(f"payload does not match SubAgent role: {self.role}")
        return self


def _required_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be blank.")
    return normalized


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return None if normalized == "" else normalized


def _bounded_text_list(
    values: list[str],
    *,
    field_name: str,
    max_chars: int,
) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = _required_text(value, field_name=field_name)
        if len(item) > max_chars:
            raise ValueError(
                f"{field_name} items must not exceed {max_chars} characters."
            )
        normalized.append(item)
    return normalized
