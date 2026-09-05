from collections.abc import Callable
from typing import Generic, TypeVar

from mycode.permissions import ToolRisk
from mycode.subagents.contracts import BoundedResultArgs, SubAgentRole
from mycode.tools.base import PydanticTool, ToolResult


DEFAULT_MAX_SUBMITTED_RESULT_CHARS = 24000

ResultArgsT = TypeVar("ResultArgsT", bound=BoundedResultArgs)


class SubmitResultTool(PydanticTool[ResultArgsT], Generic[ResultArgsT]):
    name = "submit_result"
    description = "Submit the role-specific structured result and finish the SubAgent run."
    capability = "control"
    risk: ToolRisk = "low"

    def __init__(
        self,
        *,
        role: SubAgentRole,
        result_model: type[ResultArgsT],
        max_result_chars: int = DEFAULT_MAX_SUBMITTED_RESULT_CHARS,
        acceptance_validator: Callable[[ResultArgsT], str | None] | None = None,
    ) -> None:
        if max_result_chars < 1:
            raise ValueError("max_result_chars must be at least 1.")
        self.role = role
        self.args_model = result_model
        self.max_result_chars = max_result_chars
        self.acceptance_validator = acceptance_validator
        self.submitted_result: ResultArgsT | None = None

    def _run(self, args: ResultArgsT) -> ToolResult:
        if self.submitted_result is not None:
            return ToolResult.failure(
                error="A structured SubAgent result has already been submitted.",
                metadata={"role": self.role, "reason": "result_already_submitted"},
            )

        serialized = args.model_dump_json()
        result_chars = len(serialized)
        if result_chars > self.max_result_chars:
            return ToolResult.failure(
                error=(
                    "Structured SubAgent result exceeds the allowed size: "
                    f"{result_chars}/{self.max_result_chars} characters."
                ),
                metadata={
                    "role": self.role,
                    "reason": "result_too_large",
                    "result_chars": result_chars,
                    "max_result_chars": self.max_result_chars,
                },
            )

        if self.acceptance_validator is not None:
            try:
                validation_error = self.acceptance_validator(args)
            except Exception as error:
                return ToolResult.failure(
                    error="Runtime result validation failed unexpectedly.",
                    metadata={
                        "role": self.role,
                        "reason": "result_runtime_validation_error",
                        "exception_type": type(error).__name__,
                    },
                )
            if validation_error is not None:
                return ToolResult.failure(
                    error=validation_error,
                    metadata={
                        "role": self.role,
                        "reason": "result_runtime_validation_failed",
                    },
                )

        self.submitted_result = args
        outcome = getattr(args, "status", None)
        if outcome is None:
            outcome = getattr(args, "recommendation", None)
        return ToolResult.success(
            content="Structured SubAgent result accepted.",
            metadata={
                "role": self.role,
                "outcome": outcome,
                "result_chars": result_chars,
                "truncated": args.truncated,
                "omitted_count": args.omitted_count,
            },
        )
