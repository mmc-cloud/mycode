from pathlib import Path
import sys

import pytest

from mycode.agent import AgentModelResponse, AgentToolCall
from mycode.llm import FakeLLMClient
from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.runner import AgentRunner
from mycode.tools import Workspace, create_default_tool_registry


@pytest.mark.parametrize(
    ("case", "expected_success"),
    [
        ("pytest", True),
        ("django_runner", True),
        ("project_script", True),
        ("python_inline", True),
        ("python_inline_failure", False),
        ("timeout", False),
    ],
)
def test_real_main_agent_validation_reaches_run_progress(
    tmp_path: Path,
    case: str,
    expected_success: bool,
) -> None:
    command, timeout_seconds = _prepare_validation_command(tmp_path, case)
    registry = create_default_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )
    llm_client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id=f"call_{case}",
                        name="run_validation",
                        arguments={
                            "command": command,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="validation recorded"),
        ],
    )
    runner = AgentRunner(llm_client=llm_client, tool_registry=registry)

    events = list(runner.run("run the relevant validation"))

    assert [event.content for event in events if event.type == "text_delta"] == [
        "validation recorded"
    ]
    assert runner.last_run_progress is not None
    assert runner.last_run_progress.verification_turn_count == 1
    assert runner.last_run_progress.last_verification_succeeded is expected_success


def test_process_start_failure_does_not_reach_verify(
    tmp_path: Path,
) -> None:
    registry = create_default_tool_registry(
        Workspace(tmp_path),
        confirmer=ApprovingConfirmer(),
    )
    llm_client = FakeLLMClient(
        responses=[],
        tool_responses=[
            AgentModelResponse(
                tool_calls=[
                    AgentToolCall(
                        id="call_missing_validator",
                        name="run_validation",
                        arguments={"command": ["missing-validator-for-mycode-tests"]},
                    )
                ],
                stop_reason="tool_calls",
            ),
            AgentModelResponse(content="validator could not start"),
        ],
    )
    runner = AgentRunner(llm_client=llm_client, tool_registry=registry)

    list(runner.run("run the relevant validation"))

    assert runner.last_run_progress is not None
    assert runner.last_run_progress.verification_turn_count == 0
    assert runner.last_run_progress.last_verification_succeeded is None


def _prepare_validation_command(
    workspace: Path,
    case: str,
) -> tuple[list[str], float]:
    if case == "pytest":
        (workspace / "test_sample.py").write_text(
            "def test_sample():\n    assert True\n",
            encoding="utf-8",
        )
        return [sys.executable, "-m", "pytest", "-q", "test_sample.py"], 30.0

    if case == "django_runner":
        (workspace / "manage.py").write_text(
            "import sys\nraise SystemExit(0 if sys.argv[1:] == ['test'] else 2)\n",
            encoding="utf-8",
        )
        return [sys.executable, "manage.py", "test"], 30.0

    if case == "project_script":
        tests_directory = workspace / "tests"
        tests_directory.mkdir()
        (tests_directory / "runtests.py").write_text(
            "raise SystemExit(0)\n",
            encoding="utf-8",
        )
        return [sys.executable, "tests/runtests.py"], 30.0

    if case == "python_inline":
        return [sys.executable, "-c", "raise SystemExit(0)"], 30.0

    if case == "python_inline_failure":
        return [sys.executable, "-c", "raise SystemExit(3)"], 30.0

    if case == "timeout":
        return [sys.executable, "-c", "import time; time.sleep(5)"], 0.05

    raise AssertionError(f"Unsupported validation test case: {case}")


class ApprovingConfirmer:
    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        return ConfirmationResult.approved()
