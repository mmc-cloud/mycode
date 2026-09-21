"""Tests for the opt-in startup profiler."""

import pytest

from mycode.application.agent_session import start_agent_application_session
from mycode.application.runtime import build_agent_runner
from mycode.application.sessions import SessionStartRequest, list_project_sessions
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig
from mycode.persistence.session_store import SessionStore
from mycode.project import ProjectIdentity
from mycode.startup_profile import (
    STARTUP_PROFILE_ENV_VAR,
    STARTUP_PROFILE_FILE_ENV_VAR,
    StartupProfiler,
    startup_profiling_enabled,
    write_startup_profile_line,
)
from mycode.tools.registry import ToolRegistry


class FakeClock:
    """Deterministic stand-in for time.perf_counter."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class FakeMCPManager:
    def __init__(self, config, observability_sink=None) -> None:
        self.config = config
        self.statuses = ()
        self.tools = ()
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FakeRunner:
    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()


@pytest.fixture(autouse=True)
def clean_profile_environment(monkeypatch):
    """Every test starts with startup profiling switched off."""
    monkeypatch.delenv(STARTUP_PROFILE_ENV_VAR, raising=False)
    monkeypatch.delenv(STARTUP_PROFILE_FILE_ENV_VAR, raising=False)


def _profiler(clock, lines, **kwargs) -> StartupProfiler:
    return StartupProfiler(clock=clock, write=lines.append, **kwargs)


def _stage_names(output: str) -> list[str]:
    return [
        line.split(": ", 1)[0].removeprefix("[startup] ")
        for line in output.splitlines()
    ]


def configured_llm() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_profiling_switch_accepts_documented_values(monkeypatch, value) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, value)
    assert startup_profiling_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2", "yes please"])
def test_profiling_switch_rejects_other_values(monkeypatch, value) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, value)
    assert startup_profiling_enabled() is False


def test_profiling_is_off_when_the_variable_is_absent() -> None:
    assert startup_profiling_enabled() is False


def test_disabled_profiler_writes_nothing_and_never_reads_the_clock() -> None:
    clock = FakeClock()
    lines: list[str] = []
    profiler = _profiler(clock, lines)
    calls_after_init = clock.calls

    with profiler.span("session.list"):
        clock.advance(1.0)

    assert profiler.enabled is False
    assert profiler.total("runtime.ready") is None
    assert lines == []
    assert clock.calls == calls_after_init


def test_enabled_profiler_reports_stage_duration_and_elapsed(monkeypatch) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    clock = FakeClock()
    lines: list[str] = []
    profiler = _profiler(clock, lines)

    clock.advance(0.25)
    with profiler.span("mcp.start"):
        clock.advance(2.4507)

    assert lines == ["[startup] mcp.start: 2450.7 ms (elapsed 2700.7 ms)"]


def test_total_reports_the_cumulative_elapsed(monkeypatch) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    clock = FakeClock()
    lines: list[str] = []
    profiler = _profiler(clock, lines)

    clock.advance(3.4567)

    assert profiler.total("runtime.ready") == pytest.approx(3456.7, abs=0.05)
    assert lines == ["[startup] runtime.ready: 3456.7 ms total"]


def test_consecutive_spans_keep_their_own_duration(monkeypatch) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    clock = FakeClock()
    lines: list[str] = []
    profiler = _profiler(clock, lines)

    with profiler.span("session.list"):
        clock.advance(0.1)
    with profiler.span("mcp.start"):
        clock.advance(0.4)

    assert lines == [
        "[startup] session.list: 100.0 ms (elapsed 100.0 ms)",
        "[startup] mcp.start: 400.0 ms (elapsed 500.0 ms)",
    ]


def test_span_reports_the_stage_even_when_the_block_raises(monkeypatch) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    clock = FakeClock()
    lines: list[str] = []
    profiler = _profiler(clock, lines)

    with pytest.raises(RuntimeError, match="boom"):
        with profiler.span("runner.build"):
            clock.advance(0.5)
            raise RuntimeError("boom")

    assert lines == ["[startup] runner.build: 500.0 ms (elapsed 500.0 ms)"]


def test_explicit_flag_overrides_the_environment(monkeypatch) -> None:
    clock = FakeClock()
    lines: list[str] = []

    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    forced_off = _profiler(clock, lines, enabled=False)
    with forced_off.span("session.list"):
        pass
    assert lines == []

    forced_on = _profiler(clock, lines, enabled=True)
    with forced_on.span("session.list"):
        clock.advance(0.05)
    assert lines == ["[startup] session.list: 50.0 ms (elapsed 50.0 ms)"]


def test_output_failure_never_reaches_the_caller(monkeypatch) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")

    def broken_write(_: str) -> None:
        raise OSError("broken pipe")

    profiler = StartupProfiler(clock=FakeClock(), write=broken_write)

    with profiler.span("session.list"):
        pass
    profiler.total("runtime.ready")
    profiler.record("runner.build", 1.0)


def test_lines_go_to_stdout_without_a_profile_file(capsys) -> None:
    write_startup_profile_line("[startup] session.list: 1.0 ms")
    assert capsys.readouterr().out == "[startup] session.list: 1.0 ms\n"


def test_profile_file_receives_the_lines(monkeypatch, tmp_path, capsys) -> None:
    path = tmp_path / "startup-profile.log"
    monkeypatch.setenv(STARTUP_PROFILE_FILE_ENV_VAR, str(path))

    write_startup_profile_line("[startup] first")
    write_startup_profile_line("[startup] second")

    assert path.read_text(encoding="utf-8") == "[startup] first\n[startup] second\n"
    assert capsys.readouterr().out == ""


def test_unwritable_profile_file_only_loses_the_line(monkeypatch, tmp_path) -> None:
    target = tmp_path / "missing-directory" / "startup-profile.log"
    monkeypatch.setenv(STARTUP_PROFILE_FILE_ENV_VAR, str(target))

    write_startup_profile_line("[startup] lost")

    assert not target.exists()


def test_session_list_is_silent_while_profiling_is_off(tmp_path, capsys) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)

    assert list_project_sessions(store, project) == []
    assert capsys.readouterr().out == ""


def test_session_list_is_profiled_when_enabled(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)

    assert list_project_sessions(store, project) == []

    assert _stage_names(capsys.readouterr().out) == ["session.list"]


def test_build_agent_runner_reports_its_stages(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")

    build_agent_runner(workspace_path=tmp_path, llm_config=configured_llm())

    assert _stage_names(capsys.readouterr().out) == [
        "runner.workspace",
        "runner.memory_store",
        "runner.instructions",
        "runner.skills",
        "runner.config",
        "runner.llm_client",
        "runner.budget",
        "runner.subagent_runtime",
        "runner.conversation",
        "runner.artifacts",
        "runner.tools",
        "runner.finalize",
    ]


def test_application_session_reports_its_stages(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(STARTUP_PROFILE_ENV_VAR, "1")
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    existing = store.create_session(project, session_id="session")
    monkeypatch.setattr(
        "mycode.application.agent_session.MCPManager",
        FakeMCPManager,
    )
    monkeypatch.setattr(
        "mycode.application.agent_session.build_agent_runner",
        lambda **kwargs: FakeRunner(),
    )

    application_session = start_agent_application_session(
        store,
        project,
        request=SessionStartRequest(mode="resume", session_id=existing.id),
        mcp_config=MCPConfig(),
        llm_config=configured_llm(),
    )
    application_session.close()

    assert _stage_names(capsys.readouterr().out) == [
        "session.open",
        "session.history",
        "session.compact_state",
        "mcp.start",
        "runner.build",
        "mcp.tools",
        "session.startup",
    ]
