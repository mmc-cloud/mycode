"""Opt-in startup profiling for the ``mycode agent`` and ``mycode tui`` flows.

The profiler is inert unless ``MYCODE_STARTUP_PROFILE`` is set to a truthy
value, so normal runs never pay for a clock call or print a line.

Two numbers are reported per stage:

- ``duration``: how long that stage itself took;
- ``elapsed``: how long the whole flow has been running when the stage ended.

``elapsed`` is measured from the moment this module is imported. Because
``mycode.cli`` imports it before every heavy dependency, that moment is close to
the start of the CLI entry point: the elapsed value therefore includes the
remaining ``mycode.cli`` imports. The Python interpreter start-up and the site
bootstrap happen before any ``mycode`` import and are not included; use
``python -X importtime`` for that lower level view.

The helper only uses the standard library, never raises, and must never be able
to change startup behaviour.

Two flows share stdout with something else and should be profiled through
``MYCODE_STARTUP_PROFILE_FILE`` instead: the Textual app captures ``print()``
while it runs, and ``mycode runtime --jsonl`` reserves stdout for its machine
protocol.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from threading import Lock
from time import perf_counter

STARTUP_PROFILE_ENV_VAR = "MYCODE_STARTUP_PROFILE"
STARTUP_PROFILE_FILE_ENV_VAR = "MYCODE_STARTUP_PROFILE_FILE"
STARTUP_PROFILE_LABEL = "[startup]"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_PROFILE_WRITE_LOCK = Lock()


def startup_profiling_enabled() -> bool:
    """Return whether ``MYCODE_STARTUP_PROFILE`` asks for startup profiling."""
    value = os.environ.get(STARTUP_PROFILE_ENV_VAR, "")
    return value.strip().casefold() in _TRUE_VALUES


def write_startup_profile_line(line: str) -> None:
    """Write one profiling line to stdout.

    When ``MYCODE_STARTUP_PROFILE_FILE`` names a file the line is appended there
    instead. The Textual app captures ``print()`` while it runs, so ``mycode
    tui`` needs that file to make its timings readable.
    """
    path = os.environ.get(STARTUP_PROFILE_FILE_ENV_VAR, "").strip()
    with _PROFILE_WRITE_LOCK:
        if path == "":
            print(line)
            return
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            # A bad profile path may only lose the line, never break startup.
            return


class StartupProfiler:
    """Measure named startup stages against one shared origin."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        clock: Callable[[], float] = perf_counter,
        write: Callable[[str], None] = write_startup_profile_line,
    ) -> None:
        self._enabled = enabled
        self._clock = clock
        self._write = write
        self._origin = clock()

    @property
    def enabled(self) -> bool:
        """Profiling is on when forced by the caller or by the environment."""
        if self._enabled is not None:
            return self._enabled
        return startup_profiling_enabled()

    @property
    def origin(self) -> float:
        return self._origin

    def elapsed_ms(self) -> float:
        """Milliseconds between this profiler's origin and now."""
        return (self._now() - self._origin) * 1000.0

    def _now(self) -> float:
        return self._clock()

    def span(self, stage: str) -> AbstractContextManager[None]:
        """Time one stage as a ``with`` block; a no-op while profiling is off."""
        if not self.enabled:
            return _NULL_SPAN
        return _StageSpan(self, stage)

    def total(self, stage: str) -> float | None:
        """Report the cumulative elapsed time reached at one checkpoint."""
        if not self.enabled:
            return None
        elapsed = self.elapsed_ms()
        self.report(f"{stage}: {elapsed:.1f} ms total")
        return elapsed

    def record(self, stage: str, duration_ms: float) -> None:
        """Report one already measured stage duration."""
        if not self.enabled:
            return
        elapsed = self.elapsed_ms()
        self.report(f"{stage}: {duration_ms:.1f} ms (elapsed {elapsed:.1f} ms)")

    def report(self, message: str) -> None:
        """Emit one preformatted message; output failures are swallowed."""
        try:
            self._write(f"{STARTUP_PROFILE_LABEL} {message}")
        except Exception:  # noqa: BLE001 - profiling must not break startup
            return


class _StageSpan:
    """Context manager returned by :meth:`StartupProfiler.span`."""

    __slots__ = ("_profiler", "_stage", "_started")

    def __init__(self, profiler: StartupProfiler, stage: str) -> None:
        self._profiler = profiler
        self._stage = stage
        self._started = profiler._now()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        duration_ms = (self._profiler._now() - self._started) * 1000.0
        self._profiler.record(self._stage, duration_ms)
        # False keeps every exception and return value untouched.
        return False


class _NullSpan:
    """Reusable no-op span used while startup profiling is disabled."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


_NULL_SPAN = _NullSpan()

_PROCESS_PROFILER = StartupProfiler()


def get_startup_profiler() -> StartupProfiler:
    """Return the profiler shared by the CLI and TUI startup flows."""
    return _PROCESS_PROFILER


def span(stage: str) -> AbstractContextManager[None]:
    """Time one stage of the process-wide startup flow."""
    return _PROCESS_PROFILER.span(stage)


def total(stage: str) -> float | None:
    """Report a cumulative checkpoint of the process-wide startup flow."""
    return _PROCESS_PROFILER.total(stage)
