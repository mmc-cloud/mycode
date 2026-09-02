import subprocess
from dataclasses import dataclass
from pathlib import Path
import time

from mycode.tools.base import ToolResult
from mycode.tools.command_output import (
    BoundedOutputCapture,
    build_output_metadata,
    finish_output_threads,
    start_output_threads,
)
from mycode.tools.process_tree import (
    create_process_tree_cleanup,
    process_tree_popen_kwargs,
)


@dataclass(frozen=True)
class CommandExecutionArgs:
    command: list[str]
    timeout_seconds: float
    max_output_chars: int


def execute_command(
    *,
    args: CommandExecutionArgs,
    cwd: Path,
    permission_metadata: dict[str, object],
    permission_status: str,
) -> ToolResult:
    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            args.command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            **process_tree_popen_kwargs(),
        )
    except OSError as error:
        return ToolResult.failure(
            error=f"Command failed to start: {error}",
            metadata={
                **permission_metadata,
                "exit_code": None,
                "timed_out": False,
                "duration_ms": _elapsed_milliseconds(started_at),
                "permission_status": permission_status,
                "exception_type": type(error).__name__,
            },
        )

    process_tree_cleanup = create_process_tree_cleanup(process)
    stdout_capture = BoundedOutputCapture(args.max_output_chars)
    stderr_capture = BoundedOutputCapture(args.max_output_chars)
    output_threads = start_output_threads(
        process=process,
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
    )

    timed_out = False
    output_capture_complete = True
    process_tree_cleanup_success = True
    returncode: int | None
    try:
        returncode = process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        process_tree_cleanup_success = process_tree_cleanup.close()
        if not process_tree_cleanup_success:
            process.kill()
        process.wait()
    finally:
        process_tree_cleanup_success = (
            process_tree_cleanup.close() and process_tree_cleanup_success
        )
        output_capture_complete = finish_output_threads(process, output_threads)

    stdout = stdout_capture.snapshot()
    stderr = stderr_capture.snapshot()
    output_metadata = build_output_metadata(stdout=stdout, stderr=stderr)

    metadata = {
        **permission_metadata,
        **output_metadata,
        "exit_code": returncode,
        "timed_out": timed_out,
        "duration_ms": _elapsed_milliseconds(started_at),
        "output_capture_complete": output_capture_complete,
        "process_tree_cleanup_method": process_tree_cleanup.method,
        "process_tree_cleanup_success": process_tree_cleanup_success,
        "permission_status": permission_status,
    }
    if process_tree_cleanup.error is not None:
        metadata["process_tree_cleanup_error"] = process_tree_cleanup.error

    if timed_out:
        timeout_error = f"Command timed out after {args.timeout_seconds:g} seconds."
        captured_output = _format_captured_output(
            stdout=stdout.content,
            stderr=stderr.content,
        )
        return ToolResult.failure(
            error=(
                timeout_error
                if captured_output == ""
                else f"{timeout_error}\n\n{captured_output}"
            ),
            metadata=metadata,
        )

    content = _format_command_result(
        exit_code=returncode,
        stdout=stdout.content,
        stderr=stderr.content,
    )

    if returncode != 0:
        return ToolResult.failure(
            error=content,
            metadata=metadata,
        )

    return ToolResult.success(content=content, metadata=metadata)


def _elapsed_milliseconds(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _format_command_result(
    *,
    exit_code: int | None,
    stdout: object,
    stderr: object,
) -> str:
    parts = [f"Command exited with code {exit_code}."]
    captured_output = _format_captured_output(stdout=stdout, stderr=stderr)
    if captured_output:
        parts.extend(["", captured_output])

    return "\n".join(parts)


def _format_captured_output(*, stdout: object, stderr: object) -> str:
    sections: list[str] = []
    final_newline = False
    for label, value in (("STDOUT", stdout), ("STDERR", stderr)):
        if not value:
            continue
        text = str(value)
        final_newline = text.endswith(("\n", "\r"))
        sections.append(f"{label}\n{text.rstrip(chr(10) + chr(13))}")

    formatted = "\n\n".join(sections)
    if formatted and final_newline:
        return f"{formatted}\n"
    return formatted
