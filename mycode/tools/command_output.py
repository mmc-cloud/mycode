import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass


OUTPUT_THREAD_JOIN_TIMEOUT_SECONDS = 1.0
OUTPUT_THREAD_CLOSE_JOIN_TIMEOUT_SECONDS = 0.2


@dataclass(frozen=True)
class CapturedOutput:
    content: str
    chars: int
    truncated: bool


class BoundedOutputCapture:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self._chars = 0
        self._tail_chars = 0
        self._chunks: deque[str] = deque()
        self._lock = threading.Lock()

    def append(self, chunk: str) -> None:
        if chunk == "":
            return

        with self._lock:
            self._chars += len(chunk)
            if self.max_chars == 0:
                return

            if len(chunk) >= self.max_chars:
                self._chunks.clear()
                tail_chunk = chunk[-self.max_chars :]
                self._chunks.append(tail_chunk)
                self._tail_chars = len(tail_chunk)
                return

            self._chunks.append(chunk)
            self._tail_chars += len(chunk)
            self._trim_to_max_chars()

    def snapshot(self) -> CapturedOutput:
        with self._lock:
            return CapturedOutput(
                content="".join(self._chunks),
                chars=self._chars,
                truncated=self._chars > self.max_chars,
            )

    def _trim_to_max_chars(self) -> None:
        overflow = self._tail_chars - self.max_chars
        while overflow > 0 and self._chunks:
            left = self._chunks[0]
            if len(left) <= overflow:
                self._chunks.popleft()
                self._tail_chars -= len(left)
                overflow -= len(left)
                continue

            self._chunks[0] = left[overflow:]
            self._tail_chars -= overflow
            break


def start_output_threads(
    *,
    process: subprocess.Popen[str],
    stdout_capture: BoundedOutputCapture,
    stderr_capture: BoundedOutputCapture,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if process.stdout is not None:
        threads.append(
            threading.Thread(
                target=_consume_output_stream,
                args=(process.stdout, stdout_capture),
                daemon=True,
            )
        )
    if process.stderr is not None:
        threads.append(
            threading.Thread(
                target=_consume_output_stream,
                args=(process.stderr, stderr_capture),
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    return threads


def finish_output_threads(
    process: subprocess.Popen[str],
    threads: list[threading.Thread],
) -> bool:
    if _join_output_threads(threads, OUTPUT_THREAD_JOIN_TIMEOUT_SECONDS):
        return True

    _close_process_output_streams(process)
    _join_output_threads(threads, OUTPUT_THREAD_CLOSE_JOIN_TIMEOUT_SECONDS)
    return False


def build_output_metadata(
    *,
    stdout: CapturedOutput,
    stderr: CapturedOutput,
) -> dict[str, object]:
    return {
        "stdout": stdout.content,
        "stderr": stderr.content,
        "stdout_chars": stdout.chars,
        "stderr_chars": stderr.chars,
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
        "output_truncation_strategy": "tail",
    }


def _consume_output_stream(
    stream: object,
    capture: BoundedOutputCapture,
) -> None:
    try:
        while True:
            try:
                chunk = stream.read(8192)  # type: ignore[attr-defined]
            except (OSError, ValueError):
                break
            if not chunk:
                break
            capture.append(str(chunk))
    finally:
        _close_stream(stream)


def _join_output_threads(
    threads: list[threading.Thread],
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)

    return not any(thread.is_alive() for thread in threads)


def _close_process_output_streams(process: subprocess.Popen[str]) -> None:
    if process.stdout is not None:
        _close_stream(process.stdout)
    if process.stderr is not None:
        _close_stream(process.stderr)


def _close_stream(stream: object) -> None:
    try:
        stream.close()  # type: ignore[attr-defined]
    except (OSError, ValueError):
        pass
