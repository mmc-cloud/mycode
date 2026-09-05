from collections.abc import Iterator

import httpx2
from mcp import MCPError


def safe_error_summary(error: BaseException) -> str:
    """Return a short startup error without exposing provider-controlled text."""
    chain = tuple(_exception_chain(error))

    if any(
        isinstance(item, (TimeoutError, httpx2.TimeoutException))
        for item in chain
    ):
        return "Connection timed out"
    if any(isinstance(item, FileNotFoundError) for item in chain):
        return "Server command not found"
    if any(isinstance(item, ConnectionRefusedError) for item in chain):
        return "Connection refused"

    for item in chain:
        if isinstance(item, httpx2.HTTPStatusError):
            status_code = item.response.status_code
            if status_code == 401:
                return "Authentication failed (401)"
            if status_code == 403:
                return "Access forbidden (403)"
            return f"HTTP request failed ({status_code})"

    if any(isinstance(item, MCPError) for item in chain):
        return "MCP protocol error"
    if any(isinstance(item, (ConnectionResetError, BrokenPipeError)) for item in chain):
        return "Connection interrupted"
    if any(isinstance(item, httpx2.ConnectError) for item in chain):
        return "Connection failed"
    if any(isinstance(item, ConnectionError) for item in chain):
        return "Connection failed"
    if any(isinstance(item, EOFError) for item in chain):
        return "Connection closed unexpectedly"
    if any(isinstance(item, OSError) for item in chain):
        return "Server process or transport failed"

    # Unknown messages may contain headers, query parameters, bodies, tool
    # arguments, or source content. The exception type is the safe fallback.
    return type(error).__name__


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        chained = (
            current.__cause__
            if current.__cause__ is not None
            else current.__context__
        )
        if chained is not None:
            pending.append(chained)
