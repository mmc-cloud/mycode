from collections.abc import Iterator
from dataclasses import dataclass
import ssl
from typing import Literal

import httpx2
from mcp import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT


MCPErrorCategory = Literal[
    "authentication",
    "permission_denied",
    "timeout",
    "connection_error",
    "tls_error",
    "protocol_error",
    "server_unavailable",
    "unknown",
]


@dataclass(frozen=True)
class MCPErrorClassification:
    category: MCPErrorCategory
    summary: str
    retryable: bool | None
    root_error_type: str


def classify_mcp_error(error: BaseException) -> MCPErrorClassification:
    """Classify an MCP failure without exposing provider-controlled text."""
    chain = tuple(_exception_chain(error))

    http_status = _first_http_status(chain, {401, 403})
    if http_status is not None:
        if http_status == 401:
            return _classification(
                chain,
                category="authentication",
                summary="Authentication failed (401)",
                retryable=False,
                predicate=lambda item: _http_status(item) == 401,
            )
        return _classification(
            chain,
            category="permission_denied",
            summary="Access forbidden (403)",
            retryable=False,
            predicate=lambda item: _http_status(item) == 403,
        )

    # Transport causes must be checked before MCPError. An MCP client may wrap
    # a httpx2.ConnectError in a generic "connection closed" MCPError.
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return _classification(
            chain,
            category="tls_error",
            summary="TLS connection failed",
            retryable=True,
            predicate=lambda item: isinstance(item, ssl.SSLError),
        )
    if any(
        isinstance(item, (TimeoutError, httpx2.TimeoutException))
        for item in chain
    ):
        return _classification(
            chain,
            category="timeout",
            summary="Connection timed out",
            retryable=True,
            predicate=lambda item: isinstance(
                item, (TimeoutError, httpx2.TimeoutException)
            ),
        )
    if any(
        isinstance(item, ConnectionRefusedError)
        for item in chain
    ):
        return _classification(
            chain,
            category="connection_error",
            summary="Connection refused",
            retryable=True,
            predicate=lambda item: isinstance(item, ConnectionRefusedError),
        )
    if any(
        isinstance(
            item,
            (
                httpx2.ConnectError,
                ConnectionResetError,
                BrokenPipeError,
                ConnectionError,
                EOFError,
            ),
        )
        for item in chain
    ):
        return _classification(
            chain,
            category="connection_error",
            summary="Connection failed",
            retryable=True,
            predicate=lambda item: isinstance(
                item,
                (
                    httpx2.ConnectError,
                    ConnectionResetError,
                    BrokenPipeError,
                    ConnectionError,
                    EOFError,
                ),
            ),
        )

    if any(_mcp_error_code(item) == CONNECTION_CLOSED for item in chain):
        return _classification(
            chain,
            category="connection_error",
            summary="Connection closed",
            retryable=True,
            predicate=lambda item: (
                _mcp_error_code(item) == CONNECTION_CLOSED
            ),
        )
    if any(_mcp_error_code(item) == REQUEST_TIMEOUT for item in chain):
        return _classification(
            chain,
            category="timeout",
            summary="Request timed out",
            retryable=True,
            predicate=lambda item: _mcp_error_code(item) == REQUEST_TIMEOUT,
        )
    if any(_is_server_unavailable(item) for item in chain):
        return _classification(
            chain,
            category="server_unavailable",
            summary="MCP server unavailable",
            retryable=False,
            predicate=_is_server_unavailable,
        )
    if any(isinstance(item, PermissionError) for item in chain):
        return _classification(
            chain,
            category="permission_denied",
            summary="Permission denied",
            retryable=False,
            predicate=lambda item: isinstance(item, PermissionError),
        )
    if any(isinstance(item, FileNotFoundError) for item in chain):
        return _classification(
            chain,
            category="unknown",
            summary="Server command not found",
            retryable=False,
            predicate=lambda item: isinstance(item, FileNotFoundError),
        )
    if any(isinstance(item, httpx2.HTTPStatusError) for item in chain):
        status_code = next(
            item.response.status_code
            for item in chain
            if isinstance(item, httpx2.HTTPStatusError)
        )
        if status_code >= 500:
            return _classification(
                chain,
                category="server_unavailable",
                summary=f"HTTP request failed ({status_code})",
                retryable=True,
                predicate=lambda item: (
                    isinstance(item, httpx2.HTTPStatusError)
                    and item.response.status_code == status_code
                ),
            )
        return _classification(
            chain,
            category="unknown",
            summary=f"HTTP request failed ({status_code})",
            retryable=False,
            predicate=lambda item: isinstance(item, httpx2.HTTPStatusError),
        )
    if any(isinstance(item, MCPError) for item in chain):
        return _classification(
            chain,
            category="protocol_error",
            summary="MCP protocol error",
            retryable=False,
            predicate=lambda item: isinstance(item, MCPError),
        )
    if any(isinstance(item, OSError) for item in chain):
        return _classification(
            chain,
            category="connection_error",
            summary="Server process or transport failed",
            retryable=False,
            predicate=lambda item: isinstance(item, OSError),
        )

    return MCPErrorClassification(
        category="unknown",
        summary=type(error).__name__,
        retryable=None,
        root_error_type=_root_error_type(chain, fallback=error),
    )


def safe_error_summary(error: BaseException) -> str:
    """Return a short safe summary for CLI and startup status output."""
    return classify_mcp_error(error).summary


def is_transient_mcp_error(classified: MCPErrorClassification) -> bool:
    """Return whether startup may retry this transport-level failure."""
    return classified.retryable is True and classified.category in {
        "timeout",
        "connection_error",
        "tls_error",
    }


def _classification(
    chain: tuple[BaseException, ...],
    *,
    category: MCPErrorCategory,
    summary: str,
    retryable: bool | None,
    predicate,
) -> MCPErrorClassification:
    matched = next((item for item in chain if predicate(item)), None)
    return MCPErrorClassification(
        category=category,
        summary=summary,
        retryable=retryable,
        root_error_type=(
            type(matched).__name__
            if matched is not None
            else type(chain[0]).__name__
        ),
    )


def _first_http_status(
    chain: tuple[BaseException, ...], status_codes: set[int]
) -> int | None:
    for item in chain:
        status_code = _http_status(item)
        if status_code in status_codes:
            return status_code
    return None


def _http_status(error: BaseException) -> int | None:
    if not isinstance(error, httpx2.HTTPStatusError):
        return None
    return error.response.status_code


def _mcp_error_code(error: BaseException) -> int | None:
    if not isinstance(error, MCPError):
        return None
    code = getattr(error, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    return None


def _is_server_unavailable(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    return str(error).strip().casefold() in {
        "mcp server is unavailable",
        "mcp server unavailable",
        "server unavailable",
    }


def _root_error_type(
    chain: tuple[BaseException, ...], *, fallback: BaseException
) -> str:
    for item in reversed(chain):
        if not isinstance(item, BaseExceptionGroup):
            return type(item).__name__
    return type(fallback).__name__


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Walk effective causes and ExceptionGroup members without looping."""
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
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
