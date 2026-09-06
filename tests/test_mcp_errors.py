import httpx2
import pytest
from mcp import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from mycode.mcp.errors import (
    classify_mcp_error,
    safe_error_summary,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("secret timeout details"), "Connection timed out"),
        (FileNotFoundError("C:/secret/server.exe"), "Server command not found"),
        (ConnectionRefusedError("secret endpoint"), "Connection refused"),
        (ConnectionError("secret endpoint"), "Connection failed"),
        (
            MCPError(CONNECTION_CLOSED, "Bearer secret-token"),
            "Connection closed",
        ),
        (MCPError(REQUEST_TIMEOUT, "secret timeout"), "Request timed out"),
        (MCPError(-32603, "Bearer secret-token"), "MCP protocol error"),
        (
            RuntimeError("Authorization: Bearer secret-token?api_key=value"),
            "RuntimeError",
        ),
    ],
)
def test_safe_error_summary_uses_fixed_or_type_only_messages(
    error: BaseException, expected: str
) -> None:
    summary = safe_error_summary(error)

    assert summary == expected
    assert "secret" not in summary.lower()
    assert "bearer" not in summary.lower()
    assert "api_key" not in summary.lower()


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "Authentication failed (401)"),
        (403, "Access forbidden (403)"),
        (500, "HTTP request failed (500)"),
    ],
)
def test_safe_error_summary_uses_only_http_status(
    status_code: int, expected: str
) -> None:
    request = httpx2.Request(
        "POST",
        "https://example.test/mcp?token=secret",
        headers={"Authorization": "Bearer secret-token"},
    )
    response = httpx2.Response(
        status_code,
        request=request,
        text="secret response body",
    )
    error = httpx2.HTTPStatusError(
        "Bearer secret-token",
        request=request,
        response=response,
    )

    assert safe_error_summary(error) == expected


def test_safe_error_summary_finds_http_status_inside_exception_group() -> None:
    request = httpx2.Request("POST", "https://example.test/mcp?token=secret")
    response = httpx2.Response(403, request=request, text="secret body")
    nested = httpx2.HTTPStatusError(
        "Bearer secret-token",
        request=request,
        response=response,
    )

    summary = safe_error_summary(ExceptionGroup("secret group", [nested]))

    assert summary == "Access forbidden (403)"


def test_transport_root_cause_wins_over_wrapping_mcp_error() -> None:
    cause = httpx2.ConnectError("https://example.test?token=secret")
    error = MCPError(-32603, "Connection closed: Bearer secret-token")
    error.__cause__ = cause

    classified = classify_mcp_error(error)

    assert classified.category == "connection_error"
    assert classified.root_error_type == "ConnectError"
    assert classified.retryable is True
    assert classified.summary == "Connection failed"
    assert "secret" not in classified.summary.lower()


@pytest.mark.parametrize(
    ("error", "category", "retryable"),
    [
        (
            httpx2.HTTPStatusError(
                "secret",
                request=httpx2.Request("GET", "https://example.test"),
                response=httpx2.Response(
                    401,
                    request=httpx2.Request("GET", "https://example.test"),
                ),
            ),
            "authentication",
            False,
        ),
        (
            httpx2.HTTPStatusError(
                "secret",
                request=httpx2.Request("GET", "https://example.test"),
                response=httpx2.Response(
                    403,
                    request=httpx2.Request("GET", "https://example.test"),
                ),
            ),
            "permission_denied",
            False,
        ),
        (TimeoutError("secret"), "timeout", True),
        (MCPError(CONNECTION_CLOSED, "secret"), "connection_error", True),
        (MCPError(REQUEST_TIMEOUT, "secret"), "timeout", True),
        (MCPError(-32603, "secret"), "protocol_error", False),
        (RuntimeError("MCP server is unavailable"), "server_unavailable", False),
        (
            PermissionError("C:/private/server.sock"),
            "permission_denied",
            False,
        ),
        (OSError("private transport detail"), "connection_error", False),
    ],
)
def test_classifies_mcp_failure_categories(error, category, retryable) -> None:
    classified = classify_mcp_error(error)

    assert classified.category == category
    assert classified.retryable is retryable
    assert "secret" not in classified.summary.lower()


def test_classification_walks_exception_group_members() -> None:
    error = ExceptionGroup(
        "secret group",
        [RuntimeError("safe wrapper"), httpx2.ConnectError("secret transport")],
    )

    classified = classify_mcp_error(error)

    assert classified.category == "connection_error"
    assert classified.root_error_type == "ConnectError"


def test_explicit_cause_wins_over_unsuppressed_context() -> None:
    context = PermissionError("private context")
    cause = httpx2.ConnectError("private cause")
    error = RuntimeError("wrapper")
    error.__context__ = context
    error.__cause__ = cause
    error.__suppress_context__ = False

    classified = classify_mcp_error(error)

    assert classified.category == "connection_error"
    assert classified.root_error_type == "ConnectError"


def test_unsuppressed_mcp_context_is_followed() -> None:
    error = RuntimeError("wrapper")
    error.__context__ = httpx2.ConnectError("private context")
    error.__suppress_context__ = False

    classified = classify_mcp_error(error)

    assert classified.category == "connection_error"
    assert classified.root_error_type == "ConnectError"


def test_suppressed_context_is_not_classified() -> None:
    error = RuntimeError("safe wrapper")
    error.__context__ = PermissionError("private permission detail")
    error.__suppress_context__ = True

    classified = classify_mcp_error(error)

    assert classified.category == "unknown"
    assert classified.retryable is None
    assert classified.root_error_type == "RuntimeError"
