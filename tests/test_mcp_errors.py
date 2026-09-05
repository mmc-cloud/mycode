import httpx2
import pytest
from mcp import MCPError

from mycode.mcp.errors import safe_error_summary


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("secret timeout details"), "Connection timed out"),
        (FileNotFoundError("C:/secret/server.exe"), "Server command not found"),
        (ConnectionRefusedError("secret endpoint"), "Connection refused"),
        (ConnectionError("secret endpoint"), "Connection failed"),
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
