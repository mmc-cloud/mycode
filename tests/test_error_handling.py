from datetime import datetime, timezone
from email.utils import format_datetime
import socket
import ssl

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from mycode.error_handling import (
    MAX_MODEL_RETRY_DELAY_SECONDS,
    classify_model_error as classify_generic_model_error,
    extract_provider_diagnostic as extract_generic_provider_diagnostic,
    format_model_error as format_generic_model_error,
)
from mycode.model_errors import ModelProviderError, safe_error_summary
from mycode.providers.openai_errors import normalize_openai_error


def _provider_error(error: Exception) -> Exception:
    return normalize_openai_error(error)


def classify_model_error(error: Exception):
    return classify_generic_model_error(_provider_error(error))


def format_model_error(error: Exception, *, operation: str) -> str:
    return format_generic_model_error(_provider_error(error), operation=operation)


def extract_provider_diagnostic(error: Exception):
    return extract_generic_provider_diagnostic(_provider_error(error))


def request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/v1/chat/completions")


def response(
    status_code: int, *, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=request(), headers=headers)


@pytest.mark.parametrize(
    ("error", "code", "message_fragment", "retryable"),
    [
        (
            APITimeoutError(request=request()),
            "timeout",
            "请求超时",
            True,
        ),
        (
            AuthenticationError(
                "invalid key",
                response=response(401),
                body=None,
            ),
            "authentication",
            "鉴权失败",
            False,
        ),
        (
            PermissionDeniedError(
                "forbidden",
                response=response(403),
                body=None,
            ),
            "permission_denied",
            "订阅状态和模型使用权限",
            False,
        ),
        (
            RateLimitError(
                "too many requests",
                response=response(429),
                body=None,
            ),
            "rate_limit",
            "限流或额度不足",
            True,
        ),
        (
            BadRequestError(
                "invalid request",
                response=response(400),
                body=None,
            ),
            "bad_request",
            "模型名称和请求参数",
            False,
        ),
        (
            NotFoundError(
                "missing model",
                response=response(404),
                body=None,
            ),
            "not_found",
            "API 地址和模型名称",
            False,
        ),
        (
            InternalServerError(
                "provider unavailable",
                response=response(503),
                body=None,
            ),
            "server_error",
            "HTTP 503",
            True,
        ),
    ],
)
def test_classify_stable_openai_errors(
    error: BaseException,
    code: str,
    message_fragment: str,
    retryable: bool,
) -> None:
    classified = classify_model_error(error)

    assert classified.code == code
    assert message_fragment in classified.message
    assert classified.retryable is retryable


@pytest.mark.parametrize(
    ("cause", "code", "message_fragment"),
    [
        (
            ssl.SSLEOFError(8, "unexpected eof"),
            "tls_error",
            "HTTPS/TLS 连接被提前关闭",
        ),
        (
            httpx.ConnectTimeout("connect timed out"),
            "timeout",
            "连接模型服务超时",
        ),
        (
            httpx.ReadTimeout("read timed out"),
            "timeout",
            "等待模型响应超时",
        ),
        (
            socket.gaierror("name resolution failed"),
            "dns_error",
            "无法解析模型服务域名",
        ),
        (
            ConnectionResetError("connection reset"),
            "connection_error",
            "响应完成前中断",
        ),
    ],
)
def test_classify_api_connection_error_from_cause_chain(
    cause: BaseException,
    code: str,
    message_fragment: str,
) -> None:
    error = APIConnectionError(request=request())
    error.__cause__ = cause

    classified = classify_model_error(error)

    assert classified.code == code
    assert message_fragment in classified.message
    assert classified.retryable is True


def test_unknown_error_uses_bounded_single_line_operation_fallback() -> None:
    error = RuntimeError("unexpected failure\nprivate detail")

    message = format_model_error(error, operation="模型流式请求失败")

    assert message == "模型流式请求失败：unexpected failure"
    assert "private detail" not in message


def test_http_408_is_retryable_timeout() -> None:
    error = APIStatusError(
        "request timeout",
        response=response(408),
        body=None,
    )

    classified = classify_model_error(error)

    assert classified.code == "timeout"
    assert classified.retryable is True
    assert "HTTP 408" in classified.message


def test_unrecognized_openai_status_error_is_normalized() -> None:
    error = APIStatusError(
        "unexpected provider status",
        response=response(418),
        body={"error": {"type": "unexpected_status"}},
    )

    normalized = normalize_openai_error(error)

    assert isinstance(normalized, ModelProviderError)
    assert normalized.code == "unknown"
    assert normalized.retryable is None
    assert normalized.provider_error_type == "APIStatusError"


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer bearer-secret-123",
        "api_key=api-secret-123",
        "sk-secret-token-123",
        "Cookie: session=secret-cookie-123",
    ],
)
def test_provider_error_summary_redacts_secrets(secret: str) -> None:
    error = APIStatusError(
        f"provider rejected {secret}",
        response=response(418),
        body=None,
    )

    normalized = normalize_openai_error(error)

    assert isinstance(normalized, ModelProviderError)
    assert "[redacted]" in str(normalized)
    assert "secret" not in str(normalized)


@pytest.mark.parametrize("error", [AttributeError("bug"), AssertionError("bug")])
def test_programming_errors_are_not_normalized(error: Exception) -> None:
    assert normalize_openai_error(error) is error


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer cutoff-bearer-token",
        "api_key=cutoff-api-secret",
        "sk-cutoffsecretvalue",
        "Cookie: session=cutoff-cookie",
    ],
)
def test_safe_error_summary_redacts_a_secret_that_crosses_the_truncation_cutoff(
    secret: str,
) -> None:
    """A long prefix must not push redaction past the 500-character cutoff."""
    error = Exception("x" * 480 + " " + secret + " tail" * 40)

    summary = safe_error_summary(error)

    assert "[redacted]" in summary
    assert len(summary) <= 500
    for fragment in secret.split("=")[-1].split(":")[-1].replace("Bearer ", "").split():
        assert fragment not in summary


def test_bad_request_format_includes_safe_provider_diagnostic() -> None:
    error = BadRequestError(
        "bad request",
        response=response(400, headers={"x-request-id": "req-400"}),
        body={
            "error": {
                "code": "MissingSessionID",
                "message": "Request is missing x-opencode-session",
            }
        },
    )

    message = format_model_error(error, operation="模型请求失败")

    assert "HTTP 400" in message
    assert "MissingSessionID" in message
    assert "x-opencode-session" in message
    assert "request-id=req-400" in message


def test_rate_limit_format_includes_provider_message_and_retry_after() -> None:
    error = RateLimitError(
        "rate limited",
        response=response(429, headers={"Retry-After": "7"}),
        body={"error": {"message": "Provider request rate exceeded"}},
    )

    message = format_model_error(error, operation="模型请求失败")

    assert "HTTP 429" in message
    assert "retry-after=7" in message
    assert "Provider request rate exceeded" in message
    assert classify_model_error(error).retry_after_seconds == 7.0


def test_server_error_format_includes_provider_code_and_message() -> None:
    error = InternalServerError(
        "provider unavailable",
        response=response(503),
        body={
            "failure": {
                "code": "UpstreamUnavailable",
                "type": "provider_error",
                "message": "Upstream model is temporarily unavailable",
            }
        },
    )

    message = format_model_error(error, operation="模型请求失败")

    assert "HTTP 503" in message
    assert "UpstreamUnavailable" in message
    assert "provider_error" in message
    assert "Upstream model is temporarily unavailable" in message


def test_provider_diagnostic_redacts_secrets_and_bounds_plain_text() -> None:
    error = BadRequestError(
        "unsafe raw error",
        response=response(
            400,
            headers={
                "x-request-id": "req-sensitive",
                "Authorization": "Bearer response-header-secret",
            },
        ),
        body={
            "error": {
                "code": "InvalidRequest",
                "message": (
                    "Authorization: Bearer secret-token; "
                    "API key=sk-super-secret; Cookie=session-secret; "
                    + "x" * 600
                ),
            },
            "Authorization": "Bearer body-secret",
        },
    )

    diagnostic = extract_provider_diagnostic(error)
    message = format_model_error(error, operation="模型请求失败")
    rendered = repr((diagnostic, message))

    assert diagnostic.request_id == "req-sensitive"
    assert diagnostic.message is not None
    assert len(diagnostic.message) <= 500
    for secret in (
        "secret-token",
        "sk-super-secret",
        "session-secret",
        "response-header-secret",
        "body-secret",
    ):
        assert secret not in rendered


def test_rate_limit_quota_exhaustion_is_not_retryable() -> None:
    error = RateLimitError(
        "quota exhausted",
        response=response(429),
        body={"type": "insufficient_quota", "code": "insufficient_quota"},
    )

    classified = classify_model_error(error)

    assert classified.code == "rate_limit"
    assert classified.retryable is False
    assert classified.retry_after_seconds is None
    assert "额度已耗尽" in classified.message


def test_rate_limit_uses_retry_after_header() -> None:
    error = RateLimitError(
        "too many requests",
        response=response(429, headers={"Retry-After": "7"}),
        body=None,
    )

    classified = classify_model_error(error)

    assert classified.retryable is True
    assert classified.retry_after_seconds == 7.0


def test_rate_limit_retry_after_seconds_is_capped() -> None:
    error = RateLimitError(
        "too many requests",
        response=response(429, headers={"Retry-After": "60"}),
        body=None,
    )

    classified = classify_model_error(error)

    assert classified.retry_after_seconds == MAX_MODEL_RETRY_DELAY_SECONDS


def test_rate_limit_retry_after_http_date_is_capped(monkeypatch) -> None:
    monkeypatch.setattr("mycode.providers.openai_errors.time", lambda: 1000.0)
    retry_at = datetime.fromtimestamp(1040.0, tz=timezone.utc)
    error = RateLimitError(
        "too many requests",
        response=response(
            429,
            headers={"Retry-After": format_datetime(retry_at, usegmt=True)},
        ),
        body=None,
    )

    classified = classify_model_error(error)

    assert classified.retry_after_seconds == MAX_MODEL_RETRY_DELAY_SECONDS


def test_explicit_model_cause_wins_over_context() -> None:
    error = APIConnectionError(request=request())
    error.__context__ = PermissionError("private context")
    error.__cause__ = httpx.ConnectTimeout("private cause")
    error.__suppress_context__ = False

    classified = classify_model_error(error)

    assert classified.code == "timeout"


def test_unsuppressed_model_context_is_followed() -> None:
    error = RuntimeError("wrapper")
    error.__context__ = httpx.ConnectTimeout("private context")
    error.__suppress_context__ = False

    classified = classify_model_error(error)

    assert classified.code == "timeout"


def test_suppressed_model_context_is_not_classified() -> None:
    error = RuntimeError("safe wrapper")
    error.__context__ = httpx.ConnectTimeout("private timeout")
    error.__suppress_context__ = True

    classified = classify_model_error(error)

    assert classified.code == "unknown"
    assert classified.message == "safe wrapper"
