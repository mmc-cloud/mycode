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

from mycode.error_handling import classify_model_error, format_model_error


def request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/v1/chat/completions")


def response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=request())


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
