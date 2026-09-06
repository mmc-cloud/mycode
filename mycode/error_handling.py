from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import socket
import ssl
from time import time
from typing import Literal

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)


MAX_MODEL_RETRY_DELAY_SECONDS = 30.0


ModelErrorCode = Literal[
    "timeout",
    "authentication",
    "permission_denied",
    "rate_limit",
    "bad_request",
    "not_found",
    "server_error",
    "tls_error",
    "dns_error",
    "connection_error",
    "unknown",
]


@dataclass(frozen=True)
class UserFacingModelError:
    code: ModelErrorCode
    message: str
    retryable: bool | None
    retry_after_seconds: float | None = None


def classify_model_error(error: BaseException) -> UserFacingModelError:
    chain = _exception_chain(error)
    status_error = next(
        (item for item in chain if isinstance(item, APIStatusError)), None
    )

    if any(isinstance(item, AuthenticationError) for item in chain):
        return UserFacingModelError(
            code="authentication",
            message="模型服务鉴权失败，请检查 API Key 和 API 地址。",
            retryable=False,
        )
    if any(isinstance(item, PermissionDeniedError) for item in chain):
        return UserFacingModelError(
            code="permission_denied",
            message="模型服务拒绝访问，请检查订阅状态和模型使用权限。",
            retryable=False,
        )
    if status_error is not None and status_error.status_code in {401, 403}:
        if status_error.status_code == 401:
            return UserFacingModelError(
                code="authentication",
                message="模型服务鉴权失败，请检查 API Key 和 API 地址。",
                retryable=False,
            )
        return UserFacingModelError(
            code="permission_denied",
            message="模型服务拒绝访问，请检查订阅状态和模型使用权限。",
            retryable=False,
        )
    rate_limit = next(
        (
            item
            for item in chain
            if isinstance(item, RateLimitError)
            or (
                isinstance(item, APIStatusError)
                and item.status_code == 429
            )
        ),
        None,
    )
    if rate_limit is not None:
        quota_exhausted = _quota_exhausted(chain)
        return UserFacingModelError(
            code="rate_limit",
            message=(
                "模型服务额度已耗尽，请检查账户额度或计费状态。"
                if quota_exhausted
                else "模型服务当前限流或额度不足，请稍后重试并检查额度。"
            ),
            retryable=not quota_exhausted,
            retry_after_seconds=(
                None if quota_exhausted else _retry_after_seconds(rate_limit)
            ),
        )
    if any(isinstance(item, BadRequestError) for item in chain):
        return UserFacingModelError(
            code="bad_request",
            message="模型服务拒绝了当前请求，请检查模型名称和请求参数。",
            retryable=False,
        )
    if any(isinstance(item, NotFoundError) for item in chain):
        return UserFacingModelError(
            code="not_found",
            message="未找到模型服务端点或指定模型，请检查 API 地址和模型名称。",
            retryable=False,
        )
    if status_error is not None and status_error.status_code == 408:
        return UserFacingModelError(
            code="timeout",
            message="模型服务请求超时（HTTP 408），请稍后重试。",
            retryable=True,
        )
    if status_error is not None and status_error.status_code in {400, 404}:
        if status_error.status_code == 400:
            return UserFacingModelError(
                code="bad_request",
                message="模型服务拒绝了当前请求，请检查模型名称和请求参数。",
                retryable=False,
            )
        return UserFacingModelError(
            code="not_found",
            message="未找到模型服务端点或指定模型，请检查 API 地址和模型名称。",
            retryable=False,
        )
    if status_error is not None and status_error.status_code >= 500:
        return UserFacingModelError(
            code="server_error",
            message=(
                f"模型服务暂时不可用（HTTP {status_error.status_code}），请稍后重试。"
            ),
            retryable=True,
        )

    if any(isinstance(item, ssl.SSLError) for item in chain):
        return UserFacingModelError(
            code="tls_error",
            message=(
                "HTTPS/TLS 连接被提前关闭；如使用代理，请尝试切换节点或改为直连。"
            ),
            retryable=True,
        )
    if any(isinstance(item, socket.gaierror) for item in chain):
        return UserFacingModelError(
            code="dns_error",
            message="无法解析模型服务域名，请检查 DNS、网络和 API 地址。",
            retryable=True,
        )
    if any(isinstance(item, httpx.ConnectTimeout) for item in chain):
        return UserFacingModelError(
            code="timeout",
            message="连接模型服务超时，请检查网络、代理节点和 API 地址。",
            retryable=True,
        )
    if any(isinstance(item, httpx.ReadTimeout) for item in chain):
        return UserFacingModelError(
            code="timeout",
            message="等待模型响应超时，请稍后重试并检查代理节点或服务状态。",
            retryable=True,
        )
    if isinstance(error, APITimeoutError) or any(
        isinstance(item, httpx.TimeoutException) for item in chain
    ):
        return UserFacingModelError(
            code="timeout",
            message="模型服务请求超时，请稍后重试并检查网络或代理节点。",
            retryable=True,
        )
    if any(
        isinstance(
            item,
            (
                ConnectionResetError,
                BrokenPipeError,
                httpx.RemoteProtocolError,
            ),
        )
        for item in chain
    ):
        return UserFacingModelError(
            code="connection_error",
            message="模型服务连接在响应完成前中断，请稍后重试或切换网络节点。",
            retryable=True,
        )
    if isinstance(error, APIConnectionError) or any(
        isinstance(item, httpx.ConnectError) for item in chain
    ):
        return UserFacingModelError(
            code="connection_error",
            message="无法连接模型服务，请检查网络、代理节点和 API 地址。",
            retryable=True,
        )

    return UserFacingModelError(
        code="unknown",
        message=error_summary(error),
        retryable=None,
    )


def format_model_error(error: BaseException, *, operation: str) -> str:
    classified = classify_model_error(error)
    if classified.code == "unknown":
        return f"{operation}：{classified.message}"
    return classified.message


def error_summary(error: BaseException) -> str:
    message = str(error).strip()
    if message == "":
        return type(error).__name__
    first_line = message.splitlines()[0].strip()
    if len(first_line) <= 500:
        return first_line
    return first_line[:497] + "..."


_QUOTA_EXHAUSTED_MARKERS = {
    "insufficient_quota",
    "quota_exceeded",
    "usage_limit",
    "billing_hard_limit_reached",
    "billing_limit",
    "billing_error",
    "credits_exhausted",
    "billing",
}


def _quota_exhausted(chain: tuple[BaseException, ...]) -> bool:
    for error in chain:
        for marker in _error_markers(error):
            if marker in _QUOTA_EXHAUSTED_MARKERS:
                return True
    return False


def _error_markers(error: BaseException):
    for attribute in ("code", "type"):
        value = getattr(error, attribute, None)
        if isinstance(value, str):
            yield value.strip().casefold()
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        yield from _mapping_error_markers(body)


def _mapping_error_markers(value: Mapping[object, object]):
    for key, nested in value.items():
        if key in {"code", "type"} and isinstance(nested, str):
            yield nested.strip().casefold()
        if isinstance(nested, Mapping):
            yield from _mapping_error_markers(nested)


def _retry_after_seconds(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(str(value).strip())
    except (TypeError, ValueError):
        delay = None
    if delay is not None and math.isfinite(delay) and delay >= 0:
        return min(delay, MAX_MODEL_RETRY_DELAY_SECONDS)
    try:
        retry_at = parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = retry_at.timestamp() - time()
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(delay):
        return None
    return min(max(0.0, delay), MAX_MODEL_RETRY_DELAY_SECONDS)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
    return tuple(chain)
