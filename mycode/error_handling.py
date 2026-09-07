from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
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


@dataclass(frozen=True)
class ProviderDiagnostic:
    http_status: int | None = None
    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None


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
    diagnostic = extract_provider_diagnostic(error)
    message = classified.message
    if (
        diagnostic.http_status is not None
        and f"HTTP {diagnostic.http_status}" not in message
    ):
        message = message.rstrip("。") + f"（HTTP {diagnostic.http_status}）。"

    details: list[str] = []
    if diagnostic.retry_after is not None:
        details.append(f"retry-after={diagnostic.retry_after}")
    provider_label = diagnostic.code or diagnostic.error_type
    if diagnostic.message is not None:
        details.append(
            diagnostic.message
            if provider_label is None
            else f"{provider_label}: {diagnostic.message}"
        )
    elif provider_label is not None:
        details.append(provider_label)
    if (
        diagnostic.error_type is not None
        and diagnostic.error_type != diagnostic.code
    ):
        details.append(f"provider-type={diagnostic.error_type}")
    if diagnostic.request_id is not None:
        details.append(f"request-id={diagnostic.request_id}")
    return "\n".join([message, *details])


def extract_provider_diagnostic(error: BaseException) -> ProviderDiagnostic:
    http_status: int | None = None
    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None

    for current in _exception_chain(error):
        if http_status is None:
            http_status = _safe_http_status(current)
        body = getattr(current, "body", None)
        if isinstance(body, Mapping):
            body_code, body_type, body_message = _mapping_provider_fields(body)
            code = code or body_code
            error_type = error_type or body_type
            message = message or body_message
        code = code or _safe_provider_text(getattr(current, "code", None), 200)
        error_type = error_type or _safe_provider_text(
            getattr(current, "type", None), 200
        )
        request_id = request_id or _safe_provider_text(
            getattr(current, "request_id", None), 200
        )
        request_id = request_id or _safe_provider_text(
            getattr(current, "_request_id", None), 200
        )
        response = getattr(current, "response", None)
        request_id = request_id or _safe_provider_text(
            _header_value(response, "x-request-id"), 200
        )
        retry_after = retry_after or _safe_provider_text(
            _header_value(response, "retry-after"), 100
        )

    return ProviderDiagnostic(
        http_status=http_status,
        code=code,
        error_type=error_type,
        message=message,
        request_id=request_id,
        retry_after=retry_after,
    )


def error_summary(error: BaseException) -> str:
    message = str(error).strip()
    if message == "":
        return type(error).__name__
    first_line = message.splitlines()[0].strip()
    if len(first_line) <= 500:
        return first_line
    return first_line[:497] + "..."


_PROVIDER_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[^\s,;]+"),
    re.compile(
        r"(?i)\b(?:authorization|api[\s_-]*key|cookie)\b"
        r"\s*(?::|=|\bis\b)?\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{6,}"),
)


def _safe_provider_text(value: object, max_chars: int) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = " ".join(str(value).split())
    if text == "":
        return None
    for pattern in _PROVIDER_SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _safe_http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
    ):
        return status
    return None


def _mapping_provider_fields(
    body: Mapping[object, object],
) -> tuple[str | None, str | None, str | None]:
    pending: list[tuple[Mapping[object, object], int]] = []
    nested_error = _mapping_value(body, "error")
    if isinstance(nested_error, Mapping):
        pending.append((nested_error, 0))
    pending.append((body, 0))
    seen: set[int] = set()
    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    while pending and (code is None or error_type is None or message is None):
        current, depth = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        code = code or _safe_provider_text(_mapping_value(current, "code"), 200)
        error_type = error_type or _safe_provider_text(
            _mapping_value(current, "type"), 200
        )
        message = message or _safe_provider_text(
            _mapping_value(current, "message"), 500
        )
        if message is None:
            message = _safe_provider_text(_mapping_value(current, "detail"), 500)
        if depth < 2:
            pending.extend(
                (nested, depth + 1)
                for nested in current.values()
                if isinstance(nested, Mapping)
            )
    return code, error_type, message


def _mapping_value(mapping: Mapping[object, object], name: str) -> object | None:
    for key, value in mapping.items():
        if isinstance(key, str) and key.casefold() == name:
            return value
    return None


def _header_value(response: object, name: str) -> object | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter(name)
    if value is not None:
        return value
    return getter(name.title())


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
