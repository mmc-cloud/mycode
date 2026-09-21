from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import socket
import ssl
from time import time

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from mycode.model_errors import (
    MAX_MODEL_RETRY_DELAY_SECONDS,
    ModelErrorCode,
    ModelProviderError,
    exception_chain,
    extract_provider_diagnostic,
    safe_error_summary,
)


def normalize_openai_error(error: Exception) -> Exception:
    """Translate OpenAI SDK failures into MyCode's provider-neutral error."""
    if isinstance(error, ModelProviderError):
        return error
    if isinstance(error, (AttributeError, AssertionError)):
        return error

    chain = exception_chain(error)
    status_error = next(
        (item for item in chain if isinstance(item, APIStatusError)), None
    )
    code: ModelErrorCode | None = None
    retryable: bool | None = None
    retry_after_seconds: float | None = None
    display_message: str | None = None

    if any(isinstance(item, AuthenticationError) for item in chain):
        code, retryable = "authentication", False
    elif any(isinstance(item, PermissionDeniedError) for item in chain):
        code, retryable = "permission_denied", False
    elif status_error is not None and status_error.status_code in {401, 403}:
        code = "authentication" if status_error.status_code == 401 else "permission_denied"
        retryable = False
    else:
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
            code, retryable = "rate_limit", not quota_exhausted
            if not quota_exhausted:
                retry_after_seconds = _retry_after_seconds(rate_limit)
        elif any(isinstance(item, BadRequestError) for item in chain):
            code, retryable = "bad_request", False
        elif any(isinstance(item, NotFoundError) for item in chain):
            code, retryable = "not_found", False
        elif status_error is not None and status_error.status_code == 408:
            code, retryable = "timeout", True
            display_message = "模型服务请求超时（HTTP 408），请稍后重试。"
        elif status_error is not None and status_error.status_code in {400, 404}:
            code = "bad_request" if status_error.status_code == 400 else "not_found"
            retryable = False
        elif status_error is not None and status_error.status_code >= 500:
            code, retryable = "server_error", True
        elif any(isinstance(item, ssl.SSLError) for item in chain):
            code, retryable = "tls_error", True
        elif any(isinstance(item, socket.gaierror) for item in chain):
            code, retryable = "dns_error", True
        elif any(isinstance(item, httpx.ConnectTimeout) for item in chain):
            code, retryable = "timeout", True
            display_message = "连接模型服务超时，请检查网络、代理节点和 API 地址。"
        elif any(isinstance(item, httpx.ReadTimeout) for item in chain):
            code, retryable = "timeout", True
            display_message = "等待模型响应超时，请稍后重试并检查代理节点或服务状态。"
        elif any(isinstance(item, APITimeoutError) for item in chain) or any(
            isinstance(item, httpx.TimeoutException) for item in chain
        ):
            code, retryable = "timeout", True
        elif any(
            isinstance(
                item,
                (ConnectionResetError, BrokenPipeError, httpx.RemoteProtocolError),
            )
            for item in chain
        ):
            code, retryable = "connection_error", True
            display_message = (
                "模型服务连接在响应完成前中断，请稍后重试或切换网络节点。"
            )
        elif any(isinstance(item, APIConnectionError) for item in chain) or any(
            isinstance(item, httpx.ConnectError) for item in chain
        ):
            code, retryable = "connection_error", True

    belongs_to_provider = any(
        isinstance(item, (OpenAIError, httpx.HTTPError)) for item in chain
    )
    if code is None and belongs_to_provider:
        code, retryable = "unknown", None
    elif code is None:
        return error

    return ModelProviderError(
        safe_error_summary(error),
        code=code,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds,
        diagnostic=extract_provider_diagnostic(error),
        provider_error_type=type(error).__name__,
        display_message=display_message,
    )


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
    return any(
        marker in _QUOTA_EXHAUSTED_MARKERS
        for current in chain
        for marker in _error_markers(current)
    )


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
