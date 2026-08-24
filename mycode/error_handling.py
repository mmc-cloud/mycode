from dataclasses import dataclass
import socket
import ssl
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


def classify_model_error(error: BaseException) -> UserFacingModelError:
    chain = _exception_chain(error)

    if isinstance(error, AuthenticationError):
        return UserFacingModelError(
            code="authentication",
            message="模型服务鉴权失败，请检查 API Key 和 API 地址。",
            retryable=False,
        )
    if isinstance(error, PermissionDeniedError):
        return UserFacingModelError(
            code="permission_denied",
            message="模型服务拒绝访问，请检查订阅状态和模型使用权限。",
            retryable=False,
        )
    if isinstance(error, RateLimitError):
        return UserFacingModelError(
            code="rate_limit",
            message="模型服务当前限流或额度不足，请稍后重试并检查额度。",
            retryable=True,
        )
    if isinstance(error, BadRequestError):
        return UserFacingModelError(
            code="bad_request",
            message="模型服务拒绝了当前请求，请检查模型名称和请求参数。",
            retryable=False,
        )
    if isinstance(error, NotFoundError):
        return UserFacingModelError(
            code="not_found",
            message="未找到模型服务端点或指定模型，请检查 API 地址和模型名称。",
            retryable=False,
        )
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return UserFacingModelError(
            code="server_error",
            message=(
                f"模型服务暂时不可用（HTTP {error.status_code}），请稍后重试。"
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


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = (
            current.__cause__
            if current.__cause__ is not None
            else current.__context__
        )
    return tuple(chain)
