from dataclasses import dataclass
from mycode.model_errors import (
    MAX_MODEL_RETRY_DELAY_SECONDS,
    ModelErrorCode,
    ModelProviderError,
    ProviderDiagnostic,
    error_summary,
    exception_chain,
    extract_provider_diagnostic,
)


@dataclass(frozen=True)
class UserFacingModelError:
    code: ModelErrorCode
    message: str
    retryable: bool | None
    retry_after_seconds: float | None = None


def classify_model_error(error: BaseException) -> UserFacingModelError:
    chain = exception_chain(error)
    provider_error = next(
        (item for item in chain if isinstance(item, ModelProviderError)), None
    )
    if provider_error is not None:
        return UserFacingModelError(
            code=provider_error.code,
            message=(
                provider_error.display_message
                or _model_error_message(
                    provider_error.code,
                    http_status=provider_error.diagnostic.http_status,
                    quota_exhausted=(
                        provider_error.code == "rate_limit"
                        and provider_error.retryable is False
                    ),
                )
            ),
            retryable=provider_error.retryable,
            retry_after_seconds=provider_error.retry_after_seconds,
        )

    return UserFacingModelError(
        code="unknown",
        message=error_summary(error),
        retryable=None,
    )


def _model_error_message(
    code: ModelErrorCode,
    *,
    http_status: int | None = None,
    quota_exhausted: bool = False,
) -> str:
    if code == "authentication":
        return "模型服务鉴权失败，请检查 API Key 和 API 地址。"
    if code == "permission_denied":
        return "模型服务拒绝访问，请检查订阅状态和模型使用权限。"
    if code == "rate_limit":
        if quota_exhausted:
            return "模型服务额度已耗尽，请检查账户额度或计费状态。"
        return "模型服务当前限流或额度不足，请稍后重试并检查额度。"
    if code == "bad_request":
        return "模型服务拒绝了当前请求，请检查模型名称和请求参数。"
    if code == "not_found":
        return "未找到模型服务端点或指定模型，请检查 API 地址和模型名称。"
    if code == "server_error":
        suffix = "" if http_status is None else f"（HTTP {http_status}）"
        return f"模型服务暂时不可用{suffix}，请稍后重试。"
    if code == "timeout":
        return "模型服务请求超时，请稍后重试并检查网络或代理节点。"
    if code == "tls_error":
        return "HTTPS/TLS 连接被提前关闭；如使用代理，请尝试切换节点或改为直连。"
    if code == "dns_error":
        return "无法解析模型服务域名，请检查 DNS、网络和 API 地址。"
    if code == "connection_error":
        return "无法连接模型服务，请检查网络、代理节点和 API 地址。"
    return "模型服务请求失败。"


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
