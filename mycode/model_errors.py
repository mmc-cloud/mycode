from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Literal


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

MAX_MODEL_RETRY_DELAY_SECONDS = 30.0


@dataclass(frozen=True)
class ProviderDiagnostic:
    http_status: int | None = None
    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None


class ModelProviderError(Exception):
    """Provider-neutral failure raised across the LLM/runtime boundary."""

    def __init__(
        self,
        summary: str,
        *,
        code: ModelErrorCode,
        retryable: bool | None,
        retry_after_seconds: float | None = None,
        diagnostic: ProviderDiagnostic | None = None,
        provider_error_type: str | None = None,
        display_message: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.diagnostic = diagnostic or ProviderDiagnostic()
        self.provider_error_type = provider_error_type
        self.display_message = display_message


def model_error_type(error: BaseException) -> str:
    if isinstance(error, ModelProviderError) and error.provider_error_type:
        return error.provider_error_type
    return type(error).__name__


def exception_chain(error: BaseException) -> tuple[BaseException, ...]:
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


def extract_provider_diagnostic(error: BaseException) -> ProviderDiagnostic:
    provider_error = next(
        (
            current
            for current in exception_chain(error)
            if isinstance(current, ModelProviderError)
        ),
        None,
    )
    if provider_error is not None:
        return provider_error.diagnostic

    http_status: int | None = None
    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None
    for current in exception_chain(error):
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
    return first_line if len(first_line) <= 500 else first_line[:497] + "..."


def safe_error_summary(error: BaseException) -> str:
    """Redact the raw message before truncating it.

    Redacting after truncation could leave a fragment of a secret that no longer
    matches any pattern, so the order is single-line, redact, then truncate.
    """
    safe_summary = _safe_provider_text(str(error), 500)
    return type(error).__name__ if safe_summary is None else safe_summary


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
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _safe_http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
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
    code = error_type = message = None
    while pending and (code is None or error_type is None or message is None):
        current, depth = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        code = code or _safe_provider_text(_mapping_value(current, "code"), 200)
        error_type = error_type or _safe_provider_text(_mapping_value(current, "type"), 200)
        message = message or _safe_provider_text(_mapping_value(current, "message"), 500)
        message = message or _safe_provider_text(_mapping_value(current, "detail"), 500)
        if depth < 2:
            pending.extend(
                (nested, depth + 1)
                for nested in current.values()
                if isinstance(nested, Mapping)
            )
    return code, error_type, message


def _mapping_value(mapping: Mapping[object, object], name: str) -> object | None:
    return next(
        (
            value
            for key, value in mapping.items()
            if isinstance(key, str) and key.casefold() == name
        ),
        None,
    )


def _header_value(response: object, name: str) -> object | None:
    getter = getattr(getattr(response, "headers", None), "get", None)
    if not callable(getter):
        return None
    return getter(name) or getter(name.title())
