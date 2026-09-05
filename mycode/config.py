import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

MYCODE_CONFIG_DIR_NAME = ".mycode"
MYCODE_ENV_FILE_NAME = ".env"
DEFAULT_LLM_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_LLM_RESERVED_OUTPUT_TOKENS = 8192
DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS = 4096
DEFAULT_LLM_MEMORY_CONTEXT_TOKENS = 2048
DEFAULT_LLM_STREAM_INCLUDE_USAGE = True


ReasoningEffort = Literal["high", "max"]
SUPPORTED_REASONING_EFFORTS: frozenset[str] = frozenset({"high", "max"})


class MissingLLMConfigError(ValueError):
    pass


class _RedactedConfigValues(Mapping[str, str | None]):
    def __init__(self, values: Mapping[str, str | None]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> str | None:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return "<redacted config values>"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    compact_model: str | None = None
    subagent_model: str | None = None
    context_window_tokens: int = DEFAULT_LLM_CONTEXT_WINDOW_TOKENS
    reserved_output_tokens: int = DEFAULT_LLM_RESERVED_OUTPUT_TOKENS
    context_safety_margin_tokens: int = DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS
    memory_context_tokens: int = DEFAULT_LLM_MEMORY_CONTEXT_TOKENS
    stream_include_usage: bool = DEFAULT_LLM_STREAM_INCLUDE_USAGE
    thinking_enabled: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        main_model = self.model.strip()
        if main_model == "":
            raise ValueError("MYCODE_MODEL must not be empty.")

        compact_model = self.compact_model
        subagent_model = self.subagent_model
        object.__setattr__(self, "model", main_model)
        object.__setattr__(
            self,
            "compact_model",
            (
                main_model
                if compact_model is None or compact_model.strip() == ""
                else compact_model.strip()
            ),
        )
        object.__setattr__(
            self,
            "subagent_model",
            main_model
            if subagent_model is None or subagent_model.strip() == ""
            else subagent_model.strip(),
        )
        if self.reasoning_effort is not None:
            normalized_effort = self.reasoning_effort.strip().lower()
            if normalized_effort not in SUPPORTED_REASONING_EFFORTS:
                raise ValueError(
                    "LLM_REASONING_EFFORT must be one of: high, max."
                )
            object.__setattr__(self, "reasoning_effort", normalized_effort)
        if self.thinking_enabled is True and self.reasoning_effort is None:
            object.__setattr__(self, "reasoning_effort", "high")
        if self.thinking_enabled is not True and self.reasoning_effort is not None:
            raise ValueError(
                "LLM_REASONING_EFFORT requires LLM_THINKING_ENABLED=true."
            )
        if self.max_output_tokens is not None:
            if self.max_output_tokens < 1:
                raise ValueError("LLM_MAX_OUTPUT_TOKENS must be at least 1.")
            if self.max_output_tokens > self.reserved_output_tokens:
                raise ValueError(
                    "LLM_MAX_OUTPUT_TOKENS must not exceed "
                    "LLM_RESERVED_OUTPUT_TOKENS."
                )


def default_user_env_file() -> Path:
    return Path.home() / MYCODE_CONFIG_DIR_NAME / MYCODE_ENV_FILE_NAME


def project_env_file(workspace_root: str | Path) -> Path:
    return Path(workspace_root) / MYCODE_CONFIG_DIR_NAME / MYCODE_ENV_FILE_NAME


def load_layered_environment(
    env_file: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    user_values = dotenv_values(
        default_user_env_file() if env_file is None else env_file
    )
    project_values = (
        {}
        if workspace_root is None
        else dotenv_values(project_env_file(workspace_root))
    )
    process_values = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    for values in (user_values, project_values, process_values):
        for name, value in values.items():
            if value is not None and value.strip() != "":
                merged[name] = value
    return merged


def load_llm_config(
    env_file: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> LLMConfig:
    values = _RedactedConfigValues(
        load_layered_environment(
            env_file,
            workspace_root=workspace_root,
        )
    )

    api_key = _required_env_value(
        "MYCODE_API_KEY",
        values,
    )
    base_url = _required_env_value(
        "MYCODE_BASE_URL",
        values,
    )
    model = _required_env_value(
        "MYCODE_MODEL",
        values,
    )
    compact_model = _env_value(
        "MYCODE_COMPACT_MODEL",
        model,
        values,
    )
    subagent_model = _env_value(
        "MYCODE_SUBAGENT_MODEL",
        model,
        values,
    )
    context_window_tokens = _int_env_value(
        "LLM_CONTEXT_WINDOW_TOKENS",
        DEFAULT_LLM_CONTEXT_WINDOW_TOKENS,
        values,
        minimum=1,
    )
    reserved_output_tokens = _int_env_value(
        "LLM_RESERVED_OUTPUT_TOKENS",
        DEFAULT_LLM_RESERVED_OUTPUT_TOKENS,
        values,
        minimum=0,
    )
    context_safety_margin_tokens = _int_env_value(
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
        DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS,
        values,
        minimum=0,
    )
    memory_context_tokens = _int_env_value(
        "LLM_MEMORY_CONTEXT_TOKENS",
        DEFAULT_LLM_MEMORY_CONTEXT_TOKENS,
        values,
        minimum=0,
    )
    stream_include_usage = _bool_env_value(
        "LLM_STREAM_INCLUDE_USAGE",
        DEFAULT_LLM_STREAM_INCLUDE_USAGE,
        values,
    )
    thinking_enabled = _optional_bool_env_value(
        "LLM_THINKING_ENABLED",
        values,
    )
    reasoning_effort = _optional_reasoning_effort_env_value(
        "LLM_REASONING_EFFORT",
        values,
    )
    max_output_tokens = _optional_int_env_value(
        "LLM_MAX_OUTPUT_TOKENS",
        values,
        minimum=1,
    )

    if reserved_output_tokens + context_safety_margin_tokens >= context_window_tokens:
        raise ValueError(
            "LLM_RESERVED_OUTPUT_TOKENS and LLM_CONTEXT_SAFETY_MARGIN_TOKENS "
            "must leave at least 1 input token."
        )

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        compact_model=compact_model,
        subagent_model=subagent_model,
        context_window_tokens=context_window_tokens,
        reserved_output_tokens=reserved_output_tokens,
        context_safety_margin_tokens=context_safety_margin_tokens,
        memory_context_tokens=memory_context_tokens,
        stream_include_usage=stream_include_usage,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )


def _required_env_value(
    name: str,
    *value_maps: Mapping[str, str | None],
) -> str:
    value = _env_value(name, "", *value_maps)

    if value == "":
        raise MissingLLMConfigError(f"Missing required environment variable: {name}")

    return value


def _env_value(
    name: str,
    default: str,
    *value_maps: Mapping[str, str | None],
) -> str:
    values = [value_map.get(name) for value_map in value_maps]

    for value in values:
        if value is None:
            continue

        value = value.strip()
        if value != "":
            return value

    return default


def _int_env_value(
    name: str,
    default: int,
    *value_maps: Mapping[str, str | None],
    minimum: int,
) -> int:
    raw_value = _env_value(name, str(default), *value_maps)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    return value


def _bool_env_value(
    name: str,
    default: bool,
    *value_maps: Mapping[str, str | None],
) -> bool:
    raw_value = _env_value(name, str(default).lower(), *value_maps).lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be true or false.")


def _optional_bool_env_value(
    name: str,
    *value_maps: Mapping[str, str | None],
) -> bool | None:
    raw_value = _optional_env_value(name, *value_maps)
    if raw_value is None:
        return None
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _optional_int_env_value(
    name: str,
    *value_maps: Mapping[str, str | None],
    minimum: int,
) -> int | None:
    raw_value = _optional_env_value(name, *value_maps)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _optional_reasoning_effort_env_value(
    name: str,
    *value_maps: Mapping[str, str | None],
) -> ReasoningEffort | None:
    raw_value = _optional_env_value(name, *value_maps)
    if raw_value is None:
        return None
    normalized = raw_value.lower()
    if normalized not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(f"{name} must be one of: high, max.")
    return normalized  # type: ignore[return-value]


def _optional_env_value(
    name: str,
    *value_maps: Mapping[str, str | None],
) -> str | None:
    values = [value_map.get(name) for value_map in value_maps]
    for value in values:
        if value is None:
            continue
        normalized = value.strip()
        if normalized != "":
            return normalized
    return None
