"""Bind user role intent to complete connections and resolve client snapshots."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from mycode.config import (
    DEFAULT_LLM_CONTEXT_WINDOW_TOKENS, DEFAULT_LLM_MEMORY_CONTEXT_TOKENS,
    DEFAULT_LLM_RESERVED_OUTPUT_TOKENS, DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS,
    LLMConfig, freeze_headers,
)
from mycode.model_selection import ENV_CONNECTION, ModelSelection, RoleSelection
from mycode.model_windows import ModelWindow

ModelRole = Literal["main", "subagent", "compact"]


@dataclass(frozen=True)
class APIConnectionConfig:
    api_key: str = field(repr=False)
    base_url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for name in ("api_key", "base_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "headers", freeze_headers(self.headers))


@dataclass(frozen=True)
class ModelConfiguration:
    selection: ModelSelection
    connections: Mapping[str, APIConnectionConfig] = field(repr=False)
    memory_context_tokens: int = DEFAULT_LLM_MEMORY_CONTEXT_TOKENS
    stream_include_usage: bool = True
    windows: Mapping[str, ModelWindow] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "connections", MappingProxyType(dict(self.connections)))
        object.__setattr__(self, "windows", MappingProxyType(dict(self.windows)))

    @classmethod
    def from_config(cls, config: LLMConfig):
        reference = ENV_CONNECTION if config.env_source else "@bootstrap"
        main = RoleSelection(connection_ref=reference, model=config.model,
            thinking={True: "on", False: "off", None: "auto"}[config.thinking_enabled],
            effort=config.reasoning_effort or "auto", max_output_tokens=config.max_output_tokens or "auto",
            context_window_tokens=(config.context_window_tokens
                if config.context_window_tokens != DEFAULT_LLM_CONTEXT_WINDOW_TOKENS else None),
            reserved_output_tokens=config.reserved_output_tokens,
            context_safety_margin_tokens=config.context_safety_margin_tokens)
        return cls(selection=ModelSelection(main=main,
            subagent=RoleSelection(model=config.subagent_model), compact=RoleSelection(model=config.compact_model)),
            connections={reference: APIConnectionConfig(config.api_key, config.base_url, config.headers)},
            memory_context_tokens=config.memory_context_tokens, stream_include_usage=config.stream_include_usage)

    def resolve(self, role: ModelRole) -> LLMConfig:
        if role not in {"main", "subagent", "compact"}:
            raise ValueError("Unknown model role.")
        ref, model = self.selection.identity(role)
        if ref not in self.connections:
            raise ValueError("Saved connection is unavailable; choose an existing connection.")
        setting = getattr(self.selection, role)
        parent = None if role == "main" else self.resolve("main")
        follows = parent is not None and self.selection.identity(role) == self.selection.identity("main")
        thinking = ({"auto": None, "on": True, "off": False}[setting.thinking] if setting.thinking is not None
            else False if role == "compact" else parent.thinking_enabled if follows else None)
        effort = setting.effort
        if effort is None:
            effort = parent.reasoning_effort if follows and thinking is True else None
        if effort == "auto":
            effort = None
        output = setting.max_output_tokens
        if output is None:
            output = parent.max_output_tokens if follows else None
        if output == "auto":
            output = None
        reserves = setting.reserved_output_tokens
        if reserves is None:
            reserves = parent.reserved_output_tokens if follows else DEFAULT_LLM_RESERVED_OUTPUT_TOKENS
        margin = setting.context_safety_margin_tokens
        if margin is None:
            margin = parent.context_safety_margin_tokens if follows else DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS
        connection = self.connections[ref]
        window = self.windows.get(role)
        tokens = window.tokens if window is not None else setting.context_window_tokens
        if tokens is None:
            tokens = parent.context_window_tokens if follows else DEFAULT_LLM_CONTEXT_WINDOW_TOKENS
        return LLMConfig(api_key=connection.api_key, base_url=connection.base_url, headers=connection.headers,
            model=model, thinking_enabled=thinking, reasoning_effort=effort, max_output_tokens=output,
            context_window_tokens=tokens, reserved_output_tokens=reserves,
            context_safety_margin_tokens=margin, memory_context_tokens=self.memory_context_tokens,
            stream_include_usage=self.stream_include_usage, env_source=False)
