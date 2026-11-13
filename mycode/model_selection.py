"""Serializable role intent. No credentials and no effective client snapshots."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_CONNECTION = "env"


def injected_reference(reference: str) -> str:
    return "@injected:" + reference


class RoleSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    connection_ref: str | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    thinking: Literal["auto", "on", "off"] | None = None
    effort: Literal["auto", "low", "high", "max"] | None = None
    context_window_tokens: int | None = Field(default=None, ge=1)
    window_binding: str | None = None
    reserved_output_tokens: int | None = Field(default=None, ge=0)
    context_safety_margin_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: Annotated[int, Field(ge=1)] | Literal["auto"] | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Model ID must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_intent(self):
        if self.connection_ref is not None and self.model is None:
            raise ValueError("A fixed connection requires a fixed model.")
        if self.effort not in {None, "auto"} and self.thinking in {"auto", "off"}:
            raise ValueError("Effort requires thinking=on.")
        return self


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    main: RoleSelection
    subagent: RoleSelection = Field(default_factory=RoleSelection)
    compact: RoleSelection = Field(default_factory=RoleSelection)

    @model_validator(mode="after")
    def main_is_fixed(self):
        if self.main.connection_ref is None or self.main.model is None:
            raise ValueError("Main requires a connection and model.")
        return self

    def identity(self, role):
        setting = getattr(self, role)
        return setting.connection_ref or self.main.connection_ref, setting.model or self.main.model

    def references(self) -> frozenset[str]:
        return frozenset(self.identity(role)[0] for role in ("main", "subagent", "compact")) - {ENV_CONNECTION}
