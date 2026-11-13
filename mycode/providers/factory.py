"""Create independent clients from a validated, session-scoped model snapshot."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType

from mycode.config import LLMConfig
from mycode.model_config import ModelConfiguration, ModelRole
from mycode.providers.openai_compatible import OpenAICompatibleLLMClient


@dataclass(frozen=True)
class ModelClientFactory:
    configuration: ModelConfiguration
    session_id: str = field(repr=False)
    _configs: Mapping[str, LLMConfig] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_configs", MappingProxyType({
            role: self.configuration.resolve(role)
            for role in ("main", "subagent", "compact")
        }))

    def config_for(self, role: ModelRole) -> LLMConfig:
        if role not in self._configs:
            raise ValueError("Unknown model role.")
        return self._configs[role]

    def create(self, role: ModelRole) -> OpenAICompatibleLLMClient:
        # Never cache a client: each SubAgent run needs independent usage and
        # response state, while all roles keep the parent session transport ID.
        return OpenAICompatibleLLMClient(
            config=self.config_for(role), session_id=self.session_id,
        )
