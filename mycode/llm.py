"""Compatibility exports for the established LLM API.

Core runtime modules import ``mycode.llm_contracts`` directly so loading the
provider-neutral contract does not import the OpenAI SDK.
"""

from mycode.llm_contracts import FakeLLMClient, LLMClient
from mycode.providers.openai_compatible import OpenAICompatibleLLMClient

__all__ = ["FakeLLMClient", "LLMClient", "OpenAICompatibleLLMClient"]
