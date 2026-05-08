from .base_llm import BaseLLM
from .base_vision_llm import BaseVisionLLM
from .openai_llm import OpenAILLM
from .azure_llm import AzureLLM
from .deepseek_llm import DeepSeekLLM
from .ollama_llm import OllamaLLM
from typing import Dict, Type


class LLMFactory:
    _providers: Dict[str, Type[BaseLLM]] = {}
    _vision_providers: Dict[str, Type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        cls._providers[name] = llm_class

    @classmethod
    def register_vision_provider(cls, name: str, vision_llm_class: Type[BaseVisionLLM]):
        """Register a Vision LLM provider

        Args:
            name: Provider name (e.g., "azure", "openai")
            vision_llm_class: Vision LLM class implementing BaseVisionLLM
        """
        cls._vision_providers[name] = vision_llm_class

    @classmethod
    def create(cls, settings: dict) -> BaseLLM:
        provider = settings['llm']['provider']
        llm_class = cls._providers.get(provider)

        if not llm_class:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return llm_class(**settings['llm'])

    @classmethod
    def create_vision_llm(cls, settings: dict) -> BaseVisionLLM:
        """Create a Vision LLM instance from settings

        Args:
            settings: Configuration dict with 'vision_llm' key containing:
                - provider: Vision LLM provider name (e.g., "azure", "openai")
                - Additional provider-specific parameters

        Returns:
            BaseVisionLLM instance

        Raises:
            ValueError: If provider is not supported or settings are invalid
            KeyError: If 'vision_llm' key is missing from settings

        Example:
            settings = {
                "vision_llm": {
                    "provider": "azure",
                    "api_key": "...",
                    "azure_endpoint": "...",
                    "api_version": "...",
                    "deployment_name": "gpt-4o"
                }
            }
            vision_llm = LLMFactory.create_vision_llm(settings)
        """
        if 'vision_llm' not in settings:
            raise KeyError("Settings must contain 'vision_llm' key")

        provider = settings['vision_llm'].get('provider')
        if not provider:
            raise ValueError("Vision LLM provider not specified in settings")

        vision_llm_class = cls._vision_providers.get(provider)

        if not vision_llm_class:
            raise ValueError(
                f"Unsupported Vision LLM provider: {provider}. "
                f"Available providers: {list(cls._vision_providers.keys())}"
            )

        return vision_llm_class(**settings['vision_llm'])

# Register built-in providers
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)

# Vision LLM providers will be registered when their modules are imported
# Example: from .azure_vision_llm import AzureVisionLLM
#          LLMFactory.register_vision_provider("azure", AzureVisionLLM)