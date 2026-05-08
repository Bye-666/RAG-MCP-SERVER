from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .azure_llm import AzureLLM
from .deepseek_llm import DeepSeekLLM
from .ollama_llm import OllamaLLM
from typing import Dict, Type


class LLMFactory:
    _providers: Dict[str, Type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        cls._providers[name] = llm_class

    @classmethod
    def create(cls, settings: dict) -> BaseLLM:
        provider = settings['llm']['provider']
        llm_class = cls._providers.get(provider)

        if not llm_class:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return llm_class(**settings['llm'])

# Register built-in providers
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)