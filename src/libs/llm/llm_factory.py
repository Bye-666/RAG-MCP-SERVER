from .base_llm import BaseLLM
from .base_vision_llm import BaseVisionLLM
from .openai_llm import OpenAILLM
from .azure_llm import AzureLLM
from .deepseek_llm import DeepSeekLLM
from .ollama_llm import OllamaLLM
from .qwen_llm import QwenLLM
from typing import Dict, Type


class LLMFactory:
    _providers: Dict[str, Type[BaseLLM]] = {}
    _vision_providers: Dict[str, Type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        cls._providers[name] = llm_class

    @classmethod
    def register_vision_provider(cls, name: str, vision_llm_class: Type[BaseVisionLLM]):
        """注册 Vision LLM 提供商

        参数:
            name: 提供商名称（例如 "azure"、"openai"）
            vision_llm_class: 实现 BaseVisionLLM 的 Vision LLM 类
        """
        cls._vision_providers[name] = vision_llm_class

    @classmethod
    def create(cls, settings: dict) -> BaseLLM:
        provider = settings['llm']['provider']
        llm_class = cls._providers.get(provider)

        if not llm_class:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

        return llm_class(**settings['llm'])

    @classmethod
    def create_vision_llm(cls, settings: dict) -> BaseVisionLLM:
        """从配置创建 Vision LLM 实例

        参数:
            settings: 包含 'vision_llm' 键的配置字典，其中包含:
                - provider: Vision LLM 提供商名称（例如 "azure"、"openai"）
                - 其他提供商特定参数

        返回:
            BaseVisionLLM 实例

        异常:
            ValueError: 如果提供商不受支持或配置无效
            KeyError: 如果配置中缺少 'vision_llm' 键

        示例:
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
            raise KeyError("配置必须包含 'vision_llm' 键")

        provider = settings['vision_llm'].get('provider')
        if not provider:
            raise ValueError("配置中未指定 Vision LLM 提供商")

        vision_llm_class = cls._vision_providers.get(provider)

        if not vision_llm_class:
            raise ValueError(
                f"不支持的 Vision LLM 提供商: {provider}。"
                f"可用的提供商: {list(cls._vision_providers.keys())}"
            )

        return vision_llm_class(**settings['vision_llm'])

# Register built-in providers
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)
LLMFactory.register_provider("qwen", QwenLLM)

# Register Vision LLM providers
try:
    from .azure_vision_llm import AzureVisionLLM
    LLMFactory.register_vision_provider("azure", AzureVisionLLM)
except ImportError:
    pass  # Azure Vision 依赖不可用