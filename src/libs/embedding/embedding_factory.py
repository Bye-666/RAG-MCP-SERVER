from typing import Dict, Type
from .base_embedding import BaseEmbedding


class EmbeddingFactory:
    _providers: Dict[str, Type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(cls, name: str, embed_class: Type[BaseEmbedding]):
        cls._providers[name] = embed_class

    @classmethod
    def create(cls, settings: dict) -> BaseEmbedding:
        provider = settings['embedding']['provider']
        embed_class = cls._providers.get(provider)

        if not embed_class:
            raise ValueError(f"不支持的嵌入提供商: {provider}")

        # 创建配置副本并移除 provider 字段，避免传递给构造函数
        embed_config = {k: v for k, v in settings['embedding'].items() if k != 'provider'}
        return embed_class(**embed_config)

# Register built-in providers
try:
    from .openai_embedding import OpenAIEmbedding
    EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
except ImportError:
    pass  # openai 包未安装

try:
    from .azure_embedding import AzureEmbedding
    EmbeddingFactory.register_provider("azure", AzureEmbedding)
except ImportError:
    pass  # azure 包未安装

try:
    from .ollama_embedding import OllamaEmbedding
    EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
except ImportError:
    pass  # requests 包未安装


# Convenience function for backward compatibility
def create_embedding_client(settings) -> BaseEmbedding:
    """
    创建 Embedding 客户端的便捷函数

    Args:
        settings: Settings 对象或配置字典

    Returns:
        BaseEmbedding 实例
    """
    # 如果是 Settings 对象，转换为字典
    if hasattr(settings, '__dict__'):
        settings_dict = settings.__dict__
    else:
        settings_dict = settings

    return EmbeddingFactory.create(settings_dict)
