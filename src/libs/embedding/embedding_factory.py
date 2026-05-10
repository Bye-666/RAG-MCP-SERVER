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
            raise ValueError(f"Unsupported embedding provider: {provider}")

        return embed_class(**settings['embedding'])

# Register built-in providers
try:
    from .openai_embedding import OpenAIEmbedding
    EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
except ImportError:
    pass  # openai package not installed

try:
    from .azure_embedding import AzureEmbedding
    EmbeddingFactory.register_provider("azure", AzureEmbedding)
except ImportError:
    pass  # azure package not installed

try:
    from .ollama_embedding import OllamaEmbedding
    EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
except ImportError:
    pass  # requests package not installed


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
