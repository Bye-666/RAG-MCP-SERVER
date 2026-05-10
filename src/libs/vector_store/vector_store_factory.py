from .base_vector_store import BaseVectorStore
from typing import Dict, Type


class VectorStoreFactory:
    _providers: Dict[str, Type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(cls, name: str, store_class: Type[BaseVectorStore]):
        cls._providers[name] = store_class

    @classmethod
    def create(cls, settings: dict) -> BaseVectorStore:
        provider = settings['vector_store']['provider']
        store_class = cls._providers.get(provider)

        if not store_class:
            raise ValueError(f"Unsupported VectorStore provider: {provider}")

        return store_class(**settings['vector_store'])


# Register built-in providers
try:
    from .chroma_store import ChromaStore
    VectorStoreFactory.register_provider("chroma", ChromaStore)
except ImportError:
    pass  # chromadb package not installed

try:
    from .qdrant_vector_store import QdrantVectorStore
    VectorStoreFactory.register_provider("qdrant", QdrantVectorStore)
except ImportError:
    pass  # qdrant-client package not installed


# Convenience function for backward compatibility
def create_vector_store(settings) -> BaseVectorStore:
    """
    创建 VectorStore 的便捷函数

    Args:
        settings: Settings 对象或配置字典

    Returns:
        BaseVectorStore 实例
    """
    # 如果是 Settings 对象，转换为字典
    if hasattr(settings, '__dict__'):
        settings_dict = settings.__dict__
    else:
        settings_dict = settings

    return VectorStoreFactory.create(settings_dict)