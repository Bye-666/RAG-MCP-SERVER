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