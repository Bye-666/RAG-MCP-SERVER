from .base_reranker import BaseReranker
from typing import Any, Dict, List, Type


class NoneReranker(BaseReranker):
    """Pass-through reranker that maintains original order"""

    def rerank(self, query, candidates, trace=None):
        return candidates


class RerankerFactory:
    _providers: Dict[str, Type[BaseReranker]] = {"none": NoneReranker}

    @classmethod
    def register_provider(cls, name: str, reranker_class: Type[BaseReranker]):
        cls._providers[name] = reranker_class

    @classmethod
    def create(cls, settings: dict) -> BaseReranker:
        provider = settings.get('reranker', {}).get('provider', 'none')
        reranker_class = cls._providers.get(provider)

        if not reranker_class:
            raise ValueError(f"Unsupported Reranker provider: {provider}")

        return reranker_class()