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

        # Special handling for LLM reranker which needs an LLM instance
        if provider == 'llm':
            from ..llm.llm_factory import LLMFactory
            llm_settings = settings.get('llm', {})
            llm = LLMFactory.create({'llm': llm_settings})

            reranker_settings = settings.get('reranker', {})
            prompt_path = reranker_settings.get('prompt_path')

            return reranker_class(llm=llm, prompt_path=prompt_path)

        return reranker_class()


# Register LLM reranker
try:
    from .llm_reranker import LLMReranker
    RerankerFactory.register_provider("llm", LLMReranker)
except ImportError:
    pass  # LLM dependencies not available
