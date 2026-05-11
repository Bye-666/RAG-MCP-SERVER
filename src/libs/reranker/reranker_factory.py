from .base_reranker import BaseReranker
from typing import Any, Dict, List, Type


class NoneReranker(BaseReranker):
    """直通重排序器，保持原始顺序"""

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
            raise ValueError(f"不支持的重排序器提供商: {provider}")

        # LLM 重排序器的特殊处理，需要 LLM 实例
        if provider == 'llm':
            from ..llm.llm_factory import LLMFactory
            llm_settings = settings.get('llm', {})
            llm = LLMFactory.create({'llm': llm_settings})

            reranker_settings = settings.get('reranker', {})
            prompt_path = reranker_settings.get('prompt_path')

            return reranker_class(llm=llm, prompt_path=prompt_path)

        # Cross-Encoder 重排序器的特殊处理，需要模型配置
        if provider == 'cross_encoder':
            reranker_settings = settings.get('reranker', {})
            model_name = reranker_settings.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            timeout = reranker_settings.get('timeout')

            return reranker_class(model_name=model_name, timeout=timeout)

        return reranker_class()


# Register LLM reranker
try:
    from .llm_reranker import LLMReranker
    RerankerFactory.register_provider("llm", LLMReranker)
except ImportError:
    pass  # LLM 依赖不可用

# Register Cross-Encoder reranker
try:
    from .cross_encoder_reranker import CrossEncoderReranker
    RerankerFactory.register_provider("cross_encoder", CrossEncoderReranker)
except ImportError:
    pass  # sentence-transformers 不可用
