from .base_evaluator import BaseEvaluator
from typing import Dict, Type


class EvaluatorFactory:
    _providers: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register_provider(cls, name: str, evaluator_class: Type[BaseEvaluator]):
        cls._providers[name] = evaluator_class

    @classmethod
    def create(cls, settings: dict) -> BaseEvaluator:
        provider = settings.get('evaluator', {}).get('provider', 'custom')
        evaluator_class = cls._providers.get(provider)

        if not evaluator_class:
            raise ValueError(f"Unsupported Evaluator provider: {provider}")

        return evaluator_class()


# 注册 ragas provider
def _register_ragas():
    """延迟注册 ragas provider，避免导入错误"""
    try:
        from ...observability.evaluation.ragas_evaluator import RagasEvaluator
        EvaluatorFactory.register_provider('ragas', RagasEvaluator)
    except ImportError:
        # Ragas 未安装时跳过注册
        pass


_register_ragas()