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