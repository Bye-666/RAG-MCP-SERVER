import pytest
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestCustomEvaluator:
    def setup_method(self):
        EvaluatorFactory._providers = {}

    def test_hit_rate_hit(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2"], ["doc2", "doc3"])
        assert result['hit_rate'] == 1.0

    def test_hit_rate_miss(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2"], ["doc99"])
        assert result['hit_rate'] == 0.0

    def test_mrr_first_position(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2"], ["doc1"])
        assert result['mrr'] == 1.0

    def test_mrr_second_position(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc2", "doc1"], ["doc1"])
        assert result['mrr'] == 0.5

    def test_mrr_no_hit(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2"], ["doc99"])
        assert result['mrr'] == 0.0

    def test_factory_create(self):
        EvaluatorFactory.register_provider("custom", CustomEvaluator)
        settings = {"evaluator": {"provider": "custom"}}
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CustomEvaluator)

    def test_factory_invalid_provider(self):
        settings = {"evaluator": {"provider": "invalid"}}
        with pytest.raises(ValueError):
            EvaluatorFactory.create(settings)