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

    def test_evaluate_empty_retrieved(self):
        """边界测试：retrieved为空"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", [], ["doc1"])
        assert result['hit_rate'] == 0.0
        assert result['mrr'] == 0.0

    def test_evaluate_empty_ground_truth(self):
        """边界测试：ground_truth为空"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2"], [])
        assert result['hit_rate'] == 0.0
        assert result['mrr'] == 0.0

    def test_evaluate_both_empty(self):
        """边界测试：两者都为空"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", [], [])
        assert result['hit_rate'] == 0.0
        assert result['mrr'] == 0.0

    def test_evaluate_multiple_ground_truth(self):
        """边界测试：多个ground_truth"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1", "doc2", "doc3"], ["doc2", "doc3"])
        assert result['hit_rate'] == 1.0  # 至少命中一个
        assert result['mrr'] > 0.0  # 应该有排名

    def test_evaluate_result_completeness(self):
        """边界测试：返回结果包含所有必需字段"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["doc1"], ["doc1"])
        assert 'hit_rate' in result
        assert 'mrr' in result
        assert isinstance(result['hit_rate'], (int, float))
        assert isinstance(result['mrr'], (int, float))

    def test_evaluate_case_sensitivity(self):
        """边界测试：ID大小写敏感性"""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate("query", ["Doc1", "doc2"], ["doc1"])
        # 应该区分大小写，不匹配
        assert result['hit_rate'] == 0.0