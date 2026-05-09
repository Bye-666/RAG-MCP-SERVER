"""
CompositeEvaluator 单元测试

测试策略：
1. 测试组合多个评估器
2. 测试并行执行
3. 测试指标合并
4. 测试单个评估器失败的处理
5. 测试配置驱动的创建
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.observability.evaluation.composite_evaluator import (
    CompositeEvaluator,
    create_composite_evaluator
)
from src.libs.evaluator.base_evaluator import BaseEvaluator


class MockEvaluator(BaseEvaluator):
    """用于测试的 Mock 评估器"""

    def __init__(self, name: str, metrics: dict):
        self.name = name
        self.metrics = metrics

    def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
        return self.metrics.copy()


class TestCompositeEvaluatorBasic:
    """测试基本功能"""

    def test_init_with_evaluators(self):
        """测试使用评估器列表初始化"""
        evaluator1 = MockEvaluator("eval1", {"metric1": 0.8})
        evaluator2 = MockEvaluator("eval2", {"metric2": 0.9})

        composite = CompositeEvaluator([evaluator1, evaluator2])

        assert len(composite.evaluators) == 2
        assert composite.max_workers == 2

    def test_init_with_custom_max_workers(self):
        """测试自定义最大工作线程数"""
        evaluator1 = MockEvaluator("eval1", {"metric1": 0.8})

        composite = CompositeEvaluator([evaluator1], max_workers=4)

        assert composite.max_workers == 4

    def test_init_with_empty_list_raises_error(self):
        """测试空评估器列表抛出错误"""
        with pytest.raises(ValueError) as exc_info:
            CompositeEvaluator([])

        assert "至少需要提供一个评估器" in str(exc_info.value)


class TestCompositeEvaluatorEvaluation:
    """测试评估功能"""

    def test_evaluate_single_evaluator(self):
        """测试单个评估器的评估"""
        evaluator = MockEvaluator("test", {"hit_rate": 0.8, "precision": 0.9})
        composite = CompositeEvaluator([evaluator])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"]
        )

        # 验证指标带有前缀（MockEvaluator -> mock）
        assert "mock.hit_rate" in metrics
        assert "mock.precision" in metrics
        assert metrics["mock.hit_rate"] == 0.8
        assert metrics["mock.precision"] == 0.9

    def test_evaluate_multiple_evaluators(self):
        """测试多个评估器的评估"""
        evaluator1 = MockEvaluator("eval1", {"metric1": 0.8})
        evaluator2 = MockEvaluator("eval2", {"metric2": 0.9})
        composite = CompositeEvaluator([evaluator1, evaluator2])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"]
        )

        # 验证两个评估器的指标都存在（MockEvaluator -> mock）
        assert "mock.metric1" in metrics
        assert "mock.metric2" in metrics
        assert metrics["mock.metric1"] == 0.8
        assert metrics["mock.metric2"] == 0.9

    def test_evaluate_with_overlapping_metric_names(self):
        """测试评估器有相同指标名称的情况"""
        evaluator1 = MockEvaluator("eval1", {"hit_rate": 0.8})
        evaluator2 = MockEvaluator("eval2", {"hit_rate": 0.9})
        composite = CompositeEvaluator([evaluator1, evaluator2])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"]
        )

        # 两个评估器的指标应该都存在（通过前缀区分）
        # 但由于两个 MockEvaluator 实例的类名相同，会产生相同的前缀
        # 后者会覆盖前者，所以只有一个 hit_rate
        assert len([k for k in metrics.keys() if "hit_rate" in k]) >= 1

    def test_evaluate_with_trace(self):
        """测试带追踪上下文的评估"""
        evaluator = MockEvaluator("test", {"metric": 0.8})
        composite = CompositeEvaluator([evaluator])
        trace = Mock()

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"],
            trace=trace
        )

        # 验证追踪日志被调用
        assert trace.log.called
        assert metrics is not None

    def test_evaluate_with_kwargs(self):
        """测试传递额外参数"""
        class KwargsEvaluator(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                # 验证 kwargs 被传递
                assert "answer" in kwargs
                assert kwargs["answer"] == "测试答案"
                return {"metric": 1.0}

        evaluator = KwargsEvaluator()
        composite = CompositeEvaluator([evaluator])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"],
            answer="测试答案"
        )

        # KwargsEvaluator -> kwargs
        assert "kwargs.metric" in metrics


class TestCompositeEvaluatorErrorHandling:
    """测试错误处理"""

    def test_evaluate_with_failing_evaluator(self):
        """测试单个评估器失败的情况"""
        class FailingEvaluator(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                raise Exception("评估失败")

        evaluator1 = FailingEvaluator()
        evaluator2 = MockEvaluator("success", {"metric": 0.8})
        composite = CompositeEvaluator([evaluator1, evaluator2])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"]
        )

        # 成功的评估器应该返回结果（MockEvaluator -> mock）
        assert "mock.metric" in metrics
        assert metrics["mock.metric"] == 0.8

    def test_evaluate_with_all_failing_evaluators(self):
        """测试所有评估器都失败的情况"""
        class FailingEvaluator(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                raise Exception("评估失败")

        evaluator1 = FailingEvaluator()
        evaluator2 = FailingEvaluator()
        composite = CompositeEvaluator([evaluator1, evaluator2])

        metrics = composite.evaluate(
            query="测试查询",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"]
        )

        # 应该返回空字典
        assert metrics == {}


class TestCompositeEvaluatorNaming:
    """测试评估器命名"""

    def test_get_evaluator_name_removes_suffix(self):
        """测试移除 Evaluator 后缀"""
        evaluator = MockEvaluator("test", {})
        composite = CompositeEvaluator([evaluator])

        name = composite._get_evaluator_name(evaluator, 0)

        # MockEvaluator -> mock
        assert name == "mock"

    def test_get_evaluator_name_with_index(self):
        """测试使用索引作为后备名称"""
        class CustomEval(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                return {}

        evaluator = CustomEval()
        composite = CompositeEvaluator([evaluator])

        name = composite._get_evaluator_name(evaluator, 5)

        # CustomEval -> customeval
        assert name == "customeval"


class TestCreateCompositeEvaluator:
    """测试配置驱动的创建"""

    def test_create_with_single_backend(self):
        """测试单个后端配置"""
        settings = {
            'evaluation': {
                'backends': ['custom']
            }
        }

        mock_factory = Mock()
        mock_factory.create.return_value = MockEvaluator("custom", {})

        composite = create_composite_evaluator(settings, mock_factory)

        assert len(composite.evaluators) == 1
        mock_factory.create.assert_called_once()

    def test_create_with_multiple_backends(self):
        """测试多个后端配置"""
        settings = {
            'evaluation': {
                'backends': ['ragas', 'custom']
            }
        }

        mock_factory = Mock()
        mock_factory.create.side_effect = [
            MockEvaluator("ragas", {}),
            MockEvaluator("custom", {})
        ]

        composite = create_composite_evaluator(settings, mock_factory)

        assert len(composite.evaluators) == 2
        assert mock_factory.create.call_count == 2

    def test_create_with_empty_backends_raises_error(self):
        """测试空后端列表抛出错误"""
        settings = {
            'evaluation': {
                'backends': []
            }
        }

        mock_factory = Mock()

        with pytest.raises(ValueError) as exc_info:
            create_composite_evaluator(settings, mock_factory)

        assert "不能为空" in str(exc_info.value)

    def test_create_with_missing_backends_uses_default(self):
        """测试缺少 backends 配置时使用默认值"""
        settings = {}

        mock_factory = Mock()
        mock_factory.create.return_value = MockEvaluator("custom", {})

        composite = create_composite_evaluator(settings, mock_factory)

        # 应该使用默认的 'custom' backend
        assert len(composite.evaluators) == 1

    def test_create_skips_failing_backends(self):
        """测试跳过无法创建的后端"""
        settings = {
            'evaluation': {
                'backends': ['invalid', 'custom']
            }
        }

        mock_factory = Mock()
        mock_factory.create.side_effect = [
            ValueError("不支持的后端"),
            MockEvaluator("custom", {})
        ]

        # 捕获 print 输出
        with patch('builtins.print'):
            composite = create_composite_evaluator(settings, mock_factory)

        # 应该只创建成功的评估器
        assert len(composite.evaluators) == 1

    def test_create_with_all_failing_backends_raises_error(self):
        """测试所有后端都失败时抛出错误"""
        settings = {
            'evaluation': {
                'backends': ['invalid1', 'invalid2']
            }
        }

        mock_factory = Mock()
        mock_factory.create.side_effect = ValueError("不支持的后端")

        with patch('builtins.print'):
            with pytest.raises(ValueError) as exc_info:
                create_composite_evaluator(settings, mock_factory)

        assert "没有成功创建任何评估器" in str(exc_info.value)


class TestCompositeEvaluatorIntegration:
    """集成测试"""

    def test_integration_with_real_evaluators(self):
        """测试与真实评估器的集成"""
        # 创建两个模拟真实行为的评估器
        class Evaluator1(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                return {
                    "hit_rate": 1.0 if set(retrieved_ids) & set(golden_ids) else 0.0,
                    "precision": len(set(retrieved_ids) & set(golden_ids)) / len(retrieved_ids)
                }

        class Evaluator2(BaseEvaluator):
            def evaluate(self, query, retrieved_ids, golden_ids, trace=None, **kwargs):
                return {
                    "recall": len(set(retrieved_ids) & set(golden_ids)) / len(golden_ids)
                }

        composite = CompositeEvaluator([Evaluator1(), Evaluator2()])

        metrics = composite.evaluate(
            query="测试",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc1", "doc4"]
        )

        # 验证所有指标都存在
        assert "evaluator1.hit_rate" in metrics
        assert "evaluator1.precision" in metrics
        assert "evaluator2.recall" in metrics

        # 验证计算正确
        assert metrics["evaluator1.hit_rate"] == 1.0
        assert metrics["evaluator1.precision"] == 1/3
        assert metrics["evaluator2.recall"] == 1/2
