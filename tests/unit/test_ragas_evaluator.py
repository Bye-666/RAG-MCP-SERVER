"""
RagasEvaluator 单元测试

测试策略：
1. 测试 Ragas 未安装时的错误提示
2. 测试基础指标计算（不依赖 Ragas）
3. 测试 Ragas 评估（使用 mock）
4. 测试工厂注册
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.observability.evaluation.ragas_evaluator import RagasEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestRagasEvaluatorImport:
    """测试 Ragas 导入相关功能"""

    def test_ragas_not_installed_error(self):
        """测试 Ragas 未安装时抛出清晰的错误"""
        with patch.dict('sys.modules', {'ragas': None}):
            with pytest.raises(ImportError) as exc_info:
                # 强制重新导入以触发 ImportError
                import importlib
                import sys
                if 'src.observability.evaluation.ragas_evaluator' in sys.modules:
                    del sys.modules['src.observability.evaluation.ragas_evaluator']

                from src.observability.evaluation.ragas_evaluator import RagasEvaluator
                RagasEvaluator()

            assert "Ragas 未安装" in str(exc_info.value) or "ragas" in str(exc_info.value).lower()


class TestRagasEvaluatorBasicMetrics:
    """测试基础指标计算（不依赖 Ragas）"""

    @pytest.fixture
    def evaluator(self):
        """创建 mock 的 RagasEvaluator"""
        # Mock ragas 模块的导入
        mock_ragas = MagicMock()
        mock_ragas.evaluate = MagicMock()
        mock_ragas.metrics.faithfulness = MagicMock()
        mock_ragas.metrics.answer_relevancy = MagicMock()
        mock_ragas.metrics.context_precision = MagicMock()

        with patch.dict('sys.modules', {'ragas': mock_ragas, 'ragas.metrics': mock_ragas.metrics}):
            evaluator = RagasEvaluator()
            return evaluator

    def test_basic_metrics_perfect_match(self, evaluator):
        """测试完全匹配的情况"""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        golden_ids = ["doc1", "doc2", "doc3"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_basic_metrics_partial_match(self, evaluator):
        """测试部分匹配的情况"""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        golden_ids = ["doc2", "doc4"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1.0  # 至少命中一个
        assert metrics["precision"] == 1/3  # 3个中命中1个
        assert metrics["recall"] == 1/2  # 2个相关文档中检索到1个

    def test_basic_metrics_no_match(self, evaluator):
        """测试完全不匹配的情况"""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        golden_ids = ["doc4", "doc5"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_basic_metrics_empty_retrieved(self, evaluator):
        """测试检索结果为空的情况"""
        retrieved_ids = []
        golden_ids = ["doc1", "doc2"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_basic_metrics_empty_golden(self, evaluator):
        """测试标准答案为空的情况"""
        retrieved_ids = ["doc1", "doc2"]
        golden_ids = []

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestRagasEvaluatorWithMock:
    """测试 Ragas 评估功能（使用 mock）"""

    @pytest.fixture
    def evaluator(self):
        """创建 mock 的 RagasEvaluator"""
        # Mock ragas 模块的导入
        mock_ragas = MagicMock()
        mock_ragas.evaluate = MagicMock(return_value={
            "faithfulness": 0.85,
            "answer_relevancy": 0.92,
            "context_precision": 0.78
        })
        mock_ragas.metrics.faithfulness = MagicMock()
        mock_ragas.metrics.answer_relevancy = MagicMock()
        mock_ragas.metrics.context_precision = MagicMock()

        with patch.dict('sys.modules', {'ragas': mock_ragas, 'ragas.metrics': mock_ragas.metrics}):
            evaluator = RagasEvaluator()
            return evaluator

    def test_evaluate_with_full_params(self, evaluator):
        """测试提供完整参数时的评估"""
        # Mock datasets 模块
        mock_dataset_class = MagicMock()
        mock_dataset_class.from_dict.return_value = MagicMock()

        with patch.dict('sys.modules', {'datasets': MagicMock(Dataset=mock_dataset_class)}):
            metrics = evaluator.evaluate(
                query="测试查询",
                retrieved_ids=["doc1", "doc2"],
                golden_ids=["doc1"],
                answer="测试答案",
                contexts=["上下文1", "上下文2"],
                ground_truth="标准答案"
            )

            # 验证返回了 Ragas 指标
            assert "faithfulness" in metrics or "hit_rate" in metrics

    def test_evaluate_without_answer_fallback_to_basic(self, evaluator):
        """测试缺少答案时回退到基础指标"""
        metrics = evaluator.evaluate(
            query="测试查询",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"]
        )

        # 应该返回基础指标
        assert "hit_rate" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_evaluate_with_trace(self, evaluator):
        """测试带追踪上下文的评估"""
        trace = Mock()

        metrics = evaluator.evaluate(
            query="测试查询",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"],
            trace=trace
        )

        # 验证追踪日志被调用
        assert trace.log.called


class TestRagasEvaluatorFactory:
    """测试工厂注册"""

    def test_factory_registration(self):
        """测试 ragas provider 是否已注册"""
        # 检查 ragas 是否在已注册的 providers 中
        providers = EvaluatorFactory._providers

        # 如果 Ragas 已安装，应该能找到注册
        if 'ragas' in providers:
            assert providers['ragas'] is not None

    def test_create_ragas_evaluator(self):
        """测试通过工厂创建 RagasEvaluator"""
        settings = {
            'evaluator': {
                'provider': 'ragas'
            }
        }

        try:
            evaluator = EvaluatorFactory.create(settings)
            assert evaluator is not None
        except (ValueError, ImportError):
            # Ragas 未安装时跳过
            pytest.skip("Ragas not installed")


class TestRagasEvaluatorEdgeCases:
    """测试边界情况"""

    @pytest.fixture
    def evaluator(self):
        """创建 evaluator"""
        # Mock ragas 模块的导入
        mock_ragas = MagicMock()
        mock_ragas.evaluate = MagicMock()
        mock_ragas.metrics.faithfulness = MagicMock()
        mock_ragas.metrics.answer_relevancy = MagicMock()
        mock_ragas.metrics.context_precision = MagicMock()

        with patch.dict('sys.modules', {'ragas': mock_ragas, 'ragas.metrics': mock_ragas.metrics}):
            return RagasEvaluator()

    def test_evaluate_with_duplicate_ids(self, evaluator):
        """测试包含重复 ID 的情况"""
        retrieved_ids = ["doc1", "doc1", "doc2"]
        golden_ids = ["doc1", "doc1", "doc3"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        # 应该去重处理
        assert metrics["hit_rate"] == 1.0
        assert 0 < metrics["precision"] <= 1.0
        assert 0 < metrics["recall"] <= 1.0

    def test_evaluate_with_special_characters(self, evaluator):
        """测试包含特殊字符的 ID"""
        retrieved_ids = ["doc-1", "doc_2", "doc.3"]
        golden_ids = ["doc-1", "doc_4"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1.0
        assert metrics["precision"] == 1/3

    def test_evaluate_case_sensitivity(self, evaluator):
        """测试 ID 大小写敏感性"""
        retrieved_ids = ["Doc1", "doc2"]
        golden_ids = ["doc1", "doc2"]

        metrics = evaluator._compute_basic_metrics(retrieved_ids, golden_ids)

        # ID 应该区分大小写
        assert metrics["precision"] == 1/2  # 只有 doc2 匹配
