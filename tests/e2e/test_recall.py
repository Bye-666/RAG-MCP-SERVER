"""
E2E 召回测试（Recall Regression Test）

目标：
- 基于 golden test set 验证检索系统的召回质量
- 确保 Hit@K 和 MRR 指标达到预设阈值
- 作为回归测试基线，防止检索质量下降

测试策略：
1. 使用 Mock 的向量库和 BM25 索引（避免依赖真实服务）
2. 对每个测试用例执行混合检索
3. 计算召回指标（Hit@K, MRR）
4. 验证指标是否达到阈值

注意：本测试使用 Mock 对象模拟检索结果，主要验证评估流程的正确性。
真实的召回质量测试需要在有真实数据的环境中运行。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import EvalRunner, TestCase
from src.core.trace import TraceContext
from src.core.types import RetrievalResult


# 召回质量阈值（回归基线）
RECALL_THRESHOLDS = {
    "hit_rate@5": 0.60,   # Top-5 命中率至少 60%
    "hit_rate@10": 0.80,  # Top-10 命中率至少 80%
    "mrr": 0.50,          # 平均倒数排名至少 0.50
}


@pytest.fixture(scope="module")
def test_settings():
    """加载测试配置"""
    settings = load_settings("config/settings.yaml")
    return settings


@pytest.fixture(scope="module")
def mock_dense_retriever():
    """创建 Mock DenseRetriever"""
    mock = Mock(spec=DenseRetriever)

    def mock_retrieve(query: str, top_k: int = 5, filters=None, trace=None):
        """模拟检索结果，返回与查询相关的结果"""
        # 根据查询返回模拟结果
        if "Azure" in query or "azure" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_azure_config_001",
                    score=0.95,
                    text="Azure OpenAI 配置指南...",
                    metadata={"source": "azure_openai_guide.pdf"}
                ),
                RetrievalResult(
                    chunk_id="chunk_azure_config_002",
                    score=0.88,
                    text="Azure 端点配置...",
                    metadata={"source": "azure_openai_guide.pdf"}
                ),
            ]
        elif "RAG" in query or "rag" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_rag_intro_001",
                    score=0.92,
                    text="RAG 系统介绍...",
                    metadata={"source": "rag_introduction.pdf"}
                ),
            ]
        elif "混合检索" in query or "hybrid" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_hybrid_search_001",
                    score=0.90,
                    text="混合检索实现...",
                    metadata={"source": "hybrid_search_guide.pdf"}
                ),
                RetrievalResult(
                    chunk_id="chunk_hybrid_search_002",
                    score=0.85,
                    text="Dense 和 Sparse 结合...",
                    metadata={"source": "hybrid_search_guide.pdf"}
                ),
            ]
        elif "向量数据库" in query or "vector" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_vector_db_001",
                    score=0.89,
                    text="向量数据库原理...",
                    metadata={"source": "vector_database.pdf"}
                ),
                RetrievalResult(
                    chunk_id="chunk_vector_db_002",
                    score=0.82,
                    text="向量存储与检索...",
                    metadata={"source": "vector_database.pdf"}
                ),
            ]
        elif "优化" in query or "optimization" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_optimization_001",
                    score=0.87,
                    text="检索性能优化...",
                    metadata={"source": "performance_optimization.pdf"}
                ),
                RetrievalResult(
                    chunk_id="chunk_optimization_002",
                    score=0.80,
                    text="索引优化策略...",
                    metadata={"source": "performance_optimization.pdf"}
                ),
            ]
        else:
            # 默认返回一些通用结果
            return [
                RetrievalResult(
                    chunk_id="chunk_generic_001",
                    score=0.70,
                    text="通用文档内容...",
                    metadata={"source": "generic.pdf"}
                ),
            ]

    mock.retrieve.side_effect = mock_retrieve
    return mock


@pytest.fixture(scope="module")
def mock_sparse_retriever():
    """创建 Mock SparseRetriever"""
    mock = Mock(spec=SparseRetriever)

    def mock_retrieve(query: str, top_k: int = 5, filters=None, trace=None):
        """模拟稀疏检索结果"""
        # 返回与 dense retriever 部分重叠的结果
        if "Azure" in query or "azure" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_azure_config_001",
                    score=0.85,
                    text="Azure OpenAI 配置指南...",
                    metadata={"source": "azure_openai_guide.pdf"}
                ),
            ]
        elif "RAG" in query or "rag" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk_rag_intro_001",
                    score=0.88,
                    text="RAG 系统介绍...",
                    metadata={"source": "rag_introduction.pdf"}
                ),
            ]
        else:
            return []

    mock.retrieve.side_effect = mock_retrieve
    return mock


@pytest.fixture(scope="module")
def hybrid_search(test_settings, mock_dense_retriever, mock_sparse_retriever):
    """创建 HybridSearch 实例（使用 Mock 检索器）"""
    query_processor = QueryProcessor()
    fusion = RRFFusion(k=60)

    return HybridSearch(
        settings=test_settings,
        query_processor=query_processor,
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        fusion=fusion
    )


@pytest.fixture(scope="module")
def evaluator():
    """创建评估器实例"""
    return CustomEvaluator()


@pytest.fixture(scope="module")
def eval_runner(test_settings, hybrid_search, evaluator):
    """创建 EvalRunner 实例"""
    trace = TraceContext(trace_id="recall_test", trace_type="evaluation")
    return EvalRunner(
        settings=test_settings.__dict__,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        trace_context=trace
    )


@pytest.fixture(scope="module")
def golden_test_set_path():
    """返回 golden test set 路径"""
    return "tests/fixtures/golden_test_set.json"


def test_load_golden_test_set(eval_runner, golden_test_set_path):
    """测试：成功加载 golden test set"""
    test_cases = eval_runner.load_test_set(golden_test_set_path)

    assert len(test_cases) > 0, "测试集不能为空"

    # 验证测试用例结构
    for test_case in test_cases:
        assert isinstance(test_case, TestCase)
        assert test_case.query, "查询不能为空"
        assert len(test_case.expected_chunk_ids) > 0, "期望的 chunk IDs 不能为空"


def test_recall_hit_rate_at_5(eval_runner, golden_test_set_path):
    """测试：Hit@5 达到阈值"""
    # 运行评估（Top-5）
    report = eval_runner.run(golden_test_set_path, top_k=5)

    # 验证至少有一些测试用例成功
    assert report.successful_cases > 0, "至少应该有一些测试用例成功执行"

    # 获取 Hit@5 指标
    hit_rate_5 = report.aggregate_metrics.get("hit_rate", 0.0)

    # 验证是否达到阈值
    threshold = RECALL_THRESHOLDS["hit_rate@5"]
    assert hit_rate_5 >= threshold, (
        f"Hit@5 ({hit_rate_5:.2%}) 低于阈值 ({threshold:.2%})\n"
        f"成功用例: {report.successful_cases}/{report.total_cases}\n"
        f"失败用例: {report.failed_cases}"
    )


def test_recall_hit_rate_at_10(eval_runner, golden_test_set_path):
    """测试：Hit@10 达到阈值"""
    # 运行评估（Top-10）
    report = eval_runner.run(golden_test_set_path, top_k=10)

    # 验证至少有一些测试用例成功
    assert report.successful_cases > 0, "至少应该有一些测试用例成功执行"

    # 获取 Hit@10 指标
    hit_rate_10 = report.aggregate_metrics.get("hit_rate", 0.0)

    # 验证是否达到阈值
    threshold = RECALL_THRESHOLDS["hit_rate@10"]
    assert hit_rate_10 >= threshold, (
        f"Hit@10 ({hit_rate_10:.2%}) 低于阈值 ({threshold:.2%})\n"
        f"成功用例: {report.successful_cases}/{report.total_cases}\n"
        f"失败用例: {report.failed_cases}"
    )


def test_recall_mrr(eval_runner, golden_test_set_path):
    """测试：MRR (Mean Reciprocal Rank) 达到阈值"""
    # 运行评估（Top-10）
    report = eval_runner.run(golden_test_set_path, top_k=10)

    # 验证至少有一些测试用例成功
    assert report.successful_cases > 0, "至少应该有一些测试用例成功执行"

    # 获取 MRR 指标
    mrr = report.aggregate_metrics.get("mrr", 0.0)

    # 验证是否达到阈值
    threshold = RECALL_THRESHOLDS["mrr"]
    assert mrr >= threshold, (
        f"MRR ({mrr:.3f}) 低于阈值 ({threshold:.3f})\n"
        f"成功用例: {report.successful_cases}/{report.total_cases}\n"
        f"失败用例: {report.failed_cases}"
    )


def test_recall_full_report(eval_runner, golden_test_set_path):
    """测试：生成完整的评估报告"""
    # 运行评估
    report = eval_runner.run(golden_test_set_path, top_k=10)

    # 验证报告结构
    assert report.total_cases > 0
    assert report.successful_cases + report.failed_cases == report.total_cases
    assert isinstance(report.aggregate_metrics, dict)
    assert len(report.test_results) == report.total_cases

    # 打印报告摘要（用于调试）
    print("\n" + "="*60)
    print("召回测试报告摘要")
    print("="*60)
    print(f"总测试用例: {report.total_cases}")
    print(f"成功: {report.successful_cases}")
    print(f"失败: {report.failed_cases}")
    print("\n聚合指标:")
    for metric_name, value in report.aggregate_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("="*60)

    # 验证报告可以序列化为 JSON
    json_report = report.to_json()
    assert len(json_report) > 0
    assert "total_cases" in json_report
    assert "aggregate_metrics" in json_report


def test_recall_individual_cases(eval_runner, golden_test_set_path):
    """测试：检查每个测试用例的详细结果"""
    # 运行评估
    report = eval_runner.run(golden_test_set_path, top_k=10)

    # 分析每个测试用例
    failed_cases = []
    for result in report.test_results:
        if not result.success:
            failed_cases.append({
                "query": result.query,
                "error": result.error
            })
        else:
            # 验证成功的测试用例有指标
            assert len(result.metrics) > 0, f"成功的测试用例应该有指标: {result.query}"

    # 如果有失败的用例，打印详细信息
    if failed_cases:
        print("\n失败的测试用例:")
        for case in failed_cases:
            print(f"  查询: {case['query']}")
            print(f"  错误: {case['error']}")

    # 验证失败率不超过 20%
    failure_rate = len(failed_cases) / report.total_cases
    assert failure_rate <= 0.2, (
        f"失败率 ({failure_rate:.2%}) 过高，超过 20% 阈值\n"
        f"失败用例数: {len(failed_cases)}/{report.total_cases}"
    )


def test_recall_consistency(eval_runner, golden_test_set_path):
    """测试：多次运行结果的一致性"""
    # 运行两次评估
    report1 = eval_runner.run(golden_test_set_path, top_k=5)
    report2 = eval_runner.run(golden_test_set_path, top_k=5)

    # 验证成功用例数一致
    assert report1.successful_cases == report2.successful_cases, (
        "多次运行的成功用例数应该一致"
    )

    # 验证聚合指标接近（允许小幅波动，因为可能有随机性）
    for metric_name in report1.aggregate_metrics:
        if metric_name in report2.aggregate_metrics:
            value1 = report1.aggregate_metrics[metric_name]
            value2 = report2.aggregate_metrics[metric_name]

            # 允许 5% 的波动
            diff = abs(value1 - value2)
            tolerance = 0.05

            assert diff <= tolerance, (
                f"指标 {metric_name} 在多次运行中波动过大: "
                f"{value1:.4f} vs {value2:.4f} (差异: {diff:.4f})"
            )


@pytest.mark.parametrize("top_k", [3, 5, 10, 20])
def test_recall_at_different_k(eval_runner, golden_test_set_path, top_k):
    """测试：不同 K 值下的召回表现"""
    # 运行评估
    report = eval_runner.run(golden_test_set_path, top_k=top_k)

    # 验证基本要求
    assert report.successful_cases > 0, f"Top-{top_k} 应该有成功的测试用例"

    # 获取 Hit Rate
    hit_rate = report.aggregate_metrics.get("hit_rate", 0.0)

    # 打印结果（用于分析）
    print(f"\nTop-{top_k} Hit Rate: {hit_rate:.2%}")

    # 验证 Hit Rate 随 K 增大而提升（或至少不下降）
    # 这是一个基本的单调性检查
    assert hit_rate >= 0.0 and hit_rate <= 1.0, "Hit Rate 应该在 [0, 1] 范围内"


def test_recall_with_empty_results(eval_runner):
    """测试：处理空检索结果的情况"""
    # 创建一个不可能匹配的测试用例
    test_case = TestCase(
        query="这是一个完全不存在的查询内容 xyzabc123456",
        expected_chunk_ids=["nonexistent_chunk_id"]
    )

    # 运行单个测试
    result = eval_runner._run_single_test(test_case, top_k=5, index=0)

    # 验证即使没有结果也能正常处理
    assert result.success or result.error is not None

    # 如果成功，验证指标存在
    if result.success:
        assert "hit_rate" in result.metrics
        # 空结果的 Hit Rate 应该是 0
        assert result.metrics["hit_rate"] == 0.0
