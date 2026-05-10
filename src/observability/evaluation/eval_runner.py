"""
EvalRunner: 评估运行器

功能：
- 读取黄金测试集（golden test set）
- 对每个测试用例执行检索
- 使用评估器计算指标
- 生成评估报告
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from ...core.trace import TraceContext
from ...libs.evaluator.base_evaluator import BaseEvaluator


@dataclass
class TestCase:
    """单个测试用例"""
    query: str
    expected_chunk_ids: List[str]
    expected_sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """单个测试用例的结果"""
    query: str
    retrieved_ids: List[str]
    expected_ids: List[str]
    metrics: Dict[str, float]
    success: bool
    error: Optional[str] = None


@dataclass
class EvalReport:
    """评估报告"""
    total_cases: int
    successful_cases: int
    failed_cases: int
    aggregate_metrics: Dict[str, float]
    test_results: List[TestResult]

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class EvalRunner:
    """评估运行器"""

    def __init__(
        self,
        settings: dict,
        hybrid_search,
        evaluator: BaseEvaluator,
        trace_context: Optional[TraceContext] = None
    ):
        """
        初始化 EvalRunner

        Args:
            settings: 配置字典
            hybrid_search: HybridSearch 实例（用于执行检索）
            evaluator: 评估器实例
            trace_context: 追踪上下文
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.trace_context = trace_context

    def load_test_set(self, test_set_path: str) -> List[TestCase]:
        """
        加载黄金测试集

        Args:
            test_set_path: 测试集文件路径

        Returns:
            测试用例列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: JSON 格式错误
        """
        path = Path(test_set_path)
        if not path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {test_set_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"测试集 JSON 格式错误: {str(e)}")

        if "test_cases" not in data:
            raise ValueError("测试集缺少 'test_cases' 字段")

        test_cases = []
        for case_data in data["test_cases"]:
            test_case = TestCase(
                query=case_data["query"],
                expected_chunk_ids=case_data["expected_chunk_ids"],
                expected_sources=case_data.get("expected_sources"),
                metadata=case_data.get("metadata")
            )
            test_cases.append(test_case)

        return test_cases

    def run(self, test_set_path: str, top_k: Optional[int] = None) -> EvalReport:
        """
        运行评估

        Args:
            test_set_path: 测试集文件路径
            top_k: 检索返回的文档数量（覆盖配置）

        Returns:
            评估报告
        """
        if self.trace_context:
            stage = self.trace_context.record_stage("evaluation_start", {"test_set": test_set_path})
            self.trace_context.finish_stage(stage)

        # 加载测试集
        test_cases = self.load_test_set(test_set_path)

        if self.trace_context:
            stage = self.trace_context.record_stage("load_test_set", {"total_cases": len(test_cases)})
            self.trace_context.finish_stage(stage)

        # 运行每个测试用例
        test_results = []
        for idx, test_case in enumerate(test_cases):
            result = self._run_single_test(test_case, top_k, idx)
            test_results.append(result)

        # 计算聚合指标
        aggregate_metrics = self._compute_aggregate_metrics(test_results)

        # 统计成功/失败数量
        successful_cases = sum(1 for r in test_results if r.success)
        failed_cases = len(test_results) - successful_cases

        report = EvalReport(
            total_cases=len(test_cases),
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            aggregate_metrics=aggregate_metrics,
            test_results=test_results
        )

        if self.trace_context:
            stage = self.trace_context.record_stage("evaluation_complete", {
                "total": report.total_cases,
                "success": report.successful_cases,
                "failed": report.failed_cases
            })
            self.trace_context.finish_stage(stage)

        return report

    def _run_single_test(
        self,
        test_case: TestCase,
        top_k: Optional[int],
        index: int
    ) -> TestResult:
        """
        运行单个测试用例

        Args:
            test_case: 测试用例
            top_k: 检索返回的文档数量
            index: 测试用例索引

        Returns:
            测试结果
        """
        try:
            # 执行检索
            k = top_k or self.settings.get('retrieval', {}).get('top_k', 5)

            search_results = self.hybrid_search.search(
                query=test_case.query,
                top_k=k,
                trace=self.trace_context
            )

            # 提取检索到的文档 ID
            retrieved_ids = [result.chunk_id for result in search_results]

            # 使用评估器计算指标
            metrics = self.evaluator.evaluate(
                query=test_case.query,
                retrieved_ids=retrieved_ids,
                golden_ids=test_case.expected_chunk_ids,
                trace=self.trace_context
            )

            return TestResult(
                query=test_case.query,
                retrieved_ids=retrieved_ids,
                expected_ids=test_case.expected_chunk_ids,
                metrics=metrics,
                success=True
            )

        except Exception as e:
            if self.trace_context:
                stage = self.trace_context.record_stage(f"test_case_{index}_failed", {"error": str(e)})
                self.trace_context.finish_stage(stage)

            return TestResult(
                query=test_case.query,
                retrieved_ids=[],
                expected_ids=test_case.expected_chunk_ids,
                metrics={},
                success=False,
                error=str(e)
            )

    def _compute_aggregate_metrics(self, test_results: List[TestResult]) -> Dict[str, float]:
        """
        计算聚合指标

        Args:
            test_results: 测试结果列表

        Returns:
            聚合指标字典
        """
        if not test_results:
            return {}

        # 收集所有成功的测试结果
        successful_results = [r for r in test_results if r.success]

        if not successful_results:
            return {}

        # 收集所有指标名称
        all_metric_names = set()
        for result in successful_results:
            all_metric_names.update(result.metrics.keys())

        # 计算每个指标的平均值
        aggregate_metrics = {}
        for metric_name in all_metric_names:
            values = [
                result.metrics[metric_name]
                for result in successful_results
                if metric_name in result.metrics
            ]
            if values:
                aggregate_metrics[metric_name] = sum(values) / len(values)

        # 计算 MRR (Mean Reciprocal Rank)
        mrr_values = []
        for result in successful_results:
            retrieved_set = result.retrieved_ids
            expected_set = set(result.expected_ids)

            # 找到第一个相关文档的位置
            for rank, doc_id in enumerate(retrieved_set, start=1):
                if doc_id in expected_set:
                    mrr_values.append(1.0 / rank)
                    break
            else:
                # 没有找到相关文档
                mrr_values.append(0.0)

        if mrr_values:
            aggregate_metrics["mrr"] = sum(mrr_values) / len(mrr_values)

        return aggregate_metrics
