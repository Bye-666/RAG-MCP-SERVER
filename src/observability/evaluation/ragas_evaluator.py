"""
RagasEvaluator: 封装 Ragas 框架的评估器实现

支持的指标：
- Faithfulness: 答案对上下文的忠实度
- Answer Relevancy: 答案与问题的相关性
- Context Precision: 上下文的精确度
"""

from typing import Dict, List, Optional
from ...core.trace import TraceContext
from ...libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    """基于 Ragas 框架的评估器实现"""

    def __init__(self, llm_provider: Optional[str] = None, embedding_provider: Optional[str] = None):
        """
        初始化 RagasEvaluator

        Args:
            llm_provider: LLM 提供商（用于 Faithfulness 等指标）
            embedding_provider: Embedding 提供商（用于 Answer Relevancy 等指标）
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )
            self._evaluate = evaluate
            self._faithfulness = faithfulness
            self._answer_relevancy = answer_relevancy
            self._context_precision = context_precision
        except ImportError as e:
            raise ImportError(
                "Ragas 未安装。请运行: pip install ragas\n"
                "或在 requirements.txt 中添加 ragas 依赖"
            ) from e

        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        使用 Ragas 评估检索结果

        Args:
            query: 查询文本
            retrieved_ids: 检索到的文档 ID 列表
            golden_ids: 标准答案文档 ID 列表
            trace: 追踪上下文
            **kwargs: 额外参数，支持：
                - answer: 生成的答案文本（用于 faithfulness 和 answer_relevancy）
                - contexts: 检索到的上下文文本列表（用于 context_precision）
                - ground_truth: 标准答案文本（用于 answer_relevancy）

        Returns:
            包含各项指标的字典，例如：
            {
                "faithfulness": 0.85,
                "answer_relevancy": 0.92,
                "context_precision": 0.78
            }
        """
        if trace:
            trace.log("开始 Ragas 评估", {
                "query": query,
                "retrieved_count": len(retrieved_ids),
                "golden_count": len(golden_ids)
            })

        # 提取额外参数
        answer = kwargs.get("answer", "")
        contexts = kwargs.get("contexts", [])
        ground_truth = kwargs.get("ground_truth", "")

        # 如果没有提供必要的参数，返回基础指标
        if not answer or not contexts:
            return self._compute_basic_metrics(retrieved_ids, golden_ids, trace)

        # 构建 Ragas 评估数据集
        try:
            from datasets import Dataset

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth] if ground_truth else [""]
            }

            dataset = Dataset.from_dict(data)

            # 选择要评估的指标
            metrics = []
            if answer and contexts:
                metrics.append(self._faithfulness)
            if answer and ground_truth:
                metrics.append(self._answer_relevancy)
            if contexts and golden_ids:
                metrics.append(self._context_precision)

            if not metrics:
                return self._compute_basic_metrics(retrieved_ids, golden_ids, trace)

            # 执行评估
            result = self._evaluate(dataset, metrics=metrics)

            # 提取指标
            metrics_dict = {}
            if "faithfulness" in result:
                metrics_dict["faithfulness"] = float(result["faithfulness"])
            if "answer_relevancy" in result:
                metrics_dict["answer_relevancy"] = float(result["answer_relevancy"])
            if "context_precision" in result:
                metrics_dict["context_precision"] = float(result["context_precision"])

            if trace:
                trace.log("Ragas 评估完成", metrics_dict)

            return metrics_dict

        except Exception as e:
            if trace:
                trace.log("Ragas 评估失败，回退到基础指标", {"error": str(e)})

            # 评估失败时回退到基础指标
            return self._compute_basic_metrics(retrieved_ids, golden_ids, trace)

    def _compute_basic_metrics(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext] = None
    ) -> Dict[str, float]:
        """
        计算基础检索指标（不依赖 Ragas）

        Args:
            retrieved_ids: 检索到的文档 ID 列表
            golden_ids: 标准答案文档 ID 列表
            trace: 追踪上下文

        Returns:
            包含基础指标的字典
        """
        if not golden_ids:
            return {
                "hit_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }

        # 计算命中数
        retrieved_set = set(retrieved_ids)
        golden_set = set(golden_ids)
        hits = len(retrieved_set & golden_set)

        # Hit Rate: 是否至少命中一个相关文档
        hit_rate = 1.0 if hits > 0 else 0.0

        # Precision: 检索结果中相关文档的比例
        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0

        # Recall: 相关文档中被检索到的比例
        recall = hits / len(golden_ids) if golden_ids else 0.0

        metrics = {
            "hit_rate": hit_rate,
            "precision": precision,
            "recall": recall
        }

        if trace:
            trace.log("基础指标计算完成", metrics)

        return metrics
