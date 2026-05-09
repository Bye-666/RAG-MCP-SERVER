"""
CompositeEvaluator: 组合多个评估器并行执行

支持：
- 组合多个 BaseEvaluator 实例
- 并行执行所有评估器
- 合并所有指标到单一结果字典
- 处理单个评估器失败的情况
"""

from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...core.trace import TraceContext
from ...libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """组合多个评估器的复合评估器"""

    def __init__(self, evaluators: List[BaseEvaluator], max_workers: Optional[int] = None):
        """
        初始化 CompositeEvaluator

        Args:
            evaluators: 评估器列表
            max_workers: 最大并行工作线程数（默认为评估器数量）
        """
        if not evaluators:
            raise ValueError("至少需要提供一个评估器")

        self.evaluators = evaluators
        self.max_workers = max_workers or len(evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        并行执行所有评估器并合并结果

        Args:
            query: 查询文本
            retrieved_ids: 检索到的文档 ID 列表
            golden_ids: 标准答案文档 ID 列表
            trace: 追踪上下文
            **kwargs: 传递给各个评估器的额外参数

        Returns:
            合并后的指标字典，格式：
            {
                "evaluator_name.metric_name": score,
                ...
            }
            如果评估器没有名称，使用索引作为前缀
        """
        if trace:
            trace.log("开始组合评估", {
                "evaluator_count": len(self.evaluators),
                "query": query
            })

        merged_metrics = {}

        # 使用线程池并行执行评估
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有评估任务
            future_to_evaluator = {
                executor.submit(
                    self._safe_evaluate,
                    evaluator,
                    query,
                    retrieved_ids,
                    golden_ids,
                    trace,
                    **kwargs
                ): (idx, evaluator)
                for idx, evaluator in enumerate(self.evaluators)
            }

            # 收集结果
            for future in as_completed(future_to_evaluator):
                idx, evaluator = future_to_evaluator[future]
                try:
                    metrics = future.result()

                    # 为每个指标添加评估器前缀
                    evaluator_name = self._get_evaluator_name(evaluator, idx)
                    for metric_name, score in metrics.items():
                        prefixed_name = f"{evaluator_name}.{metric_name}"
                        merged_metrics[prefixed_name] = score

                except Exception as e:
                    if trace:
                        trace.log(f"评估器 {idx} 执行失败", {"error": str(e)})
                    # 继续执行其他评估器

        if trace:
            trace.log("组合评估完成", {
                "total_metrics": len(merged_metrics)
            })

        return merged_metrics

    def _safe_evaluate(
        self,
        evaluator: BaseEvaluator,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext],
        **kwargs
    ) -> Dict[str, float]:
        """
        安全地执行单个评估器

        Args:
            evaluator: 评估器实例
            query: 查询文本
            retrieved_ids: 检索到的文档 ID 列表
            golden_ids: 标准答案文档 ID 列表
            trace: 追踪上下文
            **kwargs: 额外参数

        Returns:
            评估指标字典

        Raises:
            Exception: 评估失败时抛出异常
        """
        try:
            return evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                trace=trace,
                **kwargs
            )
        except Exception as e:
            # 重新抛出异常，由调用者处理
            raise Exception(f"评估器执行失败: {str(e)}") from e

    def _get_evaluator_name(self, evaluator: BaseEvaluator, index: int) -> str:
        """
        获取评估器名称

        Args:
            evaluator: 评估器实例
            index: 评估器索引

        Returns:
            评估器名称（类名或索引）
        """
        # 尝试获取评估器的类名
        class_name = evaluator.__class__.__name__

        # 移除常见的后缀
        if class_name.endswith("Evaluator"):
            class_name = class_name[:-9]  # 移除 "Evaluator"

        # 转换为小写
        return class_name.lower() or f"evaluator_{index}"


def create_composite_evaluator(
    settings: dict,
    evaluator_factory
) -> CompositeEvaluator:
    """
    根据配置创建 CompositeEvaluator

    Args:
        settings: 配置字典，期望包含 evaluation.backends 列表
        evaluator_factory: EvaluatorFactory 类或实例

    Returns:
        CompositeEvaluator 实例

    Example:
        settings = {
            'evaluation': {
                'backends': ['ragas', 'custom']
            }
        }
        composite = create_composite_evaluator(settings, EvaluatorFactory)
    """
    backends = settings.get('evaluation', {}).get('backends', ['custom'])

    if not backends:
        raise ValueError("evaluation.backends 不能为空")

    evaluators = []
    for backend in backends:
        try:
            # 为每个 backend 创建配置
            backend_settings = {
                'evaluator': {
                    'provider': backend
                }
            }
            evaluator = evaluator_factory.create(backend_settings)
            evaluators.append(evaluator)
        except Exception as e:
            # 跳过无法创建的评估器
            print(f"警告: 无法创建评估器 '{backend}': {str(e)}")

    if not evaluators:
        raise ValueError("没有成功创建任何评估器")

    return CompositeEvaluator(evaluators)
