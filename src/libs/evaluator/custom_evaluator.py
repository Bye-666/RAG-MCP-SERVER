from .base_evaluator import BaseEvaluator
from typing import List


class CustomEvaluator(BaseEvaluator):
    """支持 hit_rate 和 MRR 指标的轻量级评估器"""

    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):
        metrics = {}

        # Hit Rate@K: 检索结果中是否出现任何黄金 id
        metrics['hit_rate'] = 1.0 if any(gid in retrieved_ids for gid in golden_ids) else 0.0

        # MRR（平均倒数排名）: 第一个相关结果的排名
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in golden_ids:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr

        return metrics