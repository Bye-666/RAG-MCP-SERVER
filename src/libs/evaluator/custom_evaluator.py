from .base_evaluator import BaseEvaluator
from typing import List


class CustomEvaluator(BaseEvaluator):
    """Lightweight evaluator supporting hit_rate and MRR metrics"""

    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):
        metrics = {}

        # Hit Rate@K: whether any golden id appears in retrieved results
        metrics['hit_rate'] = 1.0 if any(gid in retrieved_ids for gid in golden_ids) else 0.0

        # MRR (Mean Reciprocal Rank): rank of first relevant result
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in golden_ids:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr

        return metrics