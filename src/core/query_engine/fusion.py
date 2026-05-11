"""
用于合并多个检索排名的倒数排名融合（RRF）。

RRF 公式: score(d) = Σ 1 / (k + rank(d))
其中 k 是常数（通常为 60），rank(d) 是在每个排名中的位置。
"""

from typing import List, Dict
from src.core.types import RetrievalResult


class RRFFusion:
    """
    用于合并稠密和稀疏检索结果的倒数排名融合。

    RRF 是一种简单而有效的融合多个排名列表的方法。
    它根据每个文档在每个列表中的排名为其分配分数，
    使用公式: score = Σ 1 / (k + rank)

    常数 k（默认 60）控制给予低排名项的权重。
    """

    def __init__(self, k: int = 60):
        """
        初始化 RRF 融合。

        参数:
            k: RRF 常数参数（默认：60）
               更高的 k 给予低排名项更多权重
        """
        if k < 0:
            raise ValueError("k 必须为非负数")
        self.k = k

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        使用 RRF 融合稠密和稀疏检索结果。

        参数:
            dense_results: 来自稠密检索器的结果（按分数排序）
            sparse_results: 来自稀疏检索器的结果（按分数排序）

        返回:
            按 RRF 分数排序的融合结果（降序）
        """
        # 为所有块构建 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        # 处理稠密结果
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            chunk_map[chunk_id] = result

        # 处理稀疏结果
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            # 如果块同时出现在两者中，优先使用稠密结果（稠密结果有更完整的元数据）
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # 按 RRF 分数排序（降序）
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建带有 RRF 分数的结果列表
        fused_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = chunk_map[chunk_id]
            # 创建带有 RRF 分数的新 RetrievalResult
            fused_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                score=rrf_score,
                text=result.text,
                metadata=result.metadata
            ))

        return fused_results
