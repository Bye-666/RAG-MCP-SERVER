"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval rankings.

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k is a constant (typically 60) and rank(d) is the position in each ranking.
"""

from typing import List, Dict
from src.core.types import RetrievalResult


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining dense and sparse retrieval results.

    RRF is a simple yet effective method for fusing multiple ranked lists.
    It assigns a score to each document based on its rank in each list,
    using the formula: score = Σ 1 / (k + rank)

    The constant k (default 60) controls the weight given to lower-ranked items.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: RRF constant parameter (default: 60)
               Higher k gives more weight to lower-ranked items
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Fuse dense and sparse retrieval results using RRF.

        Args:
            dense_results: Results from dense retriever (sorted by score)
            sparse_results: Results from sparse retriever (sorted by score)

        Returns:
            Fused results sorted by RRF score (descending)
        """
        # Build RRF scores for all chunks
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            chunk_map[chunk_id] = result

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            # Prefer dense result if chunk appears in both (dense has more complete metadata)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # Sort by RRF score (descending)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result list with RRF scores
        fused_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = chunk_map[chunk_id]
            # Create new RetrievalResult with RRF score
            fused_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                score=rrf_score,
                text=result.text,
                metadata=result.metadata
            ))

        return fused_results
