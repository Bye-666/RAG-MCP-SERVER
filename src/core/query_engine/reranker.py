"""
Reranker for refining retrieval results.

Wraps libs.reranker backend with fallback mechanism for robustness.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker


@dataclass
class RerankResult:
    """Result from reranking operation

    Attributes:
        results: Reranked list of RetrievalResult
        fallback: True if reranking failed and original order was used
        error: Error message if fallback occurred
    """
    results: List[RetrievalResult]
    fallback: bool = False
    error: Optional[str] = None


class Reranker:
    """
    Reranker for refining retrieval results.

    Wraps a reranker backend (None/CrossEncoder/LLM) with fallback mechanism.
    If reranking fails or times out, returns original ranking with fallback flag.
    """

    def __init__(
        self,
        settings: Settings,
        reranker_backend: Optional[BaseReranker] = None
    ):
        """
        Initialize Reranker.

        Args:
            settings: Application settings
            reranker_backend: Optional reranker backend (for dependency injection)
        """
        self.settings = settings

        # Use injected backend or create from settings
        if reranker_backend is not None:
            self.backend = reranker_backend
        else:
            from src.libs.reranker.reranker_factory import RerankerFactory
            self.backend = RerankerFactory.create(settings.__dict__)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        trace: Optional[TraceContext] = None
    ) -> RerankResult:
        """
        Rerank candidates based on relevance to query.

        Args:
            query: User query string
            candidates: List of candidate results to rerank
            trace: Optional trace context

        Returns:
            RerankResult with reranked results and fallback status

        Raises:
            ValueError: If query is empty or candidates is None
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if candidates is None:
            raise ValueError("Candidates cannot be None")

        # Empty candidates - return as-is
        if not candidates:
            return RerankResult(results=[], fallback=False)

        # Convert RetrievalResult to dict format for backend
        candidate_dicts = [
            {
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in candidates
        ]

        # Attempt reranking with fallback
        if trace:
            stage = trace.record_stage("reranker", {
                "candidate_count": len(candidates),
                "backend": self.backend.__class__.__name__
            })

        try:
            # Call backend reranker
            reranked_dicts = self.backend.rerank(
                query=query,
                candidates=candidate_dicts,
                trace=trace
            )

            # Convert back to RetrievalResult
            reranked_results = []
            for item in reranked_dicts:
                reranked_results.append(RetrievalResult(
                    chunk_id=item["id"],
                    score=item["score"],
                    text=item["text"],
                    metadata=item.get("metadata", {})
                ))

            if trace:
                trace.finish_stage(stage, {
                    "success": True,
                    "fallback": False
                })

            return RerankResult(
                results=reranked_results,
                fallback=False
            )

        except Exception as e:
            # Fallback: return original ranking
            if trace:
                trace.finish_stage(stage, {
                    "success": False,
                    "fallback": True,
                    "error": str(e)
                })

            return RerankResult(
                results=candidates,
                fallback=True,
                error=str(e)
            )
