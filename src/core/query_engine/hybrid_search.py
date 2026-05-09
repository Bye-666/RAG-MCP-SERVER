"""
Hybrid Search Engine combining dense and sparse retrieval.

Orchestrates QueryProcessor, DenseRetriever, SparseRetriever, and RRF Fusion
for comprehensive semantic + keyword search.
"""

from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion


class HybridSearch:
    """
    Hybrid search engine combining dense and sparse retrieval.

    Pipeline:
    1. QueryProcessor: extract keywords and parse filters
    2. Parallel retrieval:
       - DenseRetriever: semantic vector search
       - SparseRetriever: BM25 keyword search
    3. RRF Fusion: combine rankings
    4. Metadata filtering: post-filter by metadata
    5. Top-K selection
    """

    def __init__(
        self,
        settings: Settings,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None
    ):
        """
        Initialize HybridSearch.

        Args:
            settings: Application settings
            query_processor: Optional query processor (for dependency injection)
            dense_retriever: Optional dense retriever (for dependency injection)
            sparse_retriever: Optional sparse retriever (for dependency injection)
            fusion: Optional RRF fusion (for dependency injection)
        """
        self.settings = settings

        # Use injected dependencies or create defaults
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self.fusion = fusion or RRFFusion(k=60)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        Args:
            query: User query string
            top_k: Number of results to return
            filters: Optional metadata filters
            trace: Optional trace context

        Returns:
            List of top-k RetrievalResult sorted by relevance

        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Step 1: Process query
        if trace:
            stage = trace.record_stage("hybrid_search_process_query", {
                "query_length": len(query)
            })

        processed_query = self.query_processor.process(query, filters=filters)

        if trace:
            trace.finish_stage(stage, {
                "keyword_count": len(processed_query.keywords),
                "has_filters": bool(processed_query.filters)
            })

        # Step 2: Parallel retrieval (with fallback)
        dense_results = []
        sparse_results = []

        # Retrieve more candidates for fusion (2x top_k)
        candidate_k = top_k * 2

        # Dense retrieval
        if trace:
            stage = trace.record_stage("hybrid_search_dense_retrieve", {
                "top_k": candidate_k
            })

        try:
            dense_results = self.dense_retriever.retrieve(
                query=query,
                top_k=candidate_k,
                filters=processed_query.filters,
                trace=trace
            )
            if trace:
                trace.finish_stage(stage, {
                    "result_count": len(dense_results),
                    "success": True
                })
        except Exception as e:
            if trace:
                trace.finish_stage(stage, {
                    "result_count": 0,
                    "success": False,
                    "error": str(e)
                })
            # Continue with sparse only

        # Sparse retrieval
        if trace:
            stage = trace.record_stage("hybrid_search_sparse_retrieve", {
                "keyword_count": len(processed_query.keywords),
                "top_k": candidate_k
            })

        try:
            if processed_query.keywords:
                sparse_results = self.sparse_retriever.retrieve(
                    keywords=processed_query.keywords,
                    top_k=candidate_k,
                    trace=trace
                )
            if trace:
                trace.finish_stage(stage, {
                    "result_count": len(sparse_results),
                    "success": True
                })
        except Exception as e:
            if trace:
                trace.finish_stage(stage, {
                    "result_count": 0,
                    "success": False,
                    "error": str(e)
                })
            # Continue with dense only

        # If both failed, return empty
        if not dense_results and not sparse_results:
            return []

        # Step 3: Fusion
        if trace:
            stage = trace.record_stage("hybrid_search_fusion", {
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results)
            })

        fused_results = self.fusion.fuse(dense_results, sparse_results)

        if trace:
            trace.finish_stage(stage, {"fused_count": len(fused_results)})

        # Step 4: Apply metadata filters (post-filter fallback)
        if filters:
            if trace:
                stage = trace.record_stage("hybrid_search_metadata_filter", {
                    "filter_count": len(filters)
                })

            fused_results = self._apply_metadata_filters(fused_results, filters)

            if trace:
                trace.finish_stage(stage, {"filtered_count": len(fused_results)})

        # Step 5: Top-K selection
        results = fused_results[:top_k]

        return results

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """
        Apply metadata filters to candidate results (post-filter fallback).

        This is a fallback mechanism for when vector store filters don't work
        or when additional filtering is needed.

        Args:
            candidates: List of candidate results
            filters: Metadata filters to apply

        Returns:
            Filtered list of results
        """
        if not filters:
            return candidates

        filtered = []
        for result in candidates:
            # Check if all filter conditions match
            match = True
            for key, value in filters.items():
                metadata_value = result.metadata.get(key)
                if metadata_value != value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered
