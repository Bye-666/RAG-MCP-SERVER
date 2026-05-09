"""
Sparse Retriever for BM25 keyword-based search.

Uses BM25 inverted index for keyword matching and retrieves full text from vector store.
"""

from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """
    Sparse retriever for BM25 keyword-based search.

    Orchestrates:
    1. Keywords → BM25 index query → top-k chunk_ids with scores
    2. Chunk_ids → VectorStore.get_by_ids() → text and metadata
    3. Merge scores with text/metadata → RetrievalResult list
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None
    ):
        """
        Initialize SparseRetriever.

        Args:
            settings: Application settings
            bm25_indexer: Optional BM25 indexer (for dependency injection)
            vector_store: Optional vector store (for dependency injection)
        """
        self.settings = settings

        # Use injected dependencies or create from settings
        if bm25_indexer is not None:
            self.bm25_indexer = bm25_indexer
        else:
            # Load BM25 index from default location
            self.bm25_indexer = BM25Indexer(settings=settings)
            self.bm25_indexer.load()

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            from src.libs.vector_store.vector_store_factory import create_vector_store
            self.vector_store = create_vector_store(settings)

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 5,
        trace: Optional[TraceContext] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks matching keywords using BM25.

        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            trace: Optional trace context

        Returns:
            List of RetrievalResult sorted by BM25 score (descending)

        Raises:
            ValueError: If keywords is empty or top_k is invalid
        """
        if not keywords:
            raise ValueError("Keywords cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Step 1: Query BM25 index
        if trace:
            stage = trace.record_stage("sparse_retriever_bm25", {
                "keyword_count": len(keywords),
                "top_k": top_k
            })

        bm25_results = self.bm25_indexer.query(keywords, top_k=top_k)

        if trace:
            trace.finish_stage(stage, {"result_count": len(bm25_results)})

        # If no results, return empty list
        if not bm25_results:
            return []

        # Step 2: Get chunk_ids and scores
        chunk_ids = [result["chunk_id"] for result in bm25_results]
        score_map = {result["chunk_id"]: result["score"] for result in bm25_results}

        # Step 3: Retrieve text and metadata from vector store
        if trace:
            stage = trace.record_stage("sparse_retriever_fetch", {
                "chunk_count": len(chunk_ids)
            })

        chunk_data = self.vector_store.get_by_ids(chunk_ids)

        if trace:
            trace.finish_stage(stage, {"fetched_count": len(chunk_data)})

        # Step 4: Merge scores with text/metadata
        # Create a map for quick lookup
        chunk_map = {item["id"]: item for item in chunk_data}

        results = []
        for chunk_id in chunk_ids:
            # Get score from BM25 results
            score = score_map.get(chunk_id, 0.0)

            # Get text and metadata from vector store
            chunk_info = chunk_map.get(chunk_id)
            if chunk_info:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=chunk_info.get("text", ""),
                    metadata=chunk_info.get("metadata", {})
                ))

        return results
