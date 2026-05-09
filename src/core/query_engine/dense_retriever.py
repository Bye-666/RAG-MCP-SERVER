"""
Dense Retriever for semantic vector search.

Combines embedding generation and vector store query for semantic retrieval.
"""

from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """
    Dense retriever for semantic vector search.

    Orchestrates:
    1. Query text → embedding vector (via EmbeddingClient)
    2. Vector → top-k similar chunks (via VectorStore)
    3. Results → RetrievalResult list
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None
    ):
        """
        Initialize DenseRetriever.

        Args:
            settings: Application settings
            embedding_client: Optional embedding client (for dependency injection)
            vector_store: Optional vector store (for dependency injection)
        """
        self.settings = settings

        # Use injected dependencies or create from settings
        if embedding_client is not None:
            self.embedding_client = embedding_client
        else:
            from src.libs.embedding.embedding_factory import create_embedding_client
            self.embedding_client = create_embedding_client(settings)

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            from src.libs.vector_store.vector_store_factory import create_vector_store
            self.vector_store = create_vector_store(settings)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k semantically similar chunks for query.

        Args:
            query: User query string
            top_k: Number of results to return
            filters: Optional metadata filters
            trace: Optional trace context

        Returns:
            List of RetrievalResult sorted by relevance score (descending)

        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Step 1: Generate query embedding
        if trace:
            stage = trace.record_stage("dense_retriever_embed", {"query_length": len(query)})

        embeddings = self.embedding_client.embed([query], trace=trace)
        query_vector = embeddings[0]

        if trace:
            trace.finish_stage(stage, {"vector_dim": len(query_vector)})

        # Step 2: Query vector store
        if trace:
            stage = trace.record_stage("dense_retriever_query", {
                "top_k": top_k,
                "has_filters": filters is not None
            })

        raw_results = self.vector_store.query(
            vector=query_vector,
            top_k=top_k,
            filters=filters,
            trace=trace
        )

        if trace:
            trace.finish_stage(stage, {"result_count": len(raw_results)})

        # Step 3: Convert to RetrievalResult
        results = []
        for item in raw_results:
            results.append(RetrievalResult(
                chunk_id=item["id"],
                score=item["score"],
                text=item["text"],
                metadata=item.get("metadata", {})
            ))

        return results
