"""Query knowledge hub tool implementation."""

from typing import Dict, Any, Optional
from ...core.config import Settings
from ...core.query_engine.hybrid_search import HybridSearch
from ...core.query_engine.reranker import Reranker
from ...core.response.response_builder import ResponseBuilder


def get_tool_schema() -> Dict[str, Any]:
    """Get the tool schema for query_knowledge_hub."""
    return {
        "name": "query_knowledge_hub",
        "description": "Search the knowledge base using hybrid retrieval (dense + sparse + reranking)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)",
                    "default": 10
                },
                "collection": {
                    "type": "string",
                    "description": "Optional collection name to filter results"
                }
            },
            "required": ["query"]
        }
    }


class QueryKnowledgeHub:
    """Query knowledge hub tool handler."""

    def __init__(
        self,
        settings: Settings,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[Reranker] = None,
        response_builder: Optional[ResponseBuilder] = None
    ):
        """
        Initialize query knowledge hub tool.

        Args:
            settings: Application settings
            hybrid_search: HybridSearch instance (creates default if None)
            reranker: Reranker instance (creates default if None)
            response_builder: ResponseBuilder instance (creates default if None)
        """
        self.settings = settings
        self.hybrid_search = hybrid_search or self._create_hybrid_search()
        self.reranker = reranker or Reranker(settings)
        self.response_builder = response_builder or ResponseBuilder()

    def _create_hybrid_search(self) -> HybridSearch:
        """Create HybridSearch instance with all dependencies."""
        from ...core.query_engine.query_processor import QueryProcessor
        from ...core.query_engine.dense_retriever import DenseRetriever
        from ...core.query_engine.sparse_retriever import SparseRetriever
        from ...core.query_engine.fusion import RRFFusion
        from ...libs.embedding import EmbeddingFactory
        from ...ingestion.storage.vector_store import VectorStoreFactory
        from ...ingestion.storage.bm25_indexer import BM25Indexer

        # Initialize components
        query_processor = QueryProcessor(self.settings)

        embedding_client = EmbeddingFactory.create(self.settings.model_dump())
        vector_store = VectorStoreFactory.create(self.settings.model_dump())
        dense_retriever = DenseRetriever(self.settings, embedding_client, vector_store)

        bm25_indexer = BM25Indexer(self.settings)
        sparse_retriever = SparseRetriever(self.settings, bm25_indexer, vector_store)

        fusion = RRFFusion(self.settings)

        return HybridSearch(
            self.settings,
            query_processor,
            dense_retriever,
            sparse_retriever,
            fusion
        )

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the query_knowledge_hub tool.

        Args:
            arguments: Tool arguments (query, top_k, collection)

        Returns:
            MCP tool result with content and metadata
        """
        # Validate arguments
        query = arguments.get('query')
        if not query:
            return {
                "content": [{"type": "text", "text": "Error: query parameter is required"}],
                "isError": True
            }

        top_k = arguments.get('top_k', 10)
        collection = arguments.get('collection')

        # Build filters
        filters = {}
        if collection:
            filters['collection'] = collection

        # Execute hybrid search
        search_results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None
        )

        # Rerank results
        rerank_result = self.reranker.rerank(
            query=query,
            candidates=[
                {
                    'chunk_id': r.chunk_id,
                    'text': r.text,
                    'score': r.score,
                    'metadata': r.metadata
                }
                for r in search_results
            ]
        )

        # Build MCP response
        mcp_response = self.response_builder.build(
            retrieval_results=rerank_result['results'],
            query=query
        )

        return {
            "content": mcp_response.content,
            "isError": mcp_response.isError
        }


# Global instance (initialized on first use)
_tool_instance: Optional[QueryKnowledgeHub] = None


def query_knowledge_hub(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query knowledge hub tool entry point.

    Args:
        arguments: Tool arguments from MCP client

    Returns:
        MCP tool result
    """
    global _tool_instance

    if _tool_instance is None:
        settings = Settings()
        _tool_instance = QueryKnowledgeHub(settings)

    return _tool_instance.execute(arguments)
