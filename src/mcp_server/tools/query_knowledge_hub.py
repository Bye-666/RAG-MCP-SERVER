"""查询知识中心工具实现。"""

from typing import Dict, Any, Optional
from ...core.config import Settings
from ...core.query_engine.hybrid_search import HybridSearch
from ...core.query_engine.reranker import Reranker
from ...core.response.response_builder import ResponseBuilder


def get_tool_schema() -> Dict[str, Any]:
    """获取 query_knowledge_hub 的工具 schema。"""
    return {
        "name": "query_knowledge_hub",
        "description": "使用混合检索（稠密 + 稀疏 + 重排序）搜索知识库",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量（默认：10）",
                    "default": 10
                },
                "collection": {
                    "type": "string",
                    "description": "可选的集合名称，用于过滤结果"
                }
            },
            "required": ["query"]
        }
    }


class QueryKnowledgeHub:
    """查询知识中心工具处理器。"""

    def __init__(
        self,
        settings: Settings,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[Reranker] = None,
        response_builder: Optional[ResponseBuilder] = None
    ):
        """
        初始化查询知识中心工具。

        Args:
            settings: 应用程序设置
            hybrid_search: HybridSearch 实例（如果为 None 则创建默认实例）
            reranker: Reranker 实例（如果为 None 则创建默认实例）
            response_builder: ResponseBuilder 实例（如果为 None 则创建默认实例）
        """
        self.settings = settings
        self.hybrid_search = hybrid_search or self._create_hybrid_search()
        self.reranker = reranker or Reranker(settings)
        self.response_builder = response_builder or ResponseBuilder()

    def _create_hybrid_search(self) -> HybridSearch:
        """创建包含所有依赖项的 HybridSearch 实例。"""
        from ...core.query_engine.query_processor import QueryProcessor
        from ...core.query_engine.dense_retriever import DenseRetriever
        from ...core.query_engine.sparse_retriever import SparseRetriever
        from ...core.query_engine.fusion import RRFFusion
        from ...libs.embedding import EmbeddingFactory
        from ...ingestion.storage.vector_store import VectorStoreFactory
        from ...ingestion.storage.bm25_indexer import BM25Indexer

        # 初始化组件
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
        执行 query_knowledge_hub 工具。

        Args:
            arguments: 工具参数（query、top_k、collection）

        Returns:
            包含内容和元数据的 MCP 工具结果
        """
        # 验证参数
        query = arguments.get('query')
        if not query:
            return {
                "content": [{"type": "text", "text": "错误：query 参数是必需的"}],
                "isError": True
            }

        top_k = arguments.get('top_k', 10)
        collection = arguments.get('collection')

        # 构建过滤器
        filters = {}
        if collection:
            filters['collection'] = collection

        # 执行混合搜索
        search_results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None
        )

        # 重排序结果
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

        # 构建 MCP 响应
        mcp_response = self.response_builder.build(
            retrieval_results=rerank_result['results'],
            query=query
        )

        return {
            "content": mcp_response.content,
            "isError": mcp_response.isError
        }


# 全局实例（首次使用时初始化）
_tool_instance: Optional[QueryKnowledgeHub] = None


def query_knowledge_hub(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    查询知识中心工具入口点。

    Args:
        arguments: 来自 MCP 客户端的工具参数

    Returns:
        MCP 工具结果
    """
    global _tool_instance

    if _tool_instance is None:
        settings = Settings()
        _tool_instance = QueryKnowledgeHub(settings)

    return _tool_instance.execute(arguments)
