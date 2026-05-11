"""
用于语义向量搜索的稠密检索器。

结合 embedding 生成和向量存储查询进行语义检索。
"""

from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """
    用于语义向量搜索的稠密检索器。

    编排:
    1. 查询文本 → embedding 向量（通过 EmbeddingClient）
    2. 向量 → top-k 相似块（通过 VectorStore）
    3. 结果 → RetrievalResult 列表
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None
    ):
        """
        初始化 DenseRetriever。

        参数:
            settings: 应用程序设置
            embedding_client: 可选的 embedding 客户端（用于依赖注入）
            vector_store: 可选的向量存储（用于依赖注入）
        """
        self.settings = settings

        # 使用注入的依赖或从设置创建
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
        检索查询的 top-k 语义相似块。

        参数:
            query: 用户查询字符串
            top_k: 返回的结果数量
            filters: 可选的元数据过滤器
            trace: 可选的追踪上下文

        返回:
            按相关性分数排序的 RetrievalResult 列表（降序）

        异常:
            ValueError: 如果查询为空或 top_k 无效
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")

        # 步骤 1: 生成查询 embedding
        if trace:
            stage = trace.record_stage("dense_retriever_embed", {"query_length": len(query)})

        embeddings = self.embedding_client.embed([query], trace=trace)
        query_vector = embeddings[0]

        if trace:
            trace.finish_stage(stage, {"vector_dim": len(query_vector)})

        # 步骤 2: 查询向量存储
        if trace:
            stage = trace.record_stage("dense_retriever_query", {
                "top_k": top_k,
                "has_filters": filters is not None
            })

        # 将空字典转换为 None（ChromaDB 不接受空字典）
        query_filters = filters if filters else None

        raw_results = self.vector_store.query(
            vector=query_vector,
            top_k=top_k,
            filters=query_filters,
            trace=trace
        )

        if trace:
            trace.finish_stage(stage, {"result_count": len(raw_results)})

        # 步骤 3: 转换为 RetrievalResult
        results = []
        for item in raw_results:
            results.append(RetrievalResult(
                chunk_id=item["id"],
                score=item["score"],
                text=item["text"],
                metadata=item.get("metadata", {})
            ))

        return results
