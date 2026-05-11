"""
用于 BM25 关键词搜索的稀疏检索器。

使用 BM25 倒排索引进行关键词匹配，并从向量存储检索完整文本。
"""

from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """
    用于 BM25 关键词搜索的稀疏检索器。

    编排:
    1. 关键词 → BM25 索引查询 → top-k chunk_ids 及分数
    2. Chunk_ids → VectorStore.get_by_ids() → 文本和元数据
    3. 合并分数与文本/元数据 → RetrievalResult 列表
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None
    ):
        """
        初始化 SparseRetriever。

        参数:
            settings: 应用程序设置
            bm25_indexer: 可选的 BM25 索引器（用于依赖注入）
            vector_store: 可选的向量存储（用于依赖注入）
        """
        self.settings = settings

        # 使用注入的依赖或从设置创建
        if bm25_indexer is not None:
            self.bm25_indexer = bm25_indexer
        else:
            # 从默认位置加载 BM25 索引
            self.bm25_indexer = BM25Indexer(settings=settings)
            # 尝试加载索引，但如果不存在不要失败
            try:
                self.bm25_indexer.load()
            except FileNotFoundError:
                # 索引尚不存在 - 将在摄取期间创建
                pass

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
        使用 BM25 检索匹配关键词的 top-k 块。

        参数:
            keywords: 要搜索的关键词列表
            top_k: 返回的结果数量
            trace: 可选的追踪上下文

        返回:
            按 BM25 分数排序的 RetrievalResult 列表（降序）

        异常:
            ValueError: 如果关键词为空或 top_k 无效
        """
        if not keywords:
            raise ValueError("关键词不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")

        # 步骤 1: 查询 BM25 索引
        if trace:
            stage = trace.record_stage("sparse_retriever_bm25", {
                "keyword_count": len(keywords),
                "top_k": top_k
            })

        bm25_results = self.bm25_indexer.query(keywords, top_k=top_k)

        if trace:
            trace.finish_stage(stage, {"result_count": len(bm25_results)})

        # 如果没有结果，返回空列表
        if not bm25_results:
            return []

        # 步骤 2: 获取 chunk_ids 和分数
        chunk_ids = [result["chunk_id"] for result in bm25_results]
        score_map = {result["chunk_id"]: result["score"] for result in bm25_results}

        # 步骤 3: 从向量存储检索文本和元数据
        if trace:
            stage = trace.record_stage("sparse_retriever_fetch", {
                "chunk_count": len(chunk_ids)
            })

        chunk_data = self.vector_store.get_by_ids(chunk_ids)

        if trace:
            trace.finish_stage(stage, {"fetched_count": len(chunk_data)})

        # 步骤 4: 合并分数与文本/元数据
        # 创建映射以便快速查找
        chunk_map = {item["id"]: item for item in chunk_data}

        results = []
        for chunk_id in chunk_ids:
            # 从 BM25 结果获取分数
            score = score_map.get(chunk_id, 0.0)

            # 从向量存储获取文本和元数据
            chunk_info = chunk_map.get(chunk_id)
            if chunk_info:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=chunk_info.get("text", ""),
                    metadata=chunk_info.get("metadata", {})
                ))

        return results
