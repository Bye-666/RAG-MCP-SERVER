"""
结合稠密和稀疏检索的混合搜索引擎。

编排 QueryProcessor、DenseRetriever、SparseRetriever 和 RRF Fusion
以实现全面的语义 + 关键词搜索。
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
    结合稠密和稀疏检索的混合搜索引擎。

    流水线:
    1. QueryProcessor: 提取关键词并解析过滤器
    2. 并行检索:
       - DenseRetriever: 语义向量搜索
       - SparseRetriever: BM25 关键词搜索
    3. RRF Fusion: 合并排名
    4. 元数据过滤: 按元数据后过滤
    5. Top-K 选择
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
        初始化 HybridSearch。

        参数:
            settings: 应用程序设置
            query_processor: 可选的查询处理器（用于依赖注入）
            dense_retriever: 可选的稠密检索器（用于依赖注入）
            sparse_retriever: 可选的稀疏检索器（用于依赖注入）
            fusion: 可选的 RRF 融合（用于依赖注入）
        """
        self.settings = settings

        # 使用注入的依赖或创建默认值
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
        执行结合稠密和稀疏检索的混合搜索。

        参数:
            query: 用户查询字符串
            top_k: 返回的结果数量
            filters: 可选的元数据过滤器
            trace: 可选的追踪上下文

        返回:
            按相关性排序的 top-k RetrievalResult 列表

        异常:
            ValueError: 如果查询为空或 top_k 无效
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")

        # 步骤 1: 处理查询
        if trace:
            stage = trace.record_stage("query_processing", {
                "query_length": len(query),
                "method": "keyword_extraction"
            })

        processed_query = self.query_processor.process(query, filters=filters)

        if trace:
            trace.finish_stage(stage, {
                "keyword_count": len(processed_query.keywords),
                "has_filters": bool(processed_query.filters)
            })

        # 步骤 2: 并行检索（带回退）
        dense_results = []
        sparse_results = []

        # 为融合检索更多候选（2x top_k）
        candidate_k = top_k * 2

        # 稠密检索
        if trace:
            stage = trace.record_stage("dense_retrieval", {
                "top_k": candidate_k,
                "method": self.dense_retriever.__class__.__name__
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
            # 仅使用稀疏检索继续

        # 稀疏检索
        if trace:
            stage = trace.record_stage("sparse_retrieval", {
                "keyword_count": len(processed_query.keywords),
                "top_k": candidate_k,
                "method": self.sparse_retriever.__class__.__name__
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
            # 仅使用稠密检索继续

        # 如果两者都失败，返回空
        if not dense_results and not sparse_results:
            return []

        # 步骤 3: 融合
        if trace:
            stage = trace.record_stage("fusion", {
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "method": "RRF"
            })

        fused_results = self.fusion.fuse(dense_results, sparse_results)

        if trace:
            trace.finish_stage(stage, {"fused_count": len(fused_results)})

        # 步骤 4: 应用元数据过滤器（后过滤回退）
        if filters:
            if trace:
                stage = trace.record_stage("hybrid_search_metadata_filter", {
                    "filter_count": len(filters)
                })

            fused_results = self._apply_metadata_filters(fused_results, filters)

            if trace:
                trace.finish_stage(stage, {"filtered_count": len(fused_results)})

        # 步骤 5: Top-K 选择
        results = fused_results[:top_k]

        return results

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """
        对候选结果应用元数据过滤器（后过滤回退）。

        这是当向量存储过滤器不起作用或需要额外过滤时的回退机制。

        参数:
            candidates: 候选结果列表
            filters: 要应用的元数据过滤器

        返回:
            过滤后的结果列表
        """
        if not filters:
            return candidates

        filtered = []
        for result in candidates:
            # 检查所有过滤条件是否匹配
            match = True
            for key, value in filters.items():
                metadata_value = result.metadata.get(key)
                if metadata_value != value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered
