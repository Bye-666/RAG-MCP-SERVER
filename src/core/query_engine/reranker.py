"""
用于优化检索结果的重排序器。

包装 libs.reranker 后端，带有回退机制以提高鲁棒性。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.core.types import RetrievalResult
from src.core.trace import TraceContext
from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker


@dataclass
class RerankResult:
    """重排序操作的结果

    属性:
        results: 重排序后的 RetrievalResult 列表
        fallback: 如果重排序失败并使用原始顺序则为 True
        error: 如果发生回退的错误消息
    """
    results: List[RetrievalResult]
    fallback: bool = False
    error: Optional[str] = None


class Reranker:
    """
    用于优化检索结果的重排序器。

    包装重排序器后端（None/CrossEncoder/LLM）并带有回退机制。
    如果重排序失败或超时，返回原始排名并带有回退标志。
    """

    def __init__(
        self,
        settings: Settings,
        reranker_backend: Optional[BaseReranker] = None
    ):
        """
        初始化 Reranker。

        参数:
            settings: 应用程序设置
            reranker_backend: 可选的重排序器后端（用于依赖注入）
        """
        self.settings = settings

        # 使用注入的后端或从设置创建
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
        根据与查询的相关性对候选结果重排序。

        参数:
            query: 用户查询字符串
            candidates: 要重排序的候选结果列表
            trace: 可选的追踪上下文

        返回:
            包含重排序结果和回退状态的 RerankResult

        异常:
            ValueError: 如果查询为空或候选结果为 None
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        if candidates is None:
            raise ValueError("候选结果不能为 None")

        # 空候选结果 - 原样返回
        if not candidates:
            return RerankResult(results=[], fallback=False)

        # 将 RetrievalResult 转换为后端的字典格式
        candidate_dicts = [
            {
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in candidates
        ]

        # 尝试重排序并带有回退
        if trace:
            stage = trace.record_stage("rerank", {
                "candidate_count": len(candidates),
                "method": self.backend.__class__.__name__
            })

        try:
            # 调用后端重排序器
            reranked_dicts = self.backend.rerank(
                query=query,
                candidates=candidate_dicts,
                trace=trace
            )

            # 转换回 RetrievalResult
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
            # 回退：返回原始排名
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
