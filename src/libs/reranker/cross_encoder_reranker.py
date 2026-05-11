"""基于 Cross-Encoder 的重排序器"""

from typing import Any, Dict, List, Optional
from .base_reranker import BaseReranker
from ...core.trace import TraceContext


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排序器，使用 sentence-transformers"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", scorer=None, timeout: Optional[float] = None):
        """
        初始化 Cross-Encoder 重排序器

        参数:
            model_name: cross-encoder 模型名称
            scorer: 可选的评分器实例（用于测试/模拟）
            timeout: 可选的评分超时时间（秒）
        """
        self.model_name = model_name
        self.timeout = timeout

        if scorer is not None:
            self.scorer = scorer
        else:
            try:
                from sentence_transformers import CrossEncoder
                self.scorer = CrossEncoder(model_name)
            except ImportError as e:
                raise ImportError(
                    "CrossEncoderReranker 需要 sentence-transformers。"
                    "使用以下命令安装: pip install sentence-transformers"
                ) from e

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 Cross-Encoder 模型对候选项进行重排序

        参数:
            query: 用户查询
            candidates: 候选项字典列表（必须有 'text' 字段）
            trace: 可选的跟踪上下文

        返回:
            带有更新分数的重排序候选项列表

        异常:
            ValueError: 如果候选项格式无效
            RuntimeError: 如果评分失败（允许回退到原始排序）
            TimeoutError: 如果评分超过超时时间（允许回退）
        """
        if not candidates:
            return []

        if trace:
            trace.log("cross_encoder_reranker", f"使用 {self.model_name} 重排序 {len(candidates)} 个候选项")

        # 验证候选项具有必需字段
        for i, cand in enumerate(candidates):
            if 'text' not in cand:
                raise ValueError(f"候选项 {i} 缺少 'text' 字段")

        # 构建查询-候选项对
        pairs = [[query, cand['text']] for cand in candidates]

        # 对对进行评分
        try:
            if self.timeout is not None:
                # 为了支持超时，我们需要包装 predict 调用
                # 现在，我们只是传递并让调用者处理超时
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Cross-encoder 评分超过 {self.timeout} 秒的超时时间")

                # 注意: signal.alarm 仅在 Unix 系统上有效
                # 为了 Windows 兼容性，我们将跳过实际的超时实现
                # 只记录接口
                try:
                    if hasattr(signal, 'alarm'):
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(self.timeout))

                    scores = self.scorer.predict(pairs)

                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)  # 取消警报
                except TimeoutError:
                    raise
            else:
                scores = self.scorer.predict(pairs)

            if trace:
                trace.log("cross_encoder_reranker", f"评分了 {len(scores)} 对")

        except TimeoutError:
            raise  # 重新抛出超时以进行回退处理
        except Exception as e:
            raise RuntimeError(f"Cross-encoder 评分失败: {str(e)}") from e

        # 构建带有分数的重排序结果
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = cand.copy()
            cand_copy['rerank_score'] = float(score)
            scored_candidates.append(cand_copy)

        # 按分数排序（降序）
        reranked = sorted(scored_candidates, key=lambda x: x['rerank_score'], reverse=True)

        if trace:
            top_scores = [f"{c.get('id', 'N/A')}:{c['rerank_score']:.3f}" for c in reranked[:3]]
            trace.log("cross_encoder_reranker", f"前 3 个分数: {top_scores}")

        return reranked
