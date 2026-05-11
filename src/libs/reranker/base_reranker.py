from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseReranker(ABC):
    """重排序器后端的抽象接口"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        根据与查询的相关性对候选项进行重排序
        返回带有更新分数的已排序候选项列表
        """
        pass