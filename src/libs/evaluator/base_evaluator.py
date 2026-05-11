from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseEvaluator(ABC):
    """评估器的抽象接口"""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext] = None
    ) -> Dict[str, float]:
        """
        根据黄金标准评估检索结果
        返回 metric_name -> score 的字典
        """
        pass