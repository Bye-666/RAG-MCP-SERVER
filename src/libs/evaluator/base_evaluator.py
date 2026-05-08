from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseEvaluator(ABC):
    """Abstract interface for evaluators"""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[TraceContext] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against golden standard
        Returns dict of metric_name -> score
        """
        pass