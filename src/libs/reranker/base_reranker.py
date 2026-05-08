from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseReranker(ABC):
    """Abstract interface for reranker backends"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates based on relevance to query
        Returns sorted list of candidates with updated scores
        """
        pass