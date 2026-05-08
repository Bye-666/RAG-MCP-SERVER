from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends"""

    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """
        Insert or update records in the vector store
        Returns list of record IDs
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store with a vector
        Returns list of results with id, score, text, metadata
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve records by their IDs
        Returns list of records with id, text, metadata
        """
        pass