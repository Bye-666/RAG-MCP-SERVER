from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ...core.trace import TraceContext


class BaseVectorStore(ABC):
    """向量存储后端的抽象接口"""

    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """
        在向量存储中插入或更新记录
        返回记录 ID 列表
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
        使用向量查询向量存储
        返回包含 id、score、text、metadata 的结果列表
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        通过 ID 检索记录
        返回包含 id、text、metadata 的记录列表
        """
        pass