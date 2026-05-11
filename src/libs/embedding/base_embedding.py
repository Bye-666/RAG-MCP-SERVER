from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.trace import TraceContext


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str], trace: Optional[TraceContext] = None) -> List[List[float]]:
        """将文本列表转换为向量"""
        pass
