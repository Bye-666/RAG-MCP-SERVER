from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.trace import TraceContext


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str], trace: Optional[TraceContext] = None) -> List[List[float]]:
        """Embed a list of texts into vectors"""
        pass
