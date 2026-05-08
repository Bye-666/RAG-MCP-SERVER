from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.trace import TraceContext

class BaseSplitter(ABC):
    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """
        Splits input text into chunks
        """
        pass