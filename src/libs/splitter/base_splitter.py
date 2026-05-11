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
        将输入文本分割为块
        """
        pass