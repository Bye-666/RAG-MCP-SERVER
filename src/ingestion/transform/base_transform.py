"""
BaseTransform: 块转换操作的抽象基类。

数据摄取管道中的所有转换操作都应继承此类。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace import TraceContext


class BaseTransform(ABC):
    """
    块转换的抽象基类。

    转换操作在数据摄取管道期间修改块，例如：
    - ChunkRefiner: 清理和精炼块文本
    - MetadataEnricher: 添加元数据，如标题、摘要、标签
    - ImageCaptioner: 为图像生成标题

    所有转换应该：
    1. 接受块列表和可选的跟踪上下文
    2. 返回修改后的块列表
    3. 优雅地处理错误，不阻塞管道
    4. 在跟踪上下文中记录其操作
    """

    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        转换块列表。

        Args:
            chunks: 要转换的 Chunk 对象列表
            trace: 可选的跟踪上下文，用于记录操作

        Returns:
            转换后的 Chunk 对象列表

        Raises:
            不应引发阻塞管道的异常。
            错误应被记录并优雅处理。
        """
        pass
