"""文本块的密集向量编码器

该模块提供 DenseEncoder 类，使用配置的嵌入提供程序将文本块
转换为密集嵌入向量。
"""

from typing import List, Optional
from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.trace import TraceContext


class DenseEncoder:
    """将文本块编码为密集嵌入向量

    该编码器接收 Chunk 对象列表，生成填充了 dense_vector 的 ChunkRecord 对象。
    它将实际的嵌入计算委托给 BaseEmbedding 实现（例如 OpenAI、Azure、Ollama）。

    Attributes:
        embedding_model: 嵌入提供程序实例
    """

    def __init__(self, embedding_model: BaseEmbedding):
        """初始化密集编码器

        Args:
            embedding_model: 用于计算嵌入的 BaseEmbedding 实例
        """
        if not isinstance(embedding_model, BaseEmbedding):
            raise TypeError("embedding_model 必须是 BaseEmbedding 的实例")
        self.embedding_model = embedding_model

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """将块编码为密集向量

        Args:
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的跟踪上下文，用于可观测性

        Returns:
            填充了 dense_vector 的 ChunkRecord 对象列表

        Raises:
            ValueError: 如果块列表为空
        """
        if not chunks:
            raise ValueError("块列表不能为空")

        # 从所有块中提取文本
        texts = [chunk.text for chunk in chunks]

        # 调用嵌入模型获取向量
        vectors = self.embedding_model.embed(texts, trace=trace)

        # 验证输出维度
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"嵌入模型返回了 {len(vectors)} 个向量，"
                f"但期望 {len(chunks)} 个"
            )

        # 创建带有密集向量的 ChunkRecord 对象
        records = []
        for chunk, vector in zip(chunks, vectors):
            record = ChunkRecord.from_chunk(
                chunk=chunk,
                dense_vector=vector
            )
            records.append(record)

        return records
