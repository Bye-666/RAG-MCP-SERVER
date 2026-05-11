"""使用 BM25 统计的文本块稀疏向量编码器

该模块提供 SparseEncoder 类，将文本块转换为适合 BM25 索引的
稀疏向量表示（词项权重）。
"""

from typing import List, Dict, Optional
from collections import Counter
import re
from src.core.types import Chunk, ChunkRecord
from src.core.trace import TraceContext


class SparseEncoder:
    """将文本块编码为用于 BM25 的稀疏向量（词项权重）

    该编码器接收 Chunk 对象列表，生成填充了 sparse_vector 的 ChunkRecord 对象。
    稀疏向量是一个将词项映射到其词频（TF）值的字典。

    Attributes:
        stopwords: 在分词期间过滤掉的停用词集合
        min_term_length: 词项被包含的最小长度
    """

    # 常见英文停用词
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }

    def __init__(
        self,
        stopwords: Optional[set] = None,
        min_term_length: int = 2
    ):
        """初始化稀疏编码器

        Args:
            stopwords: 要过滤掉的停用词集合（如果为 None 则使用默认值）
            min_term_length: 词项被包含的最小长度
        """
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_term_length = min_term_length

    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为词项

        Args:
            text: 要分词的输入文本

        Returns:
            标准化词项列表（小写，已过滤）
        """
        # 转换为小写并提取字母数字标记
        tokens = re.findall(r'\b\w+\b', text.lower())

        # 按长度和停用词过滤
        filtered = [
            token for token in tokens
            if len(token) >= self.min_term_length and token not in self.stopwords
        ]

        return filtered

    def _compute_term_weights(self, tokens: List[str]) -> Dict[str, float]:
        """计算标记的词项频率权重

        Args:
            tokens: 来自块的标记列表

        Returns:
            将词项映射到其 TF 值的字典
        """
        if not tokens:
            return {}

        # 计算词项频率
        term_counts = Counter(tokens)
        doc_length = len(tokens)

        # 计算标准化词项频率（TF）
        # TF = (term_count / doc_length)
        term_weights = {
            term: count / doc_length
            for term, count in term_counts.items()
        }

        return term_weights

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """将块编码为稀疏向量（词项权重）

        Args:
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的跟踪上下文，用于可观测性

        Returns:
            填充了 sparse_vector 的 ChunkRecord 对象列表

        Raises:
            ValueError: 如果块列表为空
        """
        if not chunks:
            raise ValueError("块列表不能为空")

        records = []
        for chunk in chunks:
            # 对块文本进行分词
            tokens = self._tokenize(chunk.text)

            # 计算词项权重
            term_weights = self._compute_term_weights(tokens)

            # 创建带有稀疏向量的 ChunkRecord
            record = ChunkRecord.from_chunk(
                chunk=chunk,
                sparse_vector=term_weights
            )
            records.append(record)

        return records
