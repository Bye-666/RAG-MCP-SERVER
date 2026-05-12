"""批量编码块的批处理器

该模块提供 BatchProcessor 类，使用密集和稀疏编码器
协调块的批量编码。
"""

from typing import List, Optional
import time
from src.core.types import Chunk, ChunkRecord
from src.core.trace import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """通过密集和稀疏编码器批量处理块

    该处理器接收块列表，将它们分成批次，
    并通过密集和稀疏编码器处理每个批次。
    它记录时间信息以便观测。

    Attributes:
        dense_encoder: 用于生成密集向量的可选密集编码器
        sparse_encoder: 用于生成稀疏向量的可选稀疏编码器
        batch_size: 每批处理的块数
    """

    def __init__(
        self,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_size: int = 10
    ):
        """初始化批处理器

        Args:
            dense_encoder: 可选的 DenseEncoder 实例
            sparse_encoder: 可选的 SparseEncoder 实例
            batch_size: 每批的块数（默认：10，适配大多数 embedding 模型的限制）

        Raises:
            ValueError: 如果两个编码器都为 None 或 batch_size < 1
        """
        if dense_encoder is None and sparse_encoder is None:
            raise ValueError("必须提供至少一个编码器（密集或稀疏）")

        if batch_size < 1:
            raise ValueError("batch_size 必须至少为 1")

        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """通过编码器批量处理块

        Args:
            chunks: 要处理的 Chunk 对象列表
            trace: 可选的跟踪上下文，用于可观测性

        Returns:
            填充了向量的 ChunkRecord 对象列表

        Raises:
            ValueError: 如果块列表为空
        """
        if not chunks:
            raise ValueError("块列表不能为空")

        all_records = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            batch_start_time = time.time()

            # 通过编码器处理批次
            batch_records = self._process_batch(batch_chunks, trace)

            batch_duration = time.time() - batch_start_time

            # 如果跟踪可用，记录批次时间
            if trace:
                trace.record_stage(
                    stage_name=f"batch_{batch_idx + 1}",
                    metadata={
                        "batch_size": len(batch_chunks),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration_ms": batch_duration * 1000
                    }
                )

            all_records.extend(batch_records)

        return all_records

    def _process_batch(
        self,
        batch_chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """通过编码器处理单个批次

        Args:
            batch_chunks: 此批次中的块
            trace: 可选的跟踪上下文

        Returns:
            此批次的 ChunkRecord 对象列表
        """
        # 如果可用，从密集编码开始
        if self.dense_encoder:
            records = self.dense_encoder.encode(batch_chunks, trace=trace)
        else:
            # 创建不带密集向量的记录
            records = [
                ChunkRecord.from_chunk(chunk)
                for chunk in batch_chunks
            ]

        # 如果可用，添加稀疏编码
        if self.sparse_encoder:
            sparse_records = self.sparse_encoder.encode(batch_chunks, trace=trace)

            # 将稀疏向量合并到现有记录中
            for record, sparse_record in zip(records, sparse_records):
                record.sparse_vector = sparse_record.sparse_vector

        return records
