"""Batch processor for encoding chunks in batches

This module provides the BatchProcessor class that orchestrates batch-wise
encoding of chunks using both dense and sparse encoders.
"""

from typing import List, Optional
import time
from src.core.types import Chunk, ChunkRecord
from src.core.trace import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """Processes chunks in batches through dense and sparse encoders

    This processor takes a list of chunks, splits them into batches,
    and processes each batch through both dense and sparse encoders.
    It records timing information for observability.

    Attributes:
        dense_encoder: Optional dense encoder for generating dense vectors
        sparse_encoder: Optional sparse encoder for generating sparse vectors
        batch_size: Number of chunks to process in each batch
    """

    def __init__(
        self,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_size: int = 32
    ):
        """Initialize the batch processor

        Args:
            dense_encoder: Optional DenseEncoder instance
            sparse_encoder: Optional SparseEncoder instance
            batch_size: Number of chunks per batch (default: 32)

        Raises:
            ValueError: If both encoders are None or batch_size < 1
        """
        if dense_encoder is None and sparse_encoder is None:
            raise ValueError("At least one encoder (dense or sparse) must be provided")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """Process chunks in batches through encoders

        Args:
            chunks: List of Chunk objects to process
            trace: Optional trace context for observability

        Returns:
            List of ChunkRecord objects with vectors populated

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        all_records = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            batch_start_time = time.time()

            # Process batch through encoders
            batch_records = self._process_batch(batch_chunks, trace)

            batch_duration = time.time() - batch_start_time

            # Log batch timing if trace is available
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
        """Process a single batch through encoders

        Args:
            batch_chunks: Chunks in this batch
            trace: Optional trace context

        Returns:
            List of ChunkRecord objects for this batch
        """
        # Start with dense encoding if available
        if self.dense_encoder:
            records = self.dense_encoder.encode(batch_chunks, trace=trace)
        else:
            # Create records without dense vectors
            records = [
                ChunkRecord.from_chunk(chunk)
                for chunk in batch_chunks
            ]

        # Add sparse encoding if available
        if self.sparse_encoder:
            sparse_records = self.sparse_encoder.encode(batch_chunks, trace=trace)

            # Merge sparse vectors into existing records
            for record, sparse_record in zip(records, sparse_records):
                record.sparse_vector = sparse_record.sparse_vector

        return records
