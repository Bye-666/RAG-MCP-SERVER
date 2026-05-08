"""Unit tests for BatchProcessor"""

import pytest
from typing import List, Optional
from src.core.types import Chunk, ChunkRecord
from src.core.trace import TraceContext
from src.libs.embedding.base_embedding import BaseEmbedding
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor


class FakeEmbedding(BaseEmbedding):
    """Fake embedding model for testing"""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.call_count = 0
        self.call_sizes = []

    def embed(self, texts: List[str], trace: Optional[TraceContext] = None) -> List[List[float]]:
        self.call_count += 1
        self.call_sizes.append(len(texts))
        return [[i * 0.1] * self.dimension for i in range(1, len(texts) + 1)]


class TestBatchProcessor:
    """Test suite for BatchProcessor"""

    def test_process_with_dense_only(self):
        """Test processing with only dense encoder"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=2)

        chunks = [
            Chunk(id="c1", text="text one", metadata={}),
            Chunk(id="c2", text="text two", metadata={}),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        assert records[0].id == "c1"
        assert records[1].id == "c2"
        assert records[0].dense_vector is not None
        assert records[0].sparse_vector is None
        assert len(records[0].dense_vector) == 4

    def test_process_with_sparse_only(self):
        """Test processing with only sparse encoder"""
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(sparse_encoder=sparse_encoder, batch_size=2)

        chunks = [
            Chunk(id="c1", text="machine learning", metadata={}),
            Chunk(id="c2", text="deep learning", metadata={}),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        assert records[0].id == "c1"
        assert records[1].id == "c2"
        assert records[0].dense_vector is None
        assert records[0].sparse_vector is not None
        assert "machine" in records[0].sparse_vector

    def test_process_with_both_encoders(self):
        """Test processing with both dense and sparse encoders"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        chunks = [
            Chunk(id="c1", text="machine learning", metadata={}),
            Chunk(id="c2", text="deep learning", metadata={}),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        assert records[0].dense_vector is not None
        assert records[0].sparse_vector is not None
        assert len(records[0].dense_vector) == 4
        assert "machine" in records[0].sparse_vector

    def test_batch_size_2_with_5_chunks(self):
        """Test batch_size=2 with 5 chunks creates 3 batches"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=2)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(1, 6)
        ]

        records = processor.process(chunks)

        # Verify 3 batches were created (2, 2, 1)
        assert model.call_count == 3
        assert model.call_sizes == [2, 2, 1]

        # Verify all chunks were processed
        assert len(records) == 5

        # Verify order is stable
        for i, record in enumerate(records, start=1):
            assert record.id == f"c{i}"
            assert record.text == f"text {i}"

    def test_order_stability(self):
        """Test that chunk order is preserved across batches"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=3)

        chunks = [
            Chunk(id=f"chunk_{i:03d}", text=f"content {i}", metadata={"index": i})
            for i in range(10)
        ]

        records = processor.process(chunks)

        assert len(records) == 10

        # Verify order matches input
        for i, record in enumerate(records):
            assert record.id == f"chunk_{i:03d}"
            assert record.text == f"content {i}"
            assert record.metadata["index"] == i

    def test_single_batch(self):
        """Test processing when all chunks fit in one batch"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=10)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        records = processor.process(chunks)

        assert model.call_count == 1
        assert len(records) == 5

    def test_exact_batch_boundary(self):
        """Test when chunk count is exact multiple of batch_size"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=3)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(9)
        ]

        records = processor.process(chunks)

        assert model.call_count == 3
        assert model.call_sizes == [3, 3, 3]
        assert len(records) == 9

    def test_batch_size_one(self):
        """Test with batch_size=1"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=1)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(3)
        ]

        records = processor.process(chunks)

        assert model.call_count == 3
        assert model.call_sizes == [1, 1, 1]
        assert len(records) == 3

    def test_large_batch_size(self):
        """Test with batch_size larger than chunk count"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=100)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        records = processor.process(chunks)

        assert model.call_count == 1
        assert len(records) == 5

    def test_metadata_preservation(self):
        """Test that metadata is preserved through processing"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        chunks = [
            Chunk(id="c1", text="text one", metadata={"source": "doc1", "page": 1}),
            Chunk(id="c2", text="text two", metadata={"source": "doc2", "page": 2}),
        ]

        records = processor.process(chunks)

        assert records[0].metadata == {"source": "doc1", "page": 1}
        assert records[1].metadata == {"source": "doc2", "page": 2}

    def test_trace_context_recording(self):
        """Test that trace context records batch information"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=2)

        chunks = [
            Chunk(id=f"c{i}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        trace = TraceContext(trace_id="test_trace")
        records = processor.process(chunks, trace=trace)

        assert len(records) == 5

        # Verify trace recorded batch stages
        stages = trace.stages
        assert len(stages) == 3  # 3 batches

        # Check first batch
        assert stages[0].stage_name == "batch_1"
        assert stages[0].metadata["batch_size"] == 2
        assert stages[0].metadata["start_idx"] == 0
        assert stages[0].metadata["end_idx"] == 2

        # Check last batch
        assert stages[2].stage_name == "batch_3"
        assert stages[2].metadata["batch_size"] == 1
        assert stages[2].metadata["start_idx"] == 4
        assert stages[2].metadata["end_idx"] == 5

    def test_no_encoders_raises_error(self):
        """Test that creating processor without encoders raises error"""
        with pytest.raises(ValueError, match="At least one encoder"):
            BatchProcessor(dense_encoder=None, sparse_encoder=None)

    def test_invalid_batch_size_raises_error(self):
        """Test that invalid batch_size raises error"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)

        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            BatchProcessor(dense_encoder=dense_encoder, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            BatchProcessor(dense_encoder=dense_encoder, batch_size=-1)

    def test_empty_chunks_raises_error(self):
        """Test that empty chunks list raises error"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=2)

        with pytest.raises(ValueError, match="chunks list cannot be empty"):
            processor.process([])

    def test_process_without_trace(self):
        """Test processing without trace context"""
        model = FakeEmbedding(dimension=4)
        dense_encoder = DenseEncoder(embedding_model=model)
        processor = BatchProcessor(dense_encoder=dense_encoder, batch_size=2)

        chunks = [
            Chunk(id="c1", text="text one", metadata={}),
            Chunk(id="c2", text="text two", metadata={}),
        ]

        # Should not raise error when trace is None
        records = processor.process(chunks, trace=None)
        assert len(records) == 2
