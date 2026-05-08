"""Unit tests for DenseEncoder

Tests cover:
- Normal encoding with single and multiple chunks
- Empty chunks list handling
- Vector dimension consistency
- Integration with BaseEmbedding interface
- Error handling for invalid inputs
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    """Fake embedding model for testing"""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.call_count = 0

    def embed(self, texts, trace=None):
        """Return fake vectors with consistent dimension"""
        self.call_count += 1
        return [[0.1 * (i + 1)] * self.dimension for i in range(len(texts))]


class TestDenseEncoder:
    """Test cases for DenseEncoder class"""

    def test_initialization_valid_model(self):
        """Test initialization with valid embedding model"""
        model = FakeEmbedding()
        encoder = DenseEncoder(embedding_model=model)
        assert encoder.embedding_model is model

    def test_initialization_invalid_model(self):
        """Test initialization with invalid model raises TypeError"""
        with pytest.raises(TypeError, match="must be an instance of BaseEmbedding"):
            DenseEncoder(embedding_model="not a model")

    def test_encode_single_chunk(self):
        """Test encoding a single chunk"""
        model = FakeEmbedding(dimension=3)
        encoder = DenseEncoder(embedding_model=model)

        chunk = Chunk(
            id="test_001",
            text="Hello world",
            metadata={"source": "test.pdf"}
        )

        records = encoder.encode([chunk])

        assert len(records) == 1
        assert isinstance(records[0], ChunkRecord)
        assert records[0].id == "test_001"
        assert records[0].text == "Hello world"
        assert records[0].dense_vector == [0.1, 0.1, 0.1]
        assert records[0].metadata == {"source": "test.pdf"}

    def test_encode_multiple_chunks(self):
        """Test encoding multiple chunks"""
        model = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding_model=model)

        chunks = [
            Chunk(id="chunk_001", text="First chunk", metadata={"page": 1}),
            Chunk(id="chunk_002", text="Second chunk", metadata={"page": 2}),
            Chunk(id="chunk_003", text="Third chunk", metadata={"page": 3})
        ]

        records = encoder.encode(chunks)

        assert len(records) == 3
        assert records[0].id == "chunk_001"
        assert records[1].id == "chunk_002"
        assert records[2].id == "chunk_003"

        # Verify vectors are different (use approximate comparison for floats)
        assert all(abs(v - 0.1) < 1e-6 for v in records[0].dense_vector)
        assert all(abs(v - 0.2) < 1e-6 for v in records[1].dense_vector)
        assert all(abs(v - 0.3) < 1e-6 for v in records[2].dense_vector)

    def test_encode_empty_chunks_list(self):
        """Test that empty chunks list raises ValueError"""
        model = FakeEmbedding()
        encoder = DenseEncoder(embedding_model=model)

        with pytest.raises(ValueError, match="chunks list cannot be empty"):
            encoder.encode([])

    def test_encode_vector_count_mismatch(self):
        """Test error handling when embedding model returns wrong number of vectors"""
        model = Mock(spec=BaseEmbedding)
        # Return only 1 vector when 3 chunks are provided
        model.embed.return_value = [[0.1, 0.2, 0.3]]

        encoder = DenseEncoder(embedding_model=model)

        chunks = [
            Chunk(id="c1", text="Text 1", metadata={}),
            Chunk(id="c2", text="Text 2", metadata={}),
            Chunk(id="c3", text="Text 3", metadata={})
        ]

        with pytest.raises(RuntimeError, match="returned 1 vectors but expected 3"):
            encoder.encode(chunks)

    def test_encode_preserves_chunk_metadata(self):
        """Test that chunk metadata is preserved in ChunkRecord"""
        model = FakeEmbedding(dimension=2)
        encoder = DenseEncoder(embedding_model=model)

        chunk = Chunk(
            id="test_001",
            text="Test text",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "page": 5,
                "title": "Section 1",
                "custom_field": "custom_value"
            },
            start_offset=100,
            end_offset=200,
            source_ref="doc_123"
        )

        records = encoder.encode([chunk])

        assert records[0].metadata == chunk.metadata
        assert records[0].metadata["source_path"] == "/path/to/doc.pdf"
        assert records[0].metadata["page"] == 5
        assert records[0].metadata["custom_field"] == "custom_value"

    def test_encode_calls_embedding_model_once(self):
        """Test that embedding model is called exactly once for batch"""
        model = FakeEmbedding()
        encoder = DenseEncoder(embedding_model=model)

        chunks = [
            Chunk(id=f"c{i}", text=f"Text {i}", metadata={})
            for i in range(5)
        ]

        encoder.encode(chunks)

        assert model.call_count == 1

    def test_encode_passes_texts_to_model(self):
        """Test that correct texts are passed to embedding model"""
        model = Mock(spec=BaseEmbedding)
        model.embed.return_value = [[0.1] * 3 for _ in range(3)]

        encoder = DenseEncoder(embedding_model=model)

        chunks = [
            Chunk(id="c1", text="First text", metadata={}),
            Chunk(id="c2", text="Second text", metadata={}),
            Chunk(id="c3", text="Third text", metadata={})
        ]

        encoder.encode(chunks)

        # Verify embed was called with correct texts
        model.embed.assert_called_once()
        call_args = model.embed.call_args[0][0]
        assert call_args == ["First text", "Second text", "Third text"]

    def test_encode_with_trace_context(self):
        """Test that trace context is passed to embedding model"""
        model = Mock(spec=BaseEmbedding)
        model.embed.return_value = [[0.1, 0.2, 0.3]]

        encoder = DenseEncoder(embedding_model=model)

        chunk = Chunk(id="c1", text="Test", metadata={})
        mock_trace = Mock()

        encoder.encode([chunk], trace=mock_trace)

        # Verify trace was passed to embed method
        model.embed.assert_called_once()
        assert model.embed.call_args[1]["trace"] is mock_trace

    def test_encode_dimension_consistency(self):
        """Test that all vectors have consistent dimensions"""
        model = FakeEmbedding(dimension=128)
        encoder = DenseEncoder(embedding_model=model)

        chunks = [
            Chunk(id=f"c{i}", text=f"Text {i}", metadata={})
            for i in range(10)
        ]

        records = encoder.encode(chunks)

        # All vectors should have dimension 128
        for record in records:
            assert len(record.dense_vector) == 128

    def test_encode_sparse_vector_remains_none(self):
        """Test that sparse_vector field remains None after dense encoding"""
        model = FakeEmbedding(dimension=3)
        encoder = DenseEncoder(embedding_model=model)

        chunk = Chunk(id="c1", text="Test", metadata={})
        records = encoder.encode([chunk])

        assert records[0].dense_vector is not None
        assert records[0].sparse_vector is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
