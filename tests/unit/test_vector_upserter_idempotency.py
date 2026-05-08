"""Unit tests for VectorUpserter

Tests cover:
- Deterministic chunk ID generation
- Idempotency (same content = same ID)
- Content change detection (different content = different ID)
- Batch upsert functionality
- Order preservation
- Integration with vector store
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """Fake vector store for testing"""

    def __init__(self):
        self.stored_records = []
        self.upsert_call_count = 0

    def upsert(self, records, trace=None):
        self.upsert_call_count += 1
        self.stored_records.extend(records)
        return [r['id'] for r in records]

    def query(self, vector, top_k=5, filters=None, trace=None):
        return []

    def get_by_ids(self, ids):
        return [r for r in self.stored_records if r['id'] in ids]


class TestVectorUpserter:
    """Test suite for VectorUpserter"""

    def test_initialization_valid_store(self):
        """Test initialization with valid vector store"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)
        assert upserter.vector_store is store

    def test_initialization_invalid_store(self):
        """Test initialization with invalid store raises TypeError"""
        with pytest.raises(TypeError, match="must be an instance of BaseVectorStore"):
            VectorUpserter(vector_store="not a store")

    def test_generate_chunk_id_deterministic(self):
        """Test that chunk ID generation is deterministic"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="doc_001_0001_abc123",
            text="Test content",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        id1 = upserter._generate_chunk_id(record)
        id2 = upserter._generate_chunk_id(record)

        assert id1 == id2

    def test_generate_chunk_id_same_content_same_id(self):
        """Test that same content produces same ID (idempotency)"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record1 = ChunkRecord(
            id="doc_001_0001_abc123",
            text="Identical content",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        record2 = ChunkRecord(
            id="doc_001_0001_xyz789",  # Different original ID
            text="Identical content",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.4, 0.5, 0.6]  # Different vector
        )

        id1 = upserter._generate_chunk_id(record1)
        id2 = upserter._generate_chunk_id(record2)

        assert id1 == id2

    def test_generate_chunk_id_different_content_different_id(self):
        """Test that different content produces different ID"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record1 = ChunkRecord(
            id="doc_001_0001_abc123",
            text="First content",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        record2 = ChunkRecord(
            id="doc_001_0001_abc123",
            text="Different content",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        id1 = upserter._generate_chunk_id(record1)
        id2 = upserter._generate_chunk_id(record2)

        assert id1 != id2

    def test_generate_chunk_id_different_source_different_id(self):
        """Test that different source path produces different ID"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record1 = ChunkRecord(
            id="doc_001_0001_abc123",
            text="Same content",
            metadata={"source_path": "/path/to/doc1.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        record2 = ChunkRecord(
            id="doc_001_0001_abc123",
            text="Same content",
            metadata={"source_path": "/path/to/doc2.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        id1 = upserter._generate_chunk_id(record1)
        id2 = upserter._generate_chunk_id(record2)

        assert id1 != id2

    def test_upsert_single_record(self):
        """Test upserting a single record"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        ids = upserter.upsert([record])

        assert len(ids) == 1
        assert isinstance(ids[0], str)
        assert len(ids[0]) > 0
        assert store.upsert_call_count == 1
        assert len(store.stored_records) == 1

    def test_upsert_multiple_records(self):
        """Test upserting multiple records"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        records = [
            ChunkRecord(
                id=f"chunk_{i:03d}",
                text=f"Content {i}",
                metadata={"source_path": "/test.pdf"},
                dense_vector=[i * 0.1, i * 0.2, i * 0.3]
            )
            for i in range(5)
        ]

        ids = upserter.upsert(records)

        assert len(ids) == 5
        assert len(set(ids)) == 5  # All IDs should be unique
        assert store.upsert_call_count == 1
        assert len(store.stored_records) == 5

    def test_upsert_empty_records_raises_error(self):
        """Test that upserting empty list raises error"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        with pytest.raises(ValueError, match="records list cannot be empty"):
            upserter.upsert([])

    def test_upsert_missing_dense_vector_raises_error(self):
        """Test that missing dense vector raises error"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=None  # Missing vector
        )

        with pytest.raises(ValueError, match="missing dense_vector"):
            upserter.upsert([record])

    def test_upsert_preserves_order(self):
        """Test that upsert preserves record order"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        records = [
            ChunkRecord(
                id=f"chunk_{i:03d}",
                text=f"Content {i}",
                metadata={"source_path": "/test.pdf", "index": i},
                dense_vector=[i * 0.1, i * 0.2, i * 0.3]
            )
            for i in range(10)
        ]

        ids = upserter.upsert(records)

        # Verify order is preserved by checking metadata
        for i, stored_record in enumerate(store.stored_records):
            assert stored_record['metadata']['index'] == i

    def test_upsert_includes_sparse_vector(self):
        """Test that sparse vector is included in metadata if present"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"test": 0.5, "content": 0.5}
        )

        upserter.upsert([record])

        stored = store.stored_records[0]
        assert 'sparse_vector' in stored['metadata']
        assert stored['metadata']['sparse_vector'] == {"test": 0.5, "content": 0.5}

    def test_upsert_without_sparse_vector(self):
        """Test that sparse vector is not added if not present"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=None
        )

        upserter.upsert([record])

        stored = store.stored_records[0]
        assert 'sparse_vector' not in stored['metadata']

    def test_upsert_preserves_metadata(self):
        """Test that metadata is preserved during upsert"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={
                "source_path": "/test.pdf",
                "page": 5,
                "title": "Section 1",
                "custom_field": "custom_value"
            },
            dense_vector=[0.1, 0.2, 0.3]
        )

        upserter.upsert([record])

        stored = store.stored_records[0]
        assert stored['metadata']['source_path'] == "/test.pdf"
        assert stored['metadata']['page'] == 5
        assert stored['metadata']['title'] == "Section 1"
        assert stored['metadata']['custom_field'] == "custom_value"

    def test_upsert_single_convenience_method(self):
        """Test upsert_single convenience method"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        chunk_id = upserter.upsert_single(record)

        assert isinstance(chunk_id, str)
        assert len(chunk_id) > 0
        assert store.upsert_call_count == 1
        assert len(store.stored_records) == 1

    def test_upsert_idempotency_same_record_twice(self):
        """Test that upserting same record twice produces same ID"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        ids1 = upserter.upsert([record])
        ids2 = upserter.upsert([record])

        assert ids1[0] == ids2[0]

    def test_upsert_record_format(self):
        """Test that records are formatted correctly for vector store"""
        store = FakeVectorStore()
        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        upserter.upsert([record])

        stored = store.stored_records[0]
        assert 'id' in stored
        assert 'vector' in stored
        assert 'text' in stored
        assert 'metadata' in stored
        assert stored['vector'] == [0.1, 0.2, 0.3]
        assert stored['text'] == "Test content"

    def test_upsert_with_trace_context(self):
        """Test that trace context is passed to vector store"""
        store = Mock(spec=BaseVectorStore)
        store.upsert.return_value = ["id1"]

        upserter = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk_001",
            text="Test content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )

        mock_trace = Mock()
        upserter.upsert([record], trace=mock_trace)

        # Verify trace was passed to vector store
        store.upsert.assert_called_once()
        call_args = store.upsert.call_args
        assert call_args[1]['trace'] is mock_trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
