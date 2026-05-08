"""Unit tests for core data types"""

import pytest
import json
from src.core.types import Document, Chunk, ChunkRecord


class TestDocument:
    """Test cases for Document class"""

    def test_document_creation_success(self):
        """Test successful document creation"""
        doc = Document(
            id="doc1",
            text="This is a test document",
            metadata={"source_path": "/path/to/doc.pdf"}
        )
        assert doc.id == "doc1"
        assert doc.text == "This is a test document"
        assert doc.metadata["source_path"] == "/path/to/doc.pdf"

    def test_document_missing_source_path(self):
        """Test that missing source_path raises ValueError"""
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Document(
                id="doc1",
                text="Test",
                metadata={"other_field": "value"}
            )

    def test_document_with_additional_metadata(self):
        """Test document with additional metadata fields"""
        doc = Document(
            id="doc1",
            text="Test",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "author": "John Doe",
                "created_at": "2024-01-01",
                "page_count": 10
            }
        )
        assert doc.metadata["author"] == "John Doe"
        assert doc.metadata["page_count"] == 10

    def test_document_with_images_metadata(self):
        """Test document with images metadata"""
        doc = Document(
            id="doc1",
            text="Test [IMAGE: img1]",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    {
                        "id": "abc123_1_0",
                        "path": "data/images/default/abc123_1_0.png",
                        "page": 1,
                        "text_offset": 5,
                        "text_length": 14,
                        "position": {"x": 100, "y": 200}
                    }
                ]
            }
        )
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0]["id"] == "abc123_1_0"

    def test_document_to_dict(self):
        """Test document serialization to dict"""
        doc = Document(
            id="doc1",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "doc1"
        assert doc_dict["text"] == "Test"
        assert doc_dict["metadata"]["source_path"] == "/path/to/doc.pdf"

    def test_document_to_json(self):
        """Test document serialization to JSON"""
        doc = Document(
            id="doc1",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "doc1"
        assert parsed["text"] == "Test"

    def test_document_from_dict(self):
        """Test document deserialization from dict"""
        data = {
            "id": "doc1",
            "text": "Test",
            "metadata": {"source_path": "/path/to/doc.pdf"}
        }
        doc = Document.from_dict(data)
        assert doc.id == "doc1"
        assert doc.text == "Test"

    def test_document_from_json(self):
        """Test document deserialization from JSON"""
        json_str = '{"id": "doc1", "text": "Test", "metadata": {"source_path": "/path/to/doc.pdf"}}'
        doc = Document.from_json(json_str)
        assert doc.id == "doc1"
        assert doc.text == "Test"

    def test_document_roundtrip_serialization(self):
        """Test document serialization roundtrip"""
        original = Document(
            id="doc1",
            text="Test document",
            metadata={"source_path": "/path/to/doc.pdf", "author": "John"}
        )
        json_str = original.to_json()
        restored = Document.from_json(json_str)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata


class TestChunk:
    """Test cases for Chunk class"""

    def test_chunk_creation_success(self):
        """Test successful chunk creation"""
        chunk = Chunk(
            id="doc1_0001_abc",
            text="This is a chunk",
            metadata={"source_path": "/path/to/doc.pdf"},
            start_offset=0,
            end_offset=15,
            source_ref="doc1"
        )
        assert chunk.id == "doc1_0001_abc"
        assert chunk.text == "This is a chunk"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 15
        assert chunk.source_ref == "doc1"

    def test_chunk_empty_text(self):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError, match="text cannot be empty"):
            Chunk(id="chunk1", text="", metadata={})

    def test_chunk_whitespace_only_text(self):
        """Test that whitespace-only text raises ValueError"""
        with pytest.raises(ValueError, match="text cannot be empty"):
            Chunk(id="chunk1", text="   ", metadata={})

    def test_chunk_invalid_offsets(self):
        """Test that invalid offsets raise ValueError"""
        with pytest.raises(ValueError, match="end_offset must be >= start_offset"):
            Chunk(
                id="chunk1",
                text="Test",
                metadata={},
                start_offset=10,
                end_offset=5
            )

    def test_chunk_default_values(self):
        """Test chunk with default values"""
        chunk = Chunk(id="chunk1", text="Test")
        assert chunk.metadata == {}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 0
        assert chunk.source_ref is None

    def test_chunk_to_dict(self):
        """Test chunk serialization to dict"""
        chunk = Chunk(
            id="chunk1",
            text="Test",
            metadata={"key": "value"},
            start_offset=0,
            end_offset=4
        )
        chunk_dict = chunk.to_dict()
        assert chunk_dict["id"] == "chunk1"
        assert chunk_dict["text"] == "Test"
        assert chunk_dict["start_offset"] == 0

    def test_chunk_to_json(self):
        """Test chunk serialization to JSON"""
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        json_str = chunk.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "chunk1"

    def test_chunk_from_dict(self):
        """Test chunk deserialization from dict"""
        data = {
            "id": "chunk1",
            "text": "Test",
            "metadata": {},
            "start_offset": 0,
            "end_offset": 4,
            "source_ref": "doc1"
        }
        chunk = Chunk.from_dict(data)
        assert chunk.id == "chunk1"
        assert chunk.source_ref == "doc1"

    def test_chunk_from_json(self):
        """Test chunk deserialization from JSON"""
        json_str = '{"id": "chunk1", "text": "Test", "metadata": {}, "start_offset": 0, "end_offset": 4, "source_ref": null}'
        chunk = Chunk.from_json(json_str)
        assert chunk.id == "chunk1"
        assert chunk.text == "Test"

    def test_chunk_roundtrip_serialization(self):
        """Test chunk serialization roundtrip"""
        original = Chunk(
            id="chunk1",
            text="Test chunk",
            metadata={"key": "value"},
            start_offset=10,
            end_offset=20,
            source_ref="doc1"
        )
        json_str = original.to_json()
        restored = Chunk.from_json(json_str)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset


class TestChunkRecord:
    """Test cases for ChunkRecord class"""

    def test_chunk_record_creation_success(self):
        """Test successful chunk record creation"""
        record = ChunkRecord(
            id="chunk1",
            text="Test chunk",
            metadata={"key": "value"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word1": 0.5, "word2": 0.3}
        )
        assert record.id == "chunk1"
        assert record.text == "Test chunk"
        assert len(record.dense_vector) == 3
        assert record.sparse_vector["word1"] == 0.5

    def test_chunk_record_empty_text(self):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError, match="text cannot be empty"):
            ChunkRecord(id="chunk1", text="", metadata={})

    def test_chunk_record_default_vectors(self):
        """Test chunk record with default None vectors"""
        record = ChunkRecord(id="chunk1", text="Test", metadata={})
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_chunk_record_to_dict(self):
        """Test chunk record serialization to dict"""
        record = ChunkRecord(
            id="chunk1",
            text="Test",
            metadata={},
            dense_vector=[0.1, 0.2],
            sparse_vector={"word": 0.5}
        )
        record_dict = record.to_dict()
        assert record_dict["id"] == "chunk1"
        assert record_dict["dense_vector"] == [0.1, 0.2]
        assert record_dict["sparse_vector"] == {"word": 0.5}

    def test_chunk_record_to_json(self):
        """Test chunk record serialization to JSON"""
        record = ChunkRecord(
            id="chunk1",
            text="Test",
            metadata={},
            dense_vector=[0.1, 0.2]
        )
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["dense_vector"] == [0.1, 0.2]

    def test_chunk_record_from_dict(self):
        """Test chunk record deserialization from dict"""
        data = {
            "id": "chunk1",
            "text": "Test",
            "metadata": {},
            "dense_vector": [0.1, 0.2],
            "sparse_vector": None
        }
        record = ChunkRecord.from_dict(data)
        assert record.id == "chunk1"
        assert record.dense_vector == [0.1, 0.2]

    def test_chunk_record_from_json(self):
        """Test chunk record deserialization from JSON"""
        json_str = '{"id": "chunk1", "text": "Test", "metadata": {}, "dense_vector": [0.1], "sparse_vector": null}'
        record = ChunkRecord.from_json(json_str)
        assert record.id == "chunk1"
        assert record.dense_vector == [0.1]

    def test_chunk_record_from_chunk(self):
        """Test creating ChunkRecord from Chunk"""
        chunk = Chunk(
            id="chunk1",
            text="Test chunk",
            metadata={"key": "value"},
            start_offset=0,
            end_offset=10
        )
        record = ChunkRecord.from_chunk(chunk)
        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.metadata == chunk.metadata
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_chunk_record_from_chunk_with_vectors(self):
        """Test creating ChunkRecord from Chunk with vectors"""
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"word": 0.5}

        record = ChunkRecord.from_chunk(chunk, dense_vec, sparse_vec)
        assert record.id == chunk.id
        assert record.dense_vector == dense_vec
        assert record.sparse_vector == sparse_vec

    def test_chunk_record_from_chunk_metadata_copy(self):
        """Test that from_chunk copies metadata (not reference)"""
        chunk = Chunk(id="chunk1", text="Test", metadata={"key": "value"})
        record = ChunkRecord.from_chunk(chunk)

        # Modify record metadata
        record.metadata["new_key"] = "new_value"

        # Original chunk metadata should be unchanged
        assert "new_key" not in chunk.metadata

    def test_chunk_record_roundtrip_serialization(self):
        """Test chunk record serialization roundtrip"""
        original = ChunkRecord(
            id="chunk1",
            text="Test",
            metadata={"key": "value"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word1": 0.5, "word2": 0.3}
        )
        json_str = original.to_json()
        restored = ChunkRecord.from_json(json_str)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector


class TestDataTypeIntegration:
    """Integration tests for data type workflow"""

    def test_document_to_chunk_workflow(self):
        """Test typical workflow: Document -> Chunk"""
        # Create document
        doc = Document(
            id="doc1",
            text="This is a test document with multiple sentences.",
            metadata={"source_path": "/path/to/doc.pdf", "author": "John"}
        )

        # Create chunk from document
        chunk = Chunk(
            id=f"{doc.id}_0001_abc",
            text="This is a test document",
            metadata=doc.metadata.copy(),
            start_offset=0,
            end_offset=23,
            source_ref=doc.id
        )

        assert chunk.source_ref == doc.id
        assert chunk.metadata["source_path"] == doc.metadata["source_path"]

    def test_chunk_to_chunk_record_workflow(self):
        """Test typical workflow: Chunk -> ChunkRecord"""
        # Create chunk
        chunk = Chunk(
            id="doc1_0001_abc",
            text="Test chunk",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        # Create chunk record with embeddings
        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"test": 0.5, "chunk": 0.3}
        record = ChunkRecord.from_chunk(chunk, dense_vec, sparse_vec)

        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.dense_vector is not None
        assert record.sparse_vector is not None

    def test_full_pipeline_serialization(self):
        """Test full pipeline with serialization"""
        # Document
        doc = Document(
            id="doc1",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )
        doc_json = doc.to_json()
        doc_restored = Document.from_json(doc_json)

        # Chunk
        chunk = Chunk(
            id=f"{doc_restored.id}_0001",
            text=doc_restored.text,
            metadata=doc_restored.metadata.copy(),
            source_ref=doc_restored.id
        )
        chunk_json = chunk.to_json()
        chunk_restored = Chunk.from_json(chunk_json)

        # ChunkRecord
        record = ChunkRecord.from_chunk(chunk_restored, [0.1, 0.2])
        record_json = record.to_json()
        record_restored = ChunkRecord.from_json(record_json)

        assert record_restored.id == chunk.id
        assert record_restored.text == doc.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
