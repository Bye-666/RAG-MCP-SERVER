"""Integration tests for ChromaStore.

Tests the complete roundtrip: upsert → query → get_by_ids
Uses temporary directories for persistence and cleans up after tests.
"""

import pytest
import tempfile
import shutil
from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class TestChromaStoreRoundtrip:
    """Integration tests for ChromaStore with real persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def chroma_store(self, temp_dir):
        """Create a ChromaStore instance with temporary persistence."""
        store = ChromaStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        yield store
        # Cleanup collection
        store.delete_collection()

    def test_basic_upsert_and_query(self, chroma_store):
        """Test basic upsert and query roundtrip."""
        # Prepare test data
        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "This is document 1",
                "metadata": {"source": "test"}
            },
            {
                "id": "doc2",
                "vector": [0.4, 0.5, 0.6],
                "text": "This is document 2",
                "metadata": {"source": "test"}
            }
        ]

        # Upsert records
        ids = chroma_store.upsert(records)
        assert len(ids) == 2
        assert "doc1" in ids
        assert "doc2" in ids

        # Query with similar vector to doc1
        query_vector = [0.1, 0.2, 0.3]
        results = chroma_store.query(query_vector, top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"  # Most similar
        assert results[0]["text"] == "This is document 1"
        assert results[0]["metadata"]["source"] == "test"
        assert results[0]["score"] > results[1]["score"]

    def test_upsert_updates_existing(self, chroma_store):
        """Test that upsert updates existing records."""
        # Initial upsert
        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Original text",
                "metadata": {"version": 1}
            }
        ]
        chroma_store.upsert(records)

        # Update with same ID
        updated_records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Updated text",
                "metadata": {"version": 2}
            }
        ]
        chroma_store.upsert(updated_records)

        # Verify update
        results = chroma_store.get_by_ids(["doc1"])
        assert len(results) == 1
        assert results[0]["text"] == "Updated text"
        assert results[0]["metadata"]["version"] == 2

    def test_query_top_k(self, chroma_store):
        """Test that top_k parameter works correctly."""
        # Insert 5 documents
        records = [
            {
                "id": f"doc{i}",
                "vector": [i * 0.1, i * 0.2, i * 0.3],
                "text": f"Document {i}",
                "metadata": {}
            }
            for i in range(5)
        ]
        chroma_store.upsert(records)

        # Query with top_k=3
        query_vector = [0.0, 0.0, 0.0]
        results = chroma_store.query(query_vector, top_k=3)

        assert len(results) == 3
        # Verify all results have required fields
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "text" in result
            assert "metadata" in result

    def test_query_with_metadata_filters(self, chroma_store):
        """Test querying with metadata filters."""
        # Insert documents with different categories
        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Category A document",
                "metadata": {"category": "A"}
            },
            {
                "id": "doc2",
                "vector": [0.1, 0.2, 0.3],
                "text": "Category B document",
                "metadata": {"category": "B"}
            },
            {
                "id": "doc3",
                "vector": [0.1, 0.2, 0.3],
                "text": "Another Category A document",
                "metadata": {"category": "A"}
            }
        ]
        chroma_store.upsert(records)

        # Query with filter for category A
        query_vector = [0.1, 0.2, 0.3]
        results = chroma_store.query(
            query_vector,
            top_k=10,
            filters={"category": "A"}
        )

        assert len(results) == 2
        assert all(r["metadata"]["category"] == "A" for r in results)

    def test_get_by_ids(self, chroma_store):
        """Test retrieving records by IDs."""
        # Insert test data
        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Document 1",
                "metadata": {"index": 1}
            },
            {
                "id": "doc2",
                "vector": [0.4, 0.5, 0.6],
                "text": "Document 2",
                "metadata": {"index": 2}
            },
            {
                "id": "doc3",
                "vector": [0.7, 0.8, 0.9],
                "text": "Document 3",
                "metadata": {"index": 3}
            }
        ]
        chroma_store.upsert(records)

        # Get specific IDs
        results = chroma_store.get_by_ids(["doc1", "doc3"])

        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "doc1" in ids
        assert "doc3" in ids
        assert "doc2" not in ids

    def test_get_by_ids_nonexistent(self, chroma_store):
        """Test getting non-existent IDs returns empty list."""
        results = chroma_store.get_by_ids(["nonexistent"])
        assert len(results) == 0

    def test_empty_upsert(self, chroma_store):
        """Test that empty upsert returns empty list."""
        ids = chroma_store.upsert([])
        assert ids == []

    def test_upsert_missing_fields(self, chroma_store):
        """Test that upsert validates required fields."""
        # Missing 'id'
        with pytest.raises(ValueError, match="missing required 'id' field"):
            chroma_store.upsert([{"vector": [0.1], "text": "test"}])

        # Missing 'vector'
        with pytest.raises(ValueError, match="missing required 'vector' field"):
            chroma_store.upsert([{"id": "1", "text": "test"}])

        # Missing 'text'
        with pytest.raises(ValueError, match="missing required 'text' field"):
            chroma_store.upsert([{"id": "1", "vector": [0.1]}])

    def test_query_invalid_inputs(self, chroma_store):
        """Test that query validates inputs."""
        # Non-list vector
        with pytest.raises(TypeError, match="vector must be a list"):
            chroma_store.query("not a list", top_k=5)

        # Empty vector
        with pytest.raises(ValueError, match="vector cannot be empty"):
            chroma_store.query([], top_k=5)

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            chroma_store.query([0.1, 0.2], top_k=0)

    def test_persistence(self, temp_dir):
        """Test that data persists across ChromaStore instances."""
        # Create first instance and insert data
        store1 = ChromaStore(
            collection_name="persist_test",
            persist_directory=temp_dir
        )
        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Persistent document",
                "metadata": {"persisted": True}
            }
        ]
        store1.upsert(records)
        del store1

        # Create second instance with same directory
        store2 = ChromaStore(
            collection_name="persist_test",
            persist_directory=temp_dir
        )

        # Verify data persisted
        results = store2.get_by_ids(["doc1"])
        assert len(results) == 1
        assert results[0]["text"] == "Persistent document"
        assert results[0]["metadata"]["persisted"] is True

        store2.delete_collection()

    def test_factory_integration(self, temp_dir):
        """Test that ChromaStore can be created via factory."""
        settings = {
            "vector_store": {
                "provider": "chroma",
                "collection_name": "factory_test",
                "persist_directory": temp_dir
            }
        }

        store = VectorStoreFactory.create(settings)

        assert isinstance(store, ChromaStore)
        assert store.collection_name == "factory_test"

        # Test basic functionality
        records = [{"id": "1", "vector": [0.1], "text": "test", "metadata": {}}]
        ids = store.upsert(records)
        assert len(ids) == 1

        store.delete_collection()

    def test_count(self, chroma_store):
        """Test counting records in collection."""
        assert chroma_store.count() == 0

        records = [
            {"id": f"doc{i}", "vector": [i * 0.1], "text": f"Doc {i}", "metadata": {}}
            for i in range(5)
        ]
        chroma_store.upsert(records)

        assert chroma_store.count() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
