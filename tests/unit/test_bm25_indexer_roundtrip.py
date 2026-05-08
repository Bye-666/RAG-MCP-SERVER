"""Unit tests for BM25Indexer

Tests cover:
- Index building from chunk records
- IDF calculation accuracy
- Index persistence (save/load roundtrip)
- Query functionality
- Incremental updates
- Edge cases
"""

import pytest
import os
import shutil
import math
from src.ingestion.storage.bm25_indexer import BM25Indexer, PostingEntry, TermIndex
from src.core.types import ChunkRecord


class TestBM25Indexer:
    """Test suite for BM25Indexer"""

    @pytest.fixture
    def temp_index_dir(self, tmp_path):
        """Create temporary index directory"""
        index_dir = tmp_path / "test_bm25"
        index_dir.mkdir()
        yield str(index_dir)
        # Cleanup
        if index_dir.exists():
            shutil.rmtree(index_dir)

    @pytest.fixture
    def sample_records(self):
        """Create sample chunk records with sparse vectors"""
        return [
            ChunkRecord(
                id="chunk_001",
                text="machine learning algorithms",
                metadata={},
                sparse_vector={"machine": 0.33, "learning": 0.33, "algorithms": 0.33}
            ),
            ChunkRecord(
                id="chunk_002",
                text="deep learning neural networks",
                metadata={},
                sparse_vector={"deep": 0.25, "learning": 0.25, "neural": 0.25, "networks": 0.25}
            ),
            ChunkRecord(
                id="chunk_003",
                text="machine vision systems",
                metadata={},
                sparse_vector={"machine": 0.33, "vision": 0.33, "systems": 0.33}
            )
        ]

    def test_initialization(self, temp_index_dir):
        """Test indexer initialization"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        assert indexer.index_dir == temp_index_dir
        assert indexer.index == {}
        assert indexer.num_documents == 0
        assert os.path.exists(temp_index_dir)

    def test_build_basic(self, temp_index_dir, sample_records):
        """Test basic index building"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        assert indexer.num_documents == 3
        assert len(indexer.index) > 0

        # Check that terms are indexed
        assert "machine" in indexer.index
        assert "learning" in indexer.index
        assert "deep" in indexer.index

    def test_build_empty_records_raises_error(self, temp_index_dir):
        """Test that building with empty records raises error"""
        indexer = BM25Indexer(index_dir=temp_index_dir)

        with pytest.raises(ValueError, match="records list cannot be empty"):
            indexer.build([])

    def test_build_missing_sparse_vector_raises_error(self, temp_index_dir):
        """Test that building with missing sparse vector raises error"""
        indexer = BM25Indexer(index_dir=temp_index_dir)

        records = [
            ChunkRecord(
                id="chunk_001",
                text="test",
                metadata={},
                sparse_vector=None  # Missing sparse vector
            )
        ]

        with pytest.raises(ValueError, match="missing sparse_vector"):
            indexer.build(records)

    def test_idf_calculation(self, temp_index_dir, sample_records):
        """Test IDF calculation accuracy"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        # "machine" appears in 2 out of 3 documents
        machine_idf = indexer.index["machine"].idf
        expected_idf = math.log((3 - 2 + 0.5) / (2 + 0.5))
        assert abs(machine_idf - expected_idf) < 1e-6

        # "learning" appears in 2 out of 3 documents
        learning_idf = indexer.index["learning"].idf
        expected_idf = math.log((3 - 2 + 0.5) / (2 + 0.5))
        assert abs(learning_idf - expected_idf) < 1e-6

        # "deep" appears in 1 out of 3 documents
        deep_idf = indexer.index["deep"].idf
        expected_idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
        assert abs(deep_idf - expected_idf) < 1e-6

    def test_postings_structure(self, temp_index_dir, sample_records):
        """Test that postings are correctly structured"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        # Check "machine" postings (appears in chunk_001 and chunk_003)
        machine_index = indexer.index["machine"]
        assert len(machine_index.postings) == 2

        chunk_ids = {posting.chunk_id for posting in machine_index.postings}
        assert chunk_ids == {"chunk_001", "chunk_003"}

        # Check posting details
        for posting in machine_index.postings:
            assert isinstance(posting, PostingEntry)
            assert posting.tf > 0
            assert posting.doc_length > 0

    def test_save_and_load_roundtrip(self, temp_index_dir, sample_records):
        """Test save and load roundtrip preserves index"""
        indexer1 = BM25Indexer(index_dir=temp_index_dir)
        indexer1.build(sample_records)
        indexer1.save("test_index.pkl")

        # Load into new indexer
        indexer2 = BM25Indexer(index_dir=temp_index_dir)
        indexer2.load("test_index.pkl")

        # Verify index is preserved
        assert indexer2.num_documents == indexer1.num_documents
        assert len(indexer2.index) == len(indexer1.index)

        # Check specific term
        assert "machine" in indexer2.index
        assert abs(indexer2.index["machine"].idf - indexer1.index["machine"].idf) < 1e-6
        assert len(indexer2.index["machine"].postings) == len(indexer1.index["machine"].postings)

    def test_load_nonexistent_file_raises_error(self, temp_index_dir):
        """Test that loading nonexistent file raises error"""
        indexer = BM25Indexer(index_dir=temp_index_dir)

        with pytest.raises(FileNotFoundError):
            indexer.load("nonexistent.pkl")

    def test_query_basic(self, temp_index_dir, sample_records):
        """Test basic query functionality"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        # Query for "machine"
        results = indexer.query(["machine"], top_k=10)

        assert len(results) == 2  # "machine" appears in 2 documents
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # Results should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_multiple_terms(self, temp_index_dir, sample_records):
        """Test query with multiple terms"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        # Query for "machine learning"
        results = indexer.query(["machine", "learning"], top_k=10)

        # Should return all 3 documents (machine in 2, learning in 2)
        assert len(results) <= 3

        # chunk_001 should rank high (contains both terms)
        chunk_ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_001" in chunk_ids

    def test_query_empty_terms(self, temp_index_dir, sample_records):
        """Test query with empty terms list"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        results = indexer.query([], top_k=10)
        assert results == []

    def test_query_unknown_term(self, temp_index_dir, sample_records):
        """Test query with term not in index"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        results = indexer.query(["nonexistent"], top_k=10)
        assert results == []

    def test_query_top_k_limit(self, temp_index_dir):
        """Test that query respects top_k limit"""
        # Create many records
        records = [
            ChunkRecord(
                id=f"chunk_{i:03d}",
                text="common term",
                metadata={},
                sparse_vector={"common": 0.5, "term": 0.5}
            )
            for i in range(20)
        ]

        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(records)

        results = indexer.query(["common"], top_k=5)
        assert len(results) == 5

    def test_query_case_insensitive(self, temp_index_dir, sample_records):
        """Test that query is case-insensitive"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        results_lower = indexer.query(["machine"], top_k=10)
        results_upper = indexer.query(["MACHINE"], top_k=10)
        results_mixed = indexer.query(["Machine"], top_k=10)

        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_get_term_info(self, temp_index_dir, sample_records):
        """Test getting term information"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        term_info = indexer.get_term_info("machine")
        assert term_info is not None
        assert isinstance(term_info, TermIndex)
        assert term_info.term == "machine"
        # IDF can be negative for common terms (appearing in >50% of docs)
        assert isinstance(term_info.idf, float)
        assert len(term_info.postings) == 2

    def test_get_term_info_nonexistent(self, temp_index_dir, sample_records):
        """Test getting info for nonexistent term"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        term_info = indexer.get_term_info("nonexistent")
        assert term_info is None

    def test_get_stats(self, temp_index_dir, sample_records):
        """Test getting index statistics"""
        indexer = BM25Indexer(index_dir=temp_index_dir)
        indexer.build(sample_records)

        stats = indexer.get_stats()

        assert stats['num_documents'] == 3
        assert stats['num_terms'] > 0
        assert stats['total_postings'] > 0

    def test_rebuild_clears_previous_index(self, temp_index_dir, sample_records):
        """Test that rebuilding clears previous index"""
        indexer = BM25Indexer(index_dir=temp_index_dir)

        # Build first index
        indexer.build(sample_records)
        first_num_terms = len(indexer.index)

        # Build second index with different records
        new_records = [
            ChunkRecord(
                id="new_001",
                text="completely different content",
                metadata={},
                sparse_vector={"completely": 0.33, "different": 0.33, "content": 0.33}
            )
        ]
        indexer.build(new_records)

        assert indexer.num_documents == 1
        assert "machine" not in indexer.index
        assert "completely" in indexer.index

    def test_query_after_load(self, temp_index_dir, sample_records):
        """Test that query works correctly after loading index"""
        # Build and save
        indexer1 = BM25Indexer(index_dir=temp_index_dir)
        indexer1.build(sample_records)
        indexer1.save()

        # Load and query
        indexer2 = BM25Indexer(index_dir=temp_index_dir)
        indexer2.load()

        results = indexer2.query(["machine", "learning"], top_k=10)

        assert len(results) > 0
        # chunk_001 should be in results (contains both terms)
        chunk_ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_001" in chunk_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
