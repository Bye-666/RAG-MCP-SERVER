"""
Unit tests for RRF Fusion.
"""

import pytest
from src.core.query_engine.fusion import RRFFusion
from src.core.types import RetrievalResult


class TestRRFFusion:
    """Test suite for RRFFusion class."""

    def test_basic_fusion(self):
        """Test basic RRF fusion with non-overlapping results."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=0.8, text="text2", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk3", score=10.0, text="text3", metadata={}),
            RetrievalResult(chunk_id="chunk4", score=8.0, text="text4", metadata={}),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # All chunks should be present
        assert len(fused) == 4
        chunk_ids = [r.chunk_id for r in fused]
        assert set(chunk_ids) == {"chunk1", "chunk2", "chunk3", "chunk4"}

        # Scores should be RRF scores
        for result in fused:
            assert 0 < result.score < 1  # RRF scores are typically small

    def test_overlapping_chunks(self):
        """Test RRF fusion with overlapping chunk IDs."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="dense_text1", metadata={"source": "dense"}),
            RetrievalResult(chunk_id="chunk2", score=0.8, text="dense_text2", metadata={"source": "dense"}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk2", score=10.0, text="sparse_text2", metadata={"source": "sparse"}),
            RetrievalResult(chunk_id="chunk3", score=8.0, text="sparse_text3", metadata={"source": "sparse"}),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # Should have 3 unique chunks
        assert len(fused) == 3
        chunk_ids = [r.chunk_id for r in fused]
        assert set(chunk_ids) == {"chunk1", "chunk2", "chunk3"}

        # chunk2 should have higher RRF score (appears in both lists)
        chunk2_result = next(r for r in fused if r.chunk_id == "chunk2")
        chunk1_result = next(r for r in fused if r.chunk_id == "chunk1")
        chunk3_result = next(r for r in fused if r.chunk_id == "chunk3")

        # chunk2 appears in both lists, so should have highest score
        assert chunk2_result.score > chunk1_result.score
        assert chunk2_result.score > chunk3_result.score

        # Should prefer dense result metadata for overlapping chunks
        assert chunk2_result.metadata == {"source": "dense"}

    def test_rrf_score_calculation(self):
        """Test RRF score calculation correctness."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk1", score=10.0, text="text1", metadata={}),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # chunk1 is rank 0 in both lists
        # RRF score = 1/(60+0+1) + 1/(60+0+1) = 1/61 + 1/61 = 2/61
        expected_score = 2.0 / 61.0
        assert len(fused) == 1
        assert abs(fused[0].score - expected_score) < 1e-9

    def test_custom_k_parameter(self):
        """Test RRF fusion with custom k parameter."""
        fusion_k10 = RRFFusion(k=10)
        fusion_k100 = RRFFusion(k=100)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk2", score=10.0, text="text2", metadata={}),
        ]

        fused_k10 = fusion_k10.fuse(dense_results, sparse_results)
        fused_k100 = fusion_k100.fuse(dense_results, sparse_results)

        # Smaller k gives higher scores
        assert fused_k10[0].score > fused_k100[0].score

    def test_invalid_k_parameter(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be non-negative"):
            RRFFusion(k=-1)

    def test_empty_dense_results(self):
        """Test fusion with empty dense results."""
        fusion = RRFFusion(k=60)

        dense_results = []
        sparse_results = [
            RetrievalResult(chunk_id="chunk1", score=10.0, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=8.0, text="text2", metadata={}),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused) == 2
        assert fused[0].chunk_id == "chunk1"
        assert fused[1].chunk_id == "chunk2"

    def test_empty_sparse_results(self):
        """Test fusion with empty sparse results."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=0.8, text="text2", metadata={}),
        ]
        sparse_results = []

        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused) == 2
        assert fused[0].chunk_id == "chunk1"
        assert fused[1].chunk_id == "chunk2"

    def test_both_empty(self):
        """Test fusion with both empty lists."""
        fusion = RRFFusion(k=60)

        fused = fusion.fuse([], [])

        assert len(fused) == 0

    def test_result_ordering(self):
        """Test that results are ordered by RRF score (descending)."""
        fusion = RRFFusion(k=60)

        # chunk1 appears in both lists at rank 0
        # chunk2 appears only in dense at rank 1
        # chunk3 appears only in sparse at rank 1
        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=0.8, text="text2", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk1", score=10.0, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk3", score=8.0, text="text3", metadata={}),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # chunk1 should be first (appears in both at rank 0)
        assert fused[0].chunk_id == "chunk1"

        # Verify descending order
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score

    def test_deterministic_output(self):
        """Test that fusion produces deterministic output."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(chunk_id="chunk1", score=0.9, text="text1", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=0.8, text="text2", metadata={}),
        ]

        sparse_results = [
            RetrievalResult(chunk_id="chunk3", score=10.0, text="text3", metadata={}),
            RetrievalResult(chunk_id="chunk2", score=8.0, text="text2", metadata={}),
        ]

        # Run fusion multiple times
        fused1 = fusion.fuse(dense_results, sparse_results)
        fused2 = fusion.fuse(dense_results, sparse_results)
        fused3 = fusion.fuse(dense_results, sparse_results)

        # Results should be identical
        assert len(fused1) == len(fused2) == len(fused3)
        for r1, r2, r3 in zip(fused1, fused2, fused3):
            assert r1.chunk_id == r2.chunk_id == r3.chunk_id
            assert abs(r1.score - r2.score) < 1e-9
            assert abs(r1.score - r3.score) < 1e-9

    def test_text_and_metadata_preserved(self):
        """Test that text and metadata are preserved in fusion."""
        fusion = RRFFusion(k=60)

        dense_results = [
            RetrievalResult(
                chunk_id="chunk1",
                score=0.9,
                text="This is chunk 1 text",
                metadata={"source": "doc1.pdf", "page": 1}
            ),
        ]

        sparse_results = [
            RetrievalResult(
                chunk_id="chunk2",
                score=10.0,
                text="This is chunk 2 text",
                metadata={"source": "doc2.pdf", "page": 2}
            ),
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # Find each chunk and verify text/metadata
        chunk1 = next(r for r in fused if r.chunk_id == "chunk1")
        chunk2 = next(r for r in fused if r.chunk_id == "chunk2")

        assert chunk1.text == "This is chunk 1 text"
        assert chunk1.metadata == {"source": "doc1.pdf", "page": 1}

        assert chunk2.text == "This is chunk 2 text"
        assert chunk2.metadata == {"source": "doc2.pdf", "page": 2}

    def test_large_ranking_difference(self):
        """Test RRF with large ranking differences."""
        fusion = RRFFusion(k=60)

        # Create long lists with one overlapping chunk at different positions
        dense_results = [
            RetrievalResult(chunk_id=f"chunk{i}", score=1.0 - i * 0.01, text=f"text{i}", metadata={})
            for i in range(100)
        ]

        sparse_results = [
            RetrievalResult(chunk_id=f"chunk{i}", score=100.0 - i, text=f"text{i}", metadata={})
            for i in range(50, 150)
        ]

        fused = fusion.fuse(dense_results, sparse_results)

        # Should have 150 unique chunks (0-149)
        assert len(fused) == 150

        # Overlapping chunks (50-99) should rank higher
        # chunk50 appears at rank 50 in dense and rank 0 in sparse
        chunk50 = next(r for r in fused if r.chunk_id == "chunk50")
        chunk0 = next(r for r in fused if r.chunk_id == "chunk0")
        chunk149 = next(r for r in fused if r.chunk_id == "chunk149")

        # chunk50 should rank higher than chunks that appear in only one list
        assert chunk50.score > chunk0.score
        assert chunk50.score > chunk149.score
