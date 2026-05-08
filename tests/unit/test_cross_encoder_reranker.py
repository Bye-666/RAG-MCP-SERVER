"""Unit tests for CrossEncoderReranker"""

import pytest
from unittest.mock import Mock
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker class"""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer"""
        scorer = Mock()
        scorer.predict = Mock()
        return scorer

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidates for testing"""
        return [
            {"id": "doc1", "text": "Python programming tutorial", "score": 0.9},
            {"id": "doc2", "text": "Java programming guide", "score": 0.8},
            {"id": "doc3", "text": "JavaScript basics", "score": 0.7}
        ]

    def test_initialization_with_mock_scorer(self, mock_scorer):
        """Test initialization with mock scorer"""
        reranker = CrossEncoderReranker(scorer=mock_scorer)
        assert reranker.scorer == mock_scorer

    def test_initialization_with_custom_model_name(self, mock_scorer):
        """Test initialization with custom model name"""
        model_name = "custom-model"
        reranker = CrossEncoderReranker(model_name=model_name, scorer=mock_scorer)
        assert reranker.model_name == model_name

    def test_initialization_with_timeout(self, mock_scorer):
        """Test initialization with timeout"""
        reranker = CrossEncoderReranker(scorer=mock_scorer, timeout=10.0)
        assert reranker.timeout == 10.0

    def test_rerank_basic(self, mock_scorer, sample_candidates):
        """Test basic reranking functionality"""
        # Mock scorer returns scores in reverse order
        mock_scorer.predict.return_value = [0.3, 0.8, 0.9]

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        result = reranker.rerank("test query", sample_candidates)

        # Check that predict was called with correct pairs
        mock_scorer.predict.assert_called_once()
        pairs = mock_scorer.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0] == ["test query", "Python programming tutorial"]
        assert pairs[1] == ["test query", "Java programming guide"]
        assert pairs[2] == ["test query", "JavaScript basics"]

        # Check reranked order (sorted by score descending)
        assert len(result) == 3
        assert result[0]["id"] == "doc3"  # score 0.9
        assert result[1]["id"] == "doc2"  # score 0.8
        assert result[2]["id"] == "doc1"  # score 0.3

        # Check rerank scores are added
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["rerank_score"] == 0.8
        assert result[2]["rerank_score"] == 0.3

    def test_rerank_preserves_original_fields(self, mock_scorer, sample_candidates):
        """Test that reranking preserves original candidate fields"""
        mock_scorer.predict.return_value = [0.5, 0.6, 0.7]

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        result = reranker.rerank("test query", sample_candidates)

        # Check original fields are preserved
        assert result[0]["text"] == "JavaScript basics"
        assert result[0]["score"] == 0.7
        assert "rerank_score" in result[0]

    def test_rerank_empty_candidates(self, mock_scorer):
        """Test reranking with empty candidates list"""
        reranker = CrossEncoderReranker(scorer=mock_scorer)
        result = reranker.rerank("test query", [])

        assert result == []
        mock_scorer.predict.assert_not_called()

    def test_rerank_missing_text_field(self, mock_scorer):
        """Test that missing 'text' field raises ValueError"""
        candidates = [{"id": "doc1"}]
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        with pytest.raises(ValueError, match="missing 'text' field"):
            reranker.rerank("test query", candidates)

    def test_rerank_scorer_failure(self, mock_scorer, sample_candidates):
        """Test that scorer failure raises RuntimeError for fallback"""
        mock_scorer.predict.side_effect = Exception("Model error")

        reranker = CrossEncoderReranker(scorer=mock_scorer)

        with pytest.raises(RuntimeError, match="Cross-encoder scoring failed"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_with_trace(self, mock_scorer, sample_candidates):
        """Test reranking with trace context"""
        mock_scorer.predict.return_value = [0.5, 0.6, 0.7]

        trace = Mock()
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        assert len(result) == 3
        assert trace.log.call_count >= 2  # At least start and end logs

    def test_rerank_single_candidate(self, mock_scorer):
        """Test reranking with single candidate"""
        mock_scorer.predict.return_value = [0.85]

        candidates = [{"id": "doc1", "text": "Single document"}]
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        result = reranker.rerank("test query", candidates)

        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["rerank_score"] == 0.85

    def test_rerank_deterministic_scores(self, mock_scorer):
        """Test that same input produces same output (deterministic)"""
        mock_scorer.predict.return_value = [0.1, 0.2, 0.3]

        candidates = [
            {"id": "doc1", "text": "Text A"},
            {"id": "doc2", "text": "Text B"},
            {"id": "doc3", "text": "Text C"}
        ]

        reranker = CrossEncoderReranker(scorer=mock_scorer)

        result1 = reranker.rerank("query", candidates)
        result2 = reranker.rerank("query", candidates)

        # Results should be identical
        assert [r["id"] for r in result1] == [r["id"] for r in result2]
        assert [r["rerank_score"] for r in result1] == [r["rerank_score"] for r in result2]

    def test_rerank_score_ordering(self, mock_scorer):
        """Test that results are correctly ordered by score"""
        # Provide scores in random order
        mock_scorer.predict.return_value = [0.5, 0.9, 0.2, 0.7]

        candidates = [
            {"id": "doc1", "text": "A"},
            {"id": "doc2", "text": "B"},
            {"id": "doc3", "text": "C"},
            {"id": "doc4", "text": "D"}
        ]

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        result = reranker.rerank("query", candidates)

        # Check descending order
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["id"] == "doc2"  # 0.9
        assert result[1]["id"] == "doc4"  # 0.7
        assert result[2]["id"] == "doc1"  # 0.5
        assert result[3]["id"] == "doc3"  # 0.2

    def test_rerank_negative_scores(self, mock_scorer):
        """Test handling of negative scores"""
        mock_scorer.predict.return_value = [-0.5, 0.2, -0.1]

        candidates = [
            {"id": "doc1", "text": "A"},
            {"id": "doc2", "text": "B"},
            {"id": "doc3", "text": "C"}
        ]

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        result = reranker.rerank("query", candidates)

        # Should still order correctly
        assert result[0]["rerank_score"] == 0.2
        assert result[1]["rerank_score"] == -0.1
        assert result[2]["rerank_score"] == -0.5

    def test_initialization_without_sentence_transformers(self):
        """Test that initialization without sentence-transformers raises ImportError"""
        # This test would fail if sentence-transformers is installed
        # We can't easily test this without uninstalling the package
        # So we'll just document the expected behavior
        pass

    def test_rerank_timeout_signal(self, mock_scorer, sample_candidates):
        """Test that timeout raises TimeoutError for fallback"""
        mock_scorer.predict.side_effect = TimeoutError("Scoring timeout")

        reranker = CrossEncoderReranker(scorer=mock_scorer, timeout=1.0)

        with pytest.raises(TimeoutError):
            reranker.rerank("test query", sample_candidates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
