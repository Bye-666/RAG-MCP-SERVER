"""
Unit tests for Reranker.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.core.query_engine.reranker import Reranker, RerankResult
from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace import TraceContext


@pytest.fixture
def settings():
    """Create test settings"""
    settings = Mock(spec=Settings)
    settings.__dict__ = {
        'llm': {},
        'embedding': {},
        'vector_store': {},
        'retrieval': {},
        'observability': {},
        'reranker': {'provider': 'none'}
    }
    return settings


@pytest.fixture
def mock_backend():
    """Create mock reranker backend"""
    backend = Mock()
    backend.__class__.__name__ = "MockReranker"
    return backend


@pytest.fixture
def sample_candidates():
    """Create sample retrieval results"""
    return [
        RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="First result",
            metadata={"source": "doc1"}
        ),
        RetrievalResult(
            chunk_id="chunk2",
            score=0.8,
            text="Second result",
            metadata={"source": "doc2"}
        ),
        RetrievalResult(
            chunk_id="chunk3",
            score=0.7,
            text="Third result",
            metadata={"source": "doc3"}
        )
    ]


def test_reranker_basic(settings, mock_backend, sample_candidates):
    """Test basic reranking functionality"""
    # Mock backend returns reversed order
    mock_backend.rerank.return_value = [
        {"id": "chunk3", "score": 0.95, "text": "Third result", "metadata": {"source": "doc3"}},
        {"id": "chunk1", "score": 0.85, "text": "First result", "metadata": {"source": "doc1"}},
        {"id": "chunk2", "score": 0.75, "text": "Second result", "metadata": {"source": "doc2"}}
    ]

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates)

    assert isinstance(result, RerankResult)
    assert not result.fallback
    assert result.error is None
    assert len(result.results) == 3
    assert result.results[0].chunk_id == "chunk3"
    assert result.results[0].score == 0.95


def test_reranker_empty_query(settings, mock_backend, sample_candidates):
    """Test error handling for empty query"""
    reranker = Reranker(settings, reranker_backend=mock_backend)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank("", sample_candidates)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank("   ", sample_candidates)


def test_reranker_none_candidates(settings, mock_backend):
    """Test error handling for None candidates"""
    reranker = Reranker(settings, reranker_backend=mock_backend)

    with pytest.raises(ValueError, match="Candidates cannot be None"):
        reranker.rerank("test query", None)


def test_reranker_empty_candidates(settings, mock_backend):
    """Test handling of empty candidate list"""
    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", [])

    assert isinstance(result, RerankResult)
    assert not result.fallback
    assert len(result.results) == 0
    # Backend should not be called for empty list
    mock_backend.rerank.assert_not_called()


def test_reranker_fallback_on_error(settings, mock_backend, sample_candidates):
    """Test fallback mechanism when backend fails"""
    # Mock backend raises exception
    mock_backend.rerank.side_effect = RuntimeError("Reranker service unavailable")

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates)

    assert isinstance(result, RerankResult)
    assert result.fallback is True
    assert "Reranker service unavailable" in result.error
    # Should return original candidates
    assert len(result.results) == 3
    assert result.results[0].chunk_id == "chunk1"
    assert result.results[1].chunk_id == "chunk2"


def test_reranker_with_trace(settings, mock_backend, sample_candidates):
    """Test TraceContext integration"""
    mock_backend.rerank.return_value = [
        {"id": "chunk1", "score": 0.9, "text": "First result", "metadata": {"source": "doc1"}}
    ]

    trace = Mock(spec=TraceContext)
    stage_id = "stage_123"
    trace.record_stage.return_value = stage_id

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates, trace=trace)

    # Verify trace calls
    trace.record_stage.assert_called_once()
    call_args = trace.record_stage.call_args
    assert call_args[0][0] == "reranker"
    assert call_args[0][1]["candidate_count"] == 3
    assert call_args[0][1]["backend"] == "MockReranker"

    trace.finish_stage.assert_called_once()
    finish_args = trace.finish_stage.call_args
    assert finish_args[0][0] == stage_id
    assert finish_args[0][1]["success"] is True
    assert finish_args[0][1]["fallback"] is False


def test_reranker_trace_on_fallback(settings, mock_backend, sample_candidates):
    """Test TraceContext records fallback"""
    mock_backend.rerank.side_effect = RuntimeError("Backend error")

    trace = Mock(spec=TraceContext)
    stage_id = "stage_123"
    trace.record_stage.return_value = stage_id

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates, trace=trace)

    # Verify fallback is recorded in trace
    trace.finish_stage.assert_called_once()
    finish_args = trace.finish_stage.call_args
    assert finish_args[0][1]["success"] is False
    assert finish_args[0][1]["fallback"] is True
    assert "Backend error" in finish_args[0][1]["error"]


def test_reranker_format_conversion(settings, mock_backend, sample_candidates):
    """Test conversion between RetrievalResult and dict format"""
    # Capture what was passed to backend
    captured_candidates = None

    def capture_and_return(query, candidates, trace=None):
        nonlocal captured_candidates
        captured_candidates = candidates
        return candidates  # Return as-is

    mock_backend.rerank.side_effect = capture_and_return

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates)

    # Verify format passed to backend
    assert len(captured_candidates) == 3
    assert captured_candidates[0]["id"] == "chunk1"
    assert captured_candidates[0]["text"] == "First result"
    assert captured_candidates[0]["score"] == 0.9
    assert captured_candidates[0]["metadata"] == {"source": "doc1"}

    # Verify format returned to caller
    assert isinstance(result.results[0], RetrievalResult)
    assert result.results[0].chunk_id == "chunk1"


def test_reranker_preserves_metadata(settings, mock_backend, sample_candidates):
    """Test that metadata is preserved through reranking"""
    mock_backend.rerank.return_value = [
        {
            "id": "chunk1",
            "score": 0.95,
            "text": "First result",
            "metadata": {"source": "doc1", "page": 5}
        }
    ]

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates)

    assert result.results[0].metadata == {"source": "doc1", "page": 5}


def test_reranker_handles_missing_metadata(settings, mock_backend, sample_candidates):
    """Test handling when backend doesn't return metadata"""
    mock_backend.rerank.return_value = [
        {"id": "chunk1", "score": 0.95, "text": "First result"}
        # No metadata field
    ]

    reranker = Reranker(settings, reranker_backend=mock_backend)
    result = reranker.rerank("test query", sample_candidates)

    assert result.results[0].metadata == {}


def test_reranker_factory_integration(settings, sample_candidates):
    """Test integration with RerankerFactory (NoneReranker)"""
    # Don't inject backend - let it create from settings
    reranker = Reranker(settings)
    result = reranker.rerank("test query", sample_candidates)

    # NoneReranker should return original order
    assert not result.fallback
    assert len(result.results) == 3
    assert result.results[0].chunk_id == "chunk1"
    assert result.results[1].chunk_id == "chunk2"
    assert result.results[2].chunk_id == "chunk3"
