"""
Integration tests for Reranker with trace support.
"""

import pytest
import json
from unittest.mock import Mock

from src.core.query_engine.reranker import Reranker, RerankResult
from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.libs.reranker.base_reranker import BaseReranker


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def mock_reranker_backend():
    """Create mock reranker backend"""
    backend = Mock(spec=BaseReranker)
    backend.rerank = Mock(return_value=[
        {"id": "chunk2", "score": 0.95, "text": "Most relevant", "metadata": {}},
        {"id": "chunk1", "score": 0.85, "text": "Second relevant", "metadata": {}},
        {"id": "chunk3", "score": 0.75, "text": "Third relevant", "metadata": {}},
    ])
    return backend


@pytest.fixture
def sample_candidates():
    """Create sample candidate results"""
    return [
        RetrievalResult(chunk_id="chunk1", score=0.9, text="First result", metadata={"source": "doc1"}),
        RetrievalResult(chunk_id="chunk2", score=0.8, text="Second result", metadata={"source": "doc2"}),
        RetrievalResult(chunk_id="chunk3", score=0.7, text="Third result", metadata={"source": "doc3"}),
    ]


class TestRerankerWithTrace:
    """Test Reranker with trace support"""

    def test_rerank_with_trace(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test reranking with trace context"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)
        trace.finish()

        # Verify reranking succeeded
        assert not result.fallback
        assert len(result.results) == 3

        # Verify trace recorded rerank stage
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "rerank" in stage_names

    def test_trace_contains_method_field(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that rerank stage contains method field"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        # Find rerank stage
        rerank_stage = next(s for s in trace.stages if s.stage_name == "rerank")
        assert "method" in rerank_stage.metadata
        assert rerank_stage.metadata["method"] == mock_reranker_backend.__class__.__name__

    def test_trace_records_candidate_count(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that trace records candidate count"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        rerank_stage = next(s for s in trace.stages if s.stage_name == "rerank")
        assert rerank_stage.metadata["candidate_count"] == 3

    def test_trace_records_success(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that trace records success status"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        rerank_stage = next(s for s in trace.stages if s.stage_name == "rerank")
        assert rerank_stage.metadata["success"] is True
        assert rerank_stage.metadata["fallback"] is False

    def test_trace_records_fallback(self, mock_settings, sample_candidates):
        """Test that trace records fallback when reranking fails"""
        # Backend that raises exception
        failing_backend = Mock(spec=BaseReranker)
        failing_backend.rerank = Mock(side_effect=Exception("Reranking failed"))

        reranker = Reranker(mock_settings, reranker_backend=failing_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        # Verify fallback occurred
        assert result.fallback is True
        assert result.error is not None

        # Verify trace recorded fallback
        rerank_stage = next(s for s in trace.stages if s.stage_name == "rerank")
        assert rerank_stage.metadata["success"] is False
        assert rerank_stage.metadata["fallback"] is True
        assert "error" in rerank_stage.metadata

    def test_trace_has_elapsed_time(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that rerank stage has elapsed time"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        rerank_stage = next(s for s in trace.stages if s.stage_name == "rerank")
        assert rerank_stage.duration_ms is not None
        assert rerank_stage.duration_ms >= 0

    def test_trace_serialization(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that trace with rerank stage can be serialized"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_candidates, trace=trace)
        trace.finish()

        # Verify serialization
        trace_dict = trace.to_dict()
        json_str = json.dumps(trace_dict)
        assert len(json_str) > 0

        # Verify trace type
        assert trace_dict["trace_type"] == "query"

    def test_rerank_without_trace(self, mock_settings, mock_reranker_backend, sample_candidates):
        """Test that reranking works without trace"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)

        result = reranker.rerank("test query", sample_candidates, trace=None)

        # Should work normally
        assert not result.fallback
        assert len(result.results) == 3

    def test_empty_candidates_with_trace(self, mock_settings, mock_reranker_backend):
        """Test reranking empty candidates with trace"""
        reranker = Reranker(mock_settings, reranker_backend=mock_reranker_backend)
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", [], trace=trace)

        # Should return empty results
        assert len(result.results) == 0
        assert not result.fallback

        # No rerank stage should be recorded for empty candidates
        # (rerank returns early before creating stage)
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "rerank" not in stage_names
