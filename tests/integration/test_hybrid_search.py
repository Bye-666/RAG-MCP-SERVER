"""
Integration tests for HybridSearch.

Tests the complete hybrid search pipeline with mocked components.
"""

import pytest
from unittest.mock import Mock

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery
from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace import TraceContext


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def mock_query_processor():
    """Create mock query processor"""
    processor = Mock(spec=QueryProcessor)
    processor.process = Mock(return_value=ProcessedQuery(
        raw_query="test query",
        keywords=["test", "query"],
        filters={}
    ))
    return processor


@pytest.fixture
def mock_dense_retriever():
    """Create mock dense retriever"""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=[
        RetrievalResult(chunk_id="dense_1", score=0.9, text="Dense result 1", metadata={"source": "doc1"}),
        RetrievalResult(chunk_id="dense_2", score=0.8, text="Dense result 2", metadata={"source": "doc2"}),
    ])
    return retriever


@pytest.fixture
def mock_sparse_retriever():
    """Create mock sparse retriever"""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=[
        RetrievalResult(chunk_id="sparse_1", score=10.0, text="Sparse result 1", metadata={"source": "doc3"}),
        RetrievalResult(chunk_id="sparse_2", score=8.0, text="Sparse result 2", metadata={"source": "doc4"}),
    ])
    return retriever


@pytest.fixture
def mock_fusion():
    """Create mock RRF fusion"""
    fusion = Mock()
    fusion.fuse = Mock(return_value=[
        RetrievalResult(chunk_id="dense_1", score=0.05, text="Dense result 1", metadata={"source": "doc1"}),
        RetrievalResult(chunk_id="sparse_1", score=0.04, text="Sparse result 1", metadata={"source": "doc3"}),
        RetrievalResult(chunk_id="dense_2", score=0.03, text="Dense result 2", metadata={"source": "doc2"}),
        RetrievalResult(chunk_id="sparse_2", score=0.02, text="Sparse result 2", metadata={"source": "doc4"}),
    ])
    return fusion


class TestHybridSearch:
    """Tests for HybridSearch"""

    def test_initialization(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test HybridSearch initialization with injected dependencies"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        assert search.settings == mock_settings
        assert search.query_processor == mock_query_processor
        assert search.dense_retriever == mock_dense_retriever
        assert search.sparse_retriever == mock_sparse_retriever
        assert search.fusion == mock_fusion

    def test_basic_search(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test basic hybrid search flow"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        results = search.search("test query", top_k=3)

        # Verify query processor was called
        mock_query_processor.process.assert_called_once_with("test query", filters=None)

        # Verify retrievers were called
        mock_dense_retriever.retrieve.assert_called_once()
        mock_sparse_retriever.retrieve.assert_called_once()

        # Verify fusion was called
        mock_fusion.fuse.assert_called_once()

        # Verify results
        assert len(results) == 3
        assert results[0].chunk_id == "dense_1"
        assert results[1].chunk_id == "sparse_1"
        assert results[2].chunk_id == "dense_2"

    def test_search_with_filters(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test search with metadata filters"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        filters = {"collection": "docs"}
        results = search.search("test query", top_k=5, filters=filters)

        # Verify filters were passed to query processor
        mock_query_processor.process.assert_called_once_with("test query", filters=filters)

    def test_search_with_trace(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test search with trace context"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        trace = TraceContext(trace_id="test-trace")
        results = search.search("test query", trace=trace)

        # Verify trace was passed to retrievers
        dense_call = mock_dense_retriever.retrieve.call_args
        assert dense_call[1]["trace"] == trace

        sparse_call = mock_sparse_retriever.retrieve.call_args
        assert sparse_call[1]["trace"] == trace

        # Verify trace stages were recorded
        assert len(trace.stages) > 0

    def test_empty_query_raises_error(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that empty query raises ValueError"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            search.search("")

    def test_invalid_top_k_raises_error(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that invalid top_k raises ValueError"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        with pytest.raises(ValueError, match="top_k must be positive"):
            search.search("test query", top_k=0)

    def test_dense_retriever_failure_fallback(self, mock_settings, mock_query_processor, mock_sparse_retriever, mock_fusion):
        """Test fallback when dense retriever fails"""
        # Dense retriever that raises exception
        failing_dense = Mock()
        failing_dense.retrieve = Mock(side_effect=Exception("Dense retrieval failed"))

        # Fusion should only receive sparse results
        mock_fusion.fuse = Mock(return_value=[
            RetrievalResult(chunk_id="sparse_1", score=0.04, text="Sparse result 1", metadata={}),
        ])

        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=failing_dense,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        results = search.search("test query")

        # Should still return results from sparse retriever
        assert len(results) > 0
        # Fusion should be called with empty dense results
        fusion_call = mock_fusion.fuse.call_args[0]
        assert fusion_call[0] == []  # dense_results is empty

    def test_sparse_retriever_failure_fallback(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_fusion):
        """Test fallback when sparse retriever fails"""
        # Sparse retriever that raises exception
        failing_sparse = Mock()
        failing_sparse.retrieve = Mock(side_effect=Exception("Sparse retrieval failed"))

        # Fusion should only receive dense results
        mock_fusion.fuse = Mock(return_value=[
            RetrievalResult(chunk_id="dense_1", score=0.05, text="Dense result 1", metadata={}),
        ])

        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=failing_sparse,
            fusion=mock_fusion
        )

        results = search.search("test query")

        # Should still return results from dense retriever
        assert len(results) > 0
        # Fusion should be called with empty sparse results
        fusion_call = mock_fusion.fuse.call_args[0]
        assert fusion_call[1] == []  # sparse_results is empty

    def test_both_retrievers_fail(self, mock_settings, mock_query_processor, mock_fusion):
        """Test when both retrievers fail"""
        failing_dense = Mock()
        failing_dense.retrieve = Mock(side_effect=Exception("Dense failed"))

        failing_sparse = Mock()
        failing_sparse.retrieve = Mock(side_effect=Exception("Sparse failed"))

        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=failing_dense,
            sparse_retriever=failing_sparse,
            fusion=mock_fusion
        )

        results = search.search("test query")

        # Should return empty list
        assert results == []
        # Fusion should not be called
        mock_fusion.fuse.assert_not_called()

    def test_metadata_filter_application(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever):
        """Test metadata filtering on fused results"""
        # Fusion returns mixed results
        fusion = Mock()
        fusion.fuse = Mock(return_value=[
            RetrievalResult(chunk_id="chunk1", score=0.05, text="text1", metadata={"collection": "docs"}),
            RetrievalResult(chunk_id="chunk2", score=0.04, text="text2", metadata={"collection": "other"}),
            RetrievalResult(chunk_id="chunk3", score=0.03, text="text3", metadata={"collection": "docs"}),
        ])

        # Query processor returns filters
        processor = Mock()
        processor.process = Mock(return_value=ProcessedQuery(
            raw_query="test",
            keywords=["test"],
            filters={"collection": "docs"}
        ))

        search = HybridSearch(
            settings=mock_settings,
            query_processor=processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=fusion
        )

        results = search.search("test query", filters={"collection": "docs"})

        # Should only return chunks with collection="docs"
        assert len(results) == 2
        assert all(r.metadata.get("collection") == "docs" for r in results)

    def test_top_k_selection(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that top_k limits results correctly"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        results = search.search("test query", top_k=2)

        # Should return exactly 2 results
        assert len(results) == 2

    def test_candidate_k_multiplier(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that retrievers are called with 2x top_k for better fusion"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        search.search("test query", top_k=5)

        # Retrievers should be called with top_k=10 (2x)
        dense_call = mock_dense_retriever.retrieve.call_args
        assert dense_call[1]["top_k"] == 10

        sparse_call = mock_sparse_retriever.retrieve.call_args
        assert sparse_call[1]["top_k"] == 10

    def test_no_keywords_sparse_skip(self, mock_settings, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that sparse retrieval is skipped when no keywords"""
        # Query processor returns no keywords
        processor = Mock()
        processor.process = Mock(return_value=ProcessedQuery(
            raw_query="???",
            keywords=[],
            filters={}
        ))

        search = HybridSearch(
            settings=mock_settings,
            query_processor=processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        search.search("???")

        # Sparse retriever should not be called
        mock_sparse_retriever.retrieve.assert_not_called()

    def test_trace_contains_all_stages(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that trace contains all required stages"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        trace = TraceContext(trace_type="query")
        results = search.search("test query", trace=trace)
        trace.finish()

        # Verify trace type
        trace_dict = trace.to_dict()
        assert trace_dict["trace_type"] == "query"

        # Verify all required stages exist
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "query_processing" in stage_names
        assert "dense_retrieval" in stage_names
        assert "sparse_retrieval" in stage_names
        assert "fusion" in stage_names

    def test_trace_stages_have_method_field(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that each stage records method field"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        trace = TraceContext(trace_type="query")
        results = search.search("test query", trace=trace)
        trace.finish()

        # Verify each stage has method field
        for stage in trace.stages:
            assert "method" in stage.metadata, f"Stage {stage.stage_name} missing method field"

    def test_trace_stages_have_elapsed_ms(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that each stage records elapsed time"""
        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        trace = TraceContext(trace_type="query")
        results = search.search("test query", trace=trace)
        trace.finish()

        # Verify each stage has duration
        for stage in trace.stages:
            assert stage.duration_ms is not None, f"Stage {stage.stage_name} missing duration"
            assert stage.duration_ms >= 0

    def test_trace_serialization(self, mock_settings, mock_query_processor, mock_dense_retriever, mock_sparse_retriever, mock_fusion):
        """Test that trace can be serialized to dict"""
        import json

        search = HybridSearch(
            settings=mock_settings,
            query_processor=mock_query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=mock_fusion
        )

        trace = TraceContext(trace_type="query")
        results = search.search("test query", trace=trace)
        trace.finish()

        # Verify trace can be serialized
        trace_dict = trace.to_dict()
        json_str = json.dumps(trace_dict)
        assert len(json_str) > 0

        # Verify required fields
        assert "trace_id" in trace_dict
        assert "trace_type" in trace_dict
        assert "started_at" in trace_dict
        assert "finished_at" in trace_dict
        assert "total_elapsed_ms" in trace_dict
        assert "stages" in trace_dict
