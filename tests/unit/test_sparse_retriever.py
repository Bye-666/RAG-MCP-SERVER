"""
Unit tests for SparseRetriever.

Tests BM25 keyword-based retrieval with mocked indexer and vector store.
"""

import pytest
from unittest.mock import Mock

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace import TraceContext


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = Mock(spec=Settings)
    settings.vector_store = {"provider": "chroma"}
    return settings


@pytest.fixture
def mock_bm25_indexer():
    """Create mock BM25 indexer"""
    indexer = Mock()
    indexer.query = Mock(return_value=[
        {"chunk_id": "chunk_001", "score": 2.5},
        {"chunk_id": "chunk_002", "score": 1.8},
        {"chunk_id": "chunk_003", "score": 1.2}
    ])
    return indexer


@pytest.fixture
def mock_vector_store():
    """Create mock vector store"""
    store = Mock()
    store.get_by_ids = Mock(return_value=[
        {
            "id": "chunk_001",
            "text": "Azure OpenAI configuration guide",
            "metadata": {"source": "docs.pdf", "page": 1}
        },
        {
            "id": "chunk_002",
            "text": "Setting up API keys",
            "metadata": {"source": "docs.pdf", "page": 2}
        },
        {
            "id": "chunk_003",
            "text": "Authentication methods",
            "metadata": {"source": "docs.pdf", "page": 3}
        }
    ])
    return store


class TestSparseRetriever:
    """Tests for SparseRetriever"""

    def test_initialization_with_injected_dependencies(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test initialization with dependency injection"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        assert retriever.settings == mock_settings
        assert retriever.bm25_indexer == mock_bm25_indexer
        assert retriever.vector_store == mock_vector_store

    def test_retrieve_basic(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test basic retrieval flow"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        keywords = ["azure", "openai", "configuration"]
        results = retriever.retrieve(keywords, top_k=3)

        # Verify BM25 indexer was called
        mock_bm25_indexer.query.assert_called_once_with(keywords, top_k=3)

        # Verify vector store was called with chunk_ids
        mock_vector_store.get_by_ids.assert_called_once()
        call_args = mock_vector_store.get_by_ids.call_args[0][0]
        assert set(call_args) == {"chunk_001", "chunk_002", "chunk_003"}

        # Verify results
        assert len(results) == 3
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 2.5
        assert results[0].text == "Azure OpenAI configuration guide"
        assert results[0].metadata == {"source": "docs.pdf", "page": 1}

    def test_retrieve_maintains_score_order(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test that results maintain BM25 score ordering"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve(["test"], top_k=3)

        # Scores should be in descending order
        assert results[0].score == 2.5
        assert results[1].score == 1.8
        assert results[2].score == 1.2

    def test_retrieve_with_trace(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test retrieval with trace context"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        trace = TraceContext(trace_id="test-trace")
        results = retriever.retrieve(["test"], trace=trace)

        # Verify trace stages were recorded
        assert len(trace.stages) >= 2
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "sparse_retriever_bm25" in stage_names
        assert "sparse_retriever_fetch" in stage_names

    def test_retrieve_empty_keywords_raises_error(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test that empty keywords raises ValueError"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        with pytest.raises(ValueError, match="Keywords cannot be empty"):
            retriever.retrieve([])

    def test_retrieve_invalid_top_k_raises_error(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test that invalid top_k raises ValueError"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve(["test"], top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve(["test"], top_k=-1)

    def test_retrieve_no_bm25_results(self, mock_settings, mock_vector_store):
        """Test retrieval when BM25 returns no results"""
        empty_indexer = Mock()
        empty_indexer.query = Mock(return_value=[])

        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=empty_indexer,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve(["nonexistent"])

        assert results == []
        # Vector store should not be called if BM25 returns nothing
        mock_vector_store.get_by_ids.assert_not_called()

    def test_retrieve_missing_chunks_in_vector_store(self, mock_settings, mock_bm25_indexer):
        """Test handling when some chunks are missing from vector store"""
        # BM25 returns 3 chunks
        mock_bm25_indexer.query = Mock(return_value=[
            {"chunk_id": "chunk_001", "score": 2.5},
            {"chunk_id": "chunk_002", "score": 1.8},
            {"chunk_id": "chunk_003", "score": 1.2}
        ])

        # Vector store only has 2 chunks
        partial_store = Mock()
        partial_store.get_by_ids = Mock(return_value=[
            {"id": "chunk_001", "text": "text 1", "metadata": {}},
            {"id": "chunk_002", "text": "text 2", "metadata": {}}
            # chunk_003 is missing
        ])

        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=partial_store
        )

        results = retriever.retrieve(["test"])

        # Should only return chunks that exist in vector store
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_001"
        assert results[1].chunk_id == "chunk_002"

    def test_retrieve_with_custom_top_k(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test retrieval with custom top_k value"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve(["test"], top_k=10)

        # Verify top_k was passed to BM25 indexer
        call_args = mock_bm25_indexer.query.call_args
        assert call_args[1]["top_k"] == 10

    def test_retrieve_single_keyword(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test retrieval with single keyword"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve(["azure"])

        assert len(results) == 3
        mock_bm25_indexer.query.assert_called_once_with(["azure"], top_k=5)

    def test_retrieve_multiple_keywords(self, mock_settings, mock_bm25_indexer, mock_vector_store):
        """Test retrieval with multiple keywords"""
        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store
        )

        keywords = ["azure", "openai", "api", "configuration"]
        results = retriever.retrieve(keywords)

        mock_bm25_indexer.query.assert_called_once_with(keywords, top_k=5)

    def test_retrieve_missing_text_field(self, mock_settings, mock_bm25_indexer):
        """Test handling when text field is missing from vector store result"""
        mock_bm25_indexer.query = Mock(return_value=[
            {"chunk_id": "chunk_001", "score": 2.5}
        ])

        store = Mock()
        store.get_by_ids = Mock(return_value=[
            {"id": "chunk_001", "metadata": {}}
            # No text field
        ])

        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=store
        )

        results = retriever.retrieve(["test"])

        assert len(results) == 1
        assert results[0].text == ""

    def test_retrieve_missing_metadata_field(self, mock_settings, mock_bm25_indexer):
        """Test handling when metadata field is missing from vector store result"""
        mock_bm25_indexer.query = Mock(return_value=[
            {"chunk_id": "chunk_001", "score": 2.5}
        ])

        store = Mock()
        store.get_by_ids = Mock(return_value=[
            {"id": "chunk_001", "text": "test text"}
            # No metadata field
        ])

        retriever = SparseRetriever(
            settings=mock_settings,
            bm25_indexer=mock_bm25_indexer,
            vector_store=store
        )

        results = retriever.retrieve(["test"])

        assert len(results) == 1
        assert results[0].metadata == {}
