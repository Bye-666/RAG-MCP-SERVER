"""
Unit tests for DenseRetriever.

Tests semantic retrieval with mocked embedding and vector store.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace import TraceContext


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = Mock(spec=Settings)
    settings.embedding = {"provider": "openai"}
    settings.vector_store = {"provider": "chroma"}
    return settings


@pytest.fixture
def mock_embedding_client():
    """Create mock embedding client"""
    client = Mock()
    client.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
    return client


@pytest.fixture
def mock_vector_store():
    """Create mock vector store"""
    store = Mock()
    store.query = Mock(return_value=[
        {
            "id": "chunk_001",
            "score": 0.95,
            "text": "Azure OpenAI configuration guide",
            "metadata": {"source": "docs.pdf", "page": 1}
        },
        {
            "id": "chunk_002",
            "score": 0.87,
            "text": "Setting up API keys",
            "metadata": {"source": "docs.pdf", "page": 2}
        }
    ])
    return store


class TestDenseRetriever:
    """Tests for DenseRetriever"""

    def test_initialization_with_injected_dependencies(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test initialization with dependency injection"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        assert retriever.settings == mock_settings
        assert retriever.embedding_client == mock_embedding_client
        assert retriever.vector_store == mock_vector_store

    def test_retrieve_basic(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test basic retrieval flow"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve("How to configure Azure?", top_k=2)

        # Verify embedding was called
        mock_embedding_client.embed.assert_called_once()
        call_args = mock_embedding_client.embed.call_args[0]
        assert call_args[0] == ["How to configure Azure?"]

        # Verify vector store was queried
        mock_vector_store.query.assert_called_once()
        query_call = mock_vector_store.query.call_args
        assert query_call[1]["vector"] == [0.1, 0.2, 0.3]
        assert query_call[1]["top_k"] == 2

        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 0.95
        assert results[0].text == "Azure OpenAI configuration guide"
        assert results[0].metadata == {"source": "docs.pdf", "page": 1}

    def test_retrieve_with_filters(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test retrieval with metadata filters"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        filters = {"collection": "docs", "doc_type": "pdf"}
        results = retriever.retrieve("test query", top_k=5, filters=filters)

        # Verify filters were passed to vector store
        query_call = mock_vector_store.query.call_args
        assert query_call[1]["filters"] == filters

    def test_retrieve_with_trace(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test retrieval with trace context"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        trace = TraceContext(trace_id="test-trace")
        results = retriever.retrieve("test query", trace=trace)

        # Verify trace was passed through
        embed_call = mock_embedding_client.embed.call_args
        assert embed_call[1]["trace"] == trace

        query_call = mock_vector_store.query.call_args
        assert query_call[1]["trace"] == trace

        # Verify trace logs were created
        assert len(trace.stages) > 0

    def test_retrieve_empty_query_raises_error(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test that empty query raises ValueError"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")

    def test_retrieve_whitespace_query_raises_error(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test that whitespace-only query raises ValueError"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ")

    def test_retrieve_invalid_top_k_raises_error(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test that invalid top_k raises ValueError"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test query", top_k=-1)

    def test_retrieve_empty_results(self, mock_settings, mock_embedding_client):
        """Test retrieval with no results"""
        empty_store = Mock()
        empty_store.query = Mock(return_value=[])

        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=empty_store
        )

        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_result_ordering(self, mock_settings, mock_embedding_client):
        """Test that results maintain score ordering"""
        store = Mock()
        store.query = Mock(return_value=[
            {"id": "chunk_1", "score": 0.9, "text": "high score", "metadata": {}},
            {"id": "chunk_2", "score": 0.7, "text": "medium score", "metadata": {}},
            {"id": "chunk_3", "score": 0.5, "text": "low score", "metadata": {}}
        ])

        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=store
        )

        results = retriever.retrieve("test query")

        assert len(results) == 3
        assert results[0].score == 0.9
        assert results[1].score == 0.7
        assert results[2].score == 0.5

    def test_retrieve_missing_metadata(self, mock_settings, mock_embedding_client):
        """Test handling of missing metadata field"""
        store = Mock()
        store.query = Mock(return_value=[
            {"id": "chunk_1", "score": 0.9, "text": "test"}
            # No metadata field
        ])

        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=store
        )

        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].metadata == {}

    def test_retrieve_with_custom_top_k(self, mock_settings, mock_embedding_client, mock_vector_store):
        """Test retrieval with custom top_k value"""
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store
        )

        results = retriever.retrieve("test query", top_k=10)

        query_call = mock_vector_store.query.call_args
        assert query_call[1]["top_k"] == 10

    def test_embedding_vector_passed_correctly(self, mock_settings, mock_vector_store):
        """Test that embedding vector is correctly passed to vector store"""
        embedding_client = Mock()
        embedding_client.embed = Mock(return_value=[[0.5, 0.6, 0.7, 0.8]])

        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=embedding_client,
            vector_store=mock_vector_store
        )

        retriever.retrieve("test query")

        query_call = mock_vector_store.query.call_args
        assert query_call[1]["vector"] == [0.5, 0.6, 0.7, 0.8]
