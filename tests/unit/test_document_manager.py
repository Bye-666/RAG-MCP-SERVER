"""
Unit tests for DocumentManager.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.ingestion.document_manager import (
    DocumentManager,
    DocumentInfo,
    DocumentDetail,
    DeleteResult,
    CollectionStats
)


@pytest.fixture
def mock_chroma_store():
    """Create mock ChromaStore"""
    store = Mock()
    store.get_by_metadata = Mock(return_value=[
        {
            "id": "chunk_1",
            "text": "Test chunk 1",
            "metadata": {"source_path": "/test/doc.pdf", "file_hash": "hash123"}
        },
        {
            "id": "chunk_2",
            "text": "Test chunk 2",
            "metadata": {"source_path": "/test/doc.pdf", "file_hash": "hash123"}
        }
    ])
    store.delete_by_metadata = Mock(return_value=2)
    store.count = Mock(return_value=10)
    return store


@pytest.fixture
def mock_bm25_indexer():
    """Create mock BM25Indexer"""
    indexer = Mock()
    indexer.remove_document = Mock(return_value=5)
    indexer.save = Mock()
    indexer.get_stats = Mock(return_value={
        "num_documents": 5,
        "num_terms": 100,
        "total_postings": 200
    })
    return indexer


@pytest.fixture
def mock_image_storage():
    """Create mock ImageStorage"""
    storage = Mock()
    storage.list_images = Mock(return_value=[
        {"image_id": "img1", "file_path": "/images/img1.png"},
        {"image_id": "img2", "file_path": "/images/img2.png"}
    ])
    storage.delete_images = Mock(return_value=2)
    return storage


@pytest.fixture
def mock_file_integrity():
    """Create mock FileIntegrityChecker"""
    checker = Mock()
    checker.list_processed = Mock(return_value=[
        {
            "file_hash": "hash123",
            "file_path": "/test/doc.pdf",
            "status": "success",
            "metadata": {"chunk_count": 2, "image_count": 2},
            "created_at": "2026-05-09T10:00:00Z",
            "updated_at": "2026-05-09T10:00:00Z"
        }
    ])
    checker.remove_record = Mock(return_value=True)
    return checker


@pytest.fixture
def document_manager(mock_chroma_store, mock_bm25_indexer, mock_image_storage, mock_file_integrity):
    """Create DocumentManager with mocked dependencies"""
    return DocumentManager(
        chroma_store=mock_chroma_store,
        bm25_indexer=mock_bm25_indexer,
        image_storage=mock_image_storage,
        file_integrity=mock_file_integrity
    )


class TestDocumentManager:
    """Test DocumentManager functionality"""

    def test_list_documents(self, document_manager, mock_file_integrity):
        """Test listing documents"""
        documents = document_manager.list_documents()

        assert len(documents) == 1
        assert isinstance(documents[0], DocumentInfo)
        assert documents[0].source_path == "/test/doc.pdf"
        assert documents[0].file_hash == "hash123"
        assert documents[0].chunk_count == 2
        assert documents[0].image_count == 2
        assert documents[0].status == "success"

        mock_file_integrity.list_processed.assert_called_once_with(status="success")

    def test_list_documents_empty(self, document_manager, mock_file_integrity):
        """Test listing documents when none exist"""
        mock_file_integrity.list_processed.return_value = []

        documents = document_manager.list_documents()

        assert len(documents) == 0

    def test_get_document_detail(self, document_manager, mock_chroma_store, mock_image_storage):
        """Test getting document details"""
        detail = document_manager.get_document_detail("/test/doc.pdf")

        assert detail is not None
        assert isinstance(detail, DocumentDetail)
        assert detail.source_path == "/test/doc.pdf"
        assert detail.file_hash == "hash123"
        assert len(detail.chunks) == 2
        assert len(detail.images) == 2

        mock_chroma_store.get_by_metadata.assert_called_once_with({"source_path": "/test/doc.pdf"})
        mock_image_storage.list_images.assert_called_once_with("/test/doc.pdf")

    def test_get_document_detail_not_found(self, document_manager, mock_chroma_store):
        """Test getting details for non-existent document"""
        mock_chroma_store.get_by_metadata.return_value = []

        detail = document_manager.get_document_detail("/nonexistent.pdf")

        assert detail is None

    def test_delete_document_success(self, document_manager, mock_chroma_store,
                                    mock_bm25_indexer, mock_image_storage, mock_file_integrity):
        """Test successful document deletion"""
        result = document_manager.delete_document("/test/doc.pdf")

        assert result.success is True
        assert result.chunks_deleted == 2
        assert result.bm25_postings_deleted == 5
        assert result.images_deleted == 2
        assert result.integrity_record_deleted is True
        assert result.error is None

        # Verify all storage systems were called
        mock_chroma_store.get_by_metadata.assert_called_once()
        mock_chroma_store.delete_by_metadata.assert_called_once_with({"source_path": "/test/doc.pdf"})
        mock_bm25_indexer.remove_document.assert_called_once_with(["chunk_1", "chunk_2"])
        mock_bm25_indexer.save.assert_called_once()
        mock_image_storage.delete_images.assert_called_once_with("/test/doc.pdf")
        mock_file_integrity.remove_record.assert_called_once_with("hash123")

    def test_delete_document_not_found(self, document_manager, mock_chroma_store):
        """Test deleting non-existent document"""
        mock_chroma_store.get_by_metadata.return_value = []

        result = document_manager.delete_document("/nonexistent.pdf")

        assert result.success is False
        assert result.error == "No chunks found for source_path: /nonexistent.pdf"

    def test_delete_document_with_error(self, document_manager, mock_chroma_store):
        """Test document deletion with error"""
        mock_chroma_store.delete_by_metadata.side_effect = Exception("Database error")

        result = document_manager.delete_document("/test/doc.pdf")

        assert result.success is False
        assert "Database error" in result.error

    def test_get_collection_stats(self, document_manager, mock_chroma_store,
                                  mock_bm25_indexer, mock_file_integrity):
        """Test getting collection statistics"""
        stats = document_manager.get_collection_stats()

        assert isinstance(stats, CollectionStats)
        assert stats.total_documents == 1
        assert stats.total_chunks == 2
        assert stats.total_images == 2
        assert stats.vector_store_count == 10
        assert stats.bm25_stats["num_documents"] == 5

        mock_file_integrity.list_processed.assert_called_once()
        mock_chroma_store.count.assert_called_once()
        mock_bm25_indexer.get_stats.assert_called_once()

    def test_get_collection_stats_empty(self, document_manager, mock_file_integrity,
                                       mock_chroma_store, mock_bm25_indexer):
        """Test getting stats for empty collection"""
        mock_file_integrity.list_processed.return_value = []
        mock_chroma_store.count.return_value = 0

        stats = document_manager.get_collection_stats()

        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_images == 0
        assert stats.vector_store_count == 0

    def test_list_documents_with_collection_filter(self, document_manager):
        """Test listing documents with collection filter"""
        # Collection filter is currently not implemented, but should not error
        documents = document_manager.list_documents(collection="test_collection")

        assert isinstance(documents, list)

    def test_delete_document_with_collection(self, document_manager):
        """Test deleting document with collection parameter"""
        # Collection parameter is currently not used, but should not error
        result = document_manager.delete_document("/test/doc.pdf", collection="test_collection")

        assert isinstance(result, DeleteResult)
