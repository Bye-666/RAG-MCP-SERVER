"""
测试 DataService 数据服务层。
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.observability.dashboard.services.data_service import DataService
from src.ingestion.document_manager import DocumentInfo, DocumentDetail, CollectionStats


@pytest.fixture
def mock_chroma_store():
    """Mock ChromaStore"""
    store = Mock()
    store.get_by_metadata.return_value = [
        {
            "id": "chunk_1",
            "text": "这是第一个 chunk",
            "metadata": {"source_path": "test.pdf", "page": 1}
        },
        {
            "id": "chunk_2",
            "text": "这是第二个 chunk",
            "metadata": {"source_path": "test.pdf", "page": 2}
        }
    ]
    store.count.return_value = 2
    return store


@pytest.fixture
def mock_image_storage():
    """Mock ImageStorage"""
    storage = Mock()
    storage.list_images.return_value = [
        {
            "image_id": "img_1",
            "file_path": "data/images/default/img_1.png",
            "collection": "default",
            "doc_hash": "abc123",
            "page_num": 1
        }
    ]
    storage.get_image_path.return_value = "data/images/default/img_1.png"
    return storage


@pytest.fixture
def mock_document_manager():
    """Mock DocumentManager"""
    manager = Mock()
    manager.list_documents.return_value = [
        DocumentInfo(
            source_path="test.pdf",
            file_hash="abc123",
            chunk_count=2,
            image_count=1,
            created_at="2026-05-09T10:00:00",
            updated_at="2026-05-09T10:00:00",
            status="success"
        )
    ]
    manager.get_document_detail.return_value = DocumentDetail(
        source_path="test.pdf",
        file_hash="abc123",
        chunks=[
            {
                "id": "chunk_1",
                "text": "这是第一个 chunk",
                "metadata": {"source_path": "test.pdf", "page": 1}
            }
        ],
        images=[
            {
                "image_id": "img_1",
                "file_path": "data/images/default/img_1.png"
            }
        ],
        metadata={"source_path": "test.pdf", "file_hash": "abc123"}
    )
    manager.get_collection_stats.return_value = CollectionStats(
        total_documents=1,
        total_chunks=2,
        total_images=1,
        vector_store_count=2,
        bm25_stats={"total_docs": 1, "total_terms": 10}
    )
    return manager


@pytest.fixture
def data_service(mock_chroma_store, mock_image_storage, mock_document_manager):
    """创建 DataService 实例"""
    return DataService(
        chroma_store=mock_chroma_store,
        image_storage=mock_image_storage,
        document_manager=mock_document_manager
    )


def test_list_documents(data_service, mock_document_manager):
    """测试列出文档"""
    docs = data_service.list_documents()

    assert len(docs) == 1
    assert docs[0].source_path == "test.pdf"
    assert docs[0].chunk_count == 2
    assert docs[0].image_count == 1
    mock_document_manager.list_documents.assert_called_once_with(None)


def test_list_documents_with_collection(data_service, mock_document_manager):
    """测试按集合列出文档"""
    docs = data_service.list_documents(collection="my_collection")

    mock_document_manager.list_documents.assert_called_once_with("my_collection")


def test_get_document_detail(data_service, mock_document_manager):
    """测试获取文档详情"""
    detail = data_service.get_document_detail("test.pdf")

    assert detail is not None
    assert detail.source_path == "test.pdf"
    assert detail.file_hash == "abc123"
    assert len(detail.chunks) == 1
    assert len(detail.images) == 1
    mock_document_manager.get_document_detail.assert_called_once_with("test.pdf")


def test_get_document_detail_not_found(data_service, mock_document_manager):
    """测试获取不存在的文档详情"""
    mock_document_manager.get_document_detail.return_value = None

    detail = data_service.get_document_detail("nonexistent.pdf")

    assert detail is None


def test_get_chunks_by_source(data_service, mock_chroma_store):
    """测试根据源路径获取 chunks"""
    chunks = data_service.get_chunks_by_source("test.pdf")

    assert len(chunks) == 2
    assert chunks[0]["id"] == "chunk_1"
    assert chunks[1]["id"] == "chunk_2"
    mock_chroma_store.get_by_metadata.assert_called_once_with({"source_path": "test.pdf"})


def test_get_images_by_source(data_service, mock_image_storage):
    """测试根据源路径获取图片"""
    images = data_service.get_images_by_source("test.pdf")

    assert len(images) == 1
    assert images[0]["image_id"] == "img_1"
    mock_image_storage.list_images.assert_called_once_with("test.pdf")


def test_get_images_by_source_exception(data_service, mock_image_storage):
    """测试获取图片时发生异常"""
    mock_image_storage.list_images.side_effect = Exception("Database error")

    images = data_service.get_images_by_source("test.pdf")

    assert images == []


def test_get_image_path(data_service, mock_image_storage):
    """测试获取图片路径"""
    path = data_service.get_image_path("img_1")

    assert path == "data/images/default/img_1.png"
    mock_image_storage.get_image_path.assert_called_once_with("img_1")


def test_get_image_path_exception(data_service, mock_image_storage):
    """测试获取图片路径时发生异常"""
    mock_image_storage.get_image_path.side_effect = Exception("Not found")

    path = data_service.get_image_path("img_1")

    assert path is None


def test_get_collection_stats(data_service, mock_document_manager):
    """测试获取集合统计信息"""
    stats = data_service.get_collection_stats()

    assert stats["total_documents"] == 1
    assert stats["total_chunks"] == 2
    assert stats["total_images"] == 1
    assert stats["vector_store_count"] == 2
    assert stats["bm25_stats"]["total_docs"] == 1
    mock_document_manager.get_collection_stats.assert_called_once_with(None)


def test_search_documents_no_keyword(data_service, mock_document_manager):
    """测试无关键词搜索（返回所有文档）"""
    docs = data_service.search_documents("")

    assert len(docs) == 1
    assert docs[0].source_path == "test.pdf"


def test_search_documents_with_keyword(data_service, mock_document_manager):
    """测试关键词搜索"""
    docs = data_service.search_documents("test")

    assert len(docs) == 1
    assert docs[0].source_path == "test.pdf"


def test_search_documents_no_match(data_service, mock_document_manager):
    """测试关键词搜索无匹配"""
    docs = data_service.search_documents("nonexistent")

    assert len(docs) == 0


def test_search_documents_case_insensitive(data_service, mock_document_manager):
    """测试关键词搜索大小写不敏感"""
    docs = data_service.search_documents("TEST")

    assert len(docs) == 1
    assert docs[0].source_path == "test.pdf"
