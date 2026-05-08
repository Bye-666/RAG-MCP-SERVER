"""
Unit tests for DocumentChunker (C4).

Tests the adapter layer between libs.splitter and core.types.
"""
import re
from unittest.mock import Mock, patch

import pytest

from src.core.types import Chunk, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing."""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def split_text(self, text: str) -> list[str]:
        """Split text into fixed-size chunks."""
        if not text:
            return []
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


@pytest.fixture
def fake_splitter():
    """Provide a fake splitter."""
    return FakeSplitter(chunk_size=50)


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock()
    settings.splitter = Mock()
    settings.splitter.provider = "recursive"
    settings.splitter.chunk_size = 100
    settings.splitter.chunk_overlap = 20
    return settings


@pytest.fixture
def sample_document():
    """Sample document without images."""
    return Document(
        id="doc_abc123",
        text="This is a test document. It has multiple sentences. We will split it into chunks.",
        metadata={
            "source_path": "/path/to/test.pdf",
            "doc_type": "pdf",
            "title": "Test Document"
        }
    )


@pytest.fixture
def document_with_images():
    """Sample document with image references."""
    return Document(
        id="doc_xyz789",
        text="First chunk text [IMAGE: img_001]. Second chunk text [IMAGE: img_002]. Third chunk without image.",
        metadata={
            "source_path": "/path/to/doc.pdf",
            "doc_type": "pdf",
            "images": [
                {
                    "image_id": "img_001",
                    "file_path": "/data/images/doc_xyz789/img_001.png",
                    "page_number": 1
                },
                {
                    "image_id": "img_002",
                    "file_path": "/data/images/doc_xyz789/img_002.png",
                    "page_number": 2
                }
            ]
        }
    )


class TestDocumentChunkerInit:
    """Test DocumentChunker initialization."""

    def test_init_with_settings(self, mock_settings):
        """Should initialize with settings and create splitter."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter()
            chunker = DocumentChunker(mock_settings)

            assert chunker.settings == mock_settings
            assert chunker.splitter is not None
            mock_factory.create.assert_called_once_with(mock_settings.splitter)


class TestSplitDocument:
    """Test split_document method."""

    def test_split_simple_document(self, mock_settings, sample_document):
        """Should split document into chunks with correct metadata."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=30)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)

            for idx, chunk in enumerate(chunks):
                assert chunk.id.startswith(f"{sample_document.id}_")
                assert chunk.source_ref == sample_document.id
                assert chunk.metadata["source_path"] == "/path/to/test.pdf"
                assert chunk.metadata["doc_type"] == "pdf"
                assert chunk.metadata["title"] == "Test Document"
                assert chunk.metadata["chunk_index"] == idx

    def test_chunk_id_uniqueness(self, mock_settings, sample_document):
        """Should generate unique chunk IDs."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=20)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)
            chunk_ids = [c.id for c in chunks]

            assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_id_deterministic(self, mock_settings, sample_document):
        """Should generate same chunk IDs for same document."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=20)
            chunker = DocumentChunker(mock_settings)

            chunks1 = chunker.split_document(sample_document)
            chunks2 = chunker.split_document(sample_document)

            ids1 = [c.id for c in chunks1]
            ids2 = [c.id for c in chunks2]

            assert ids1 == ids2

    def test_chunk_id_format(self, mock_settings, sample_document):
        """Should generate chunk IDs in correct format."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=20)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            pattern = re.compile(r'^doc_abc123_\d{4}_[a-f0-9]{8}$')
            for chunk in chunks:
                assert pattern.match(chunk.id), f"Invalid chunk ID format: {chunk.id}"

    def test_metadata_inheritance(self, mock_settings, sample_document):
        """Should inherit all document metadata."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=30)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            for chunk in chunks:
                assert chunk.metadata["source_path"] == sample_document.metadata["source_path"]
                assert chunk.metadata["doc_type"] == sample_document.metadata["doc_type"]
                assert chunk.metadata["title"] == sample_document.metadata["title"]

    def test_chunk_index_sequential(self, mock_settings, sample_document):
        """Should assign sequential chunk_index starting from 0."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=20)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            for idx, chunk in enumerate(chunks):
                assert chunk.metadata["chunk_index"] == idx

    def test_source_ref_correct(self, mock_settings, sample_document):
        """Should set source_ref to parent document ID."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=30)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            for chunk in chunks:
                assert chunk.source_ref == sample_document.id

    def test_empty_document(self, mock_settings):
        """Should handle empty document."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter()
            chunker = DocumentChunker(mock_settings)

            doc = Document(id="doc_empty", text="", metadata={"source_path": "/path/to/empty.pdf"})
            chunks = chunker.split_document(doc)

            assert chunks == []


class TestImageReferenceDistribution:
    """Test image reference distribution to chunks."""

    def test_distribute_images_to_chunks(self, mock_settings, document_with_images):
        """Should distribute images only to chunks that reference them."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=40)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(document_with_images)

            chunk_with_img1 = next((c for c in chunks if "[IMAGE: img_001]" in c.text), None)
            assert chunk_with_img1 is not None
            assert "images" in chunk_with_img1.metadata
            assert len(chunk_with_img1.metadata["images"]) == 1
            assert chunk_with_img1.metadata["images"][0]["image_id"] == "img_001"
            assert "image_refs" in chunk_with_img1.metadata
            assert chunk_with_img1.metadata["image_refs"] == ["img_001"]

            chunk_with_img2 = next((c for c in chunks if "[IMAGE: img_002]" in c.text), None)
            assert chunk_with_img2 is not None
            assert "images" in chunk_with_img2.metadata
            assert len(chunk_with_img2.metadata["images"]) == 1
            assert chunk_with_img2.metadata["images"][0]["image_id"] == "img_002"
            assert chunk_with_img2.metadata["image_refs"] == ["img_002"]

    def test_no_images_in_chunk_without_placeholder(self, mock_settings, document_with_images):
        """Should not add images field to chunks without image placeholders."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=40)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(document_with_images)

            chunk_without_img = next((c for c in chunks if "without image" in c.text), None)
            if chunk_without_img:
                assert "images" not in chunk_without_img.metadata
                assert "image_refs" not in chunk_without_img.metadata

    def test_chunk_with_multiple_images(self, mock_settings):
        """Should handle chunk with multiple image references."""
        doc = Document(
            id="doc_multi",
            text="Text [IMAGE: img_001] more text [IMAGE: img_002] end.",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    {
                        "image_id": "img_001",
                        "file_path": "/path/img_001.png",
                        "page_number": 1
                    },
                    {
                        "image_id": "img_002",
                        "file_path": "/path/img_002.png",
                        "page_number": 2
                    }
                ]
            }
        )

        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=100)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(doc)

            assert len(chunks) == 1
            chunk = chunks[0]
            assert "images" in chunk.metadata
            assert len(chunk.metadata["images"]) == 2
            assert chunk.metadata["image_refs"] == ["img_001", "img_002"]

    def test_document_without_images_metadata(self, mock_settings, sample_document):
        """Should handle document without images metadata field."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=30)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            for chunk in chunks:
                assert "images" not in chunk.metadata
                assert "image_refs" not in chunk.metadata


class TestChunkSerialization:
    """Test that output chunks are serializable."""

    def test_chunks_serializable(self, mock_settings, sample_document):
        """Should produce chunks that can be serialized to dict."""
        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.return_value = FakeSplitter(chunk_size=30)
            chunker = DocumentChunker(mock_settings)

            chunks = chunker.split_document(sample_document)

            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                assert isinstance(chunk_dict, dict)
                assert "id" in chunk_dict
                assert "text" in chunk_dict
                assert "metadata" in chunk_dict
                assert "source_ref" in chunk_dict


class TestConfigurationDriven:
    """Test that chunking behavior is driven by configuration."""

    def test_different_chunk_sizes(self, sample_document):
        """Should produce different number of chunks with different chunk_size."""
        settings_small = Mock()
        settings_small.splitter = Mock()
        settings_small.splitter.provider = "recursive"
        settings_small.splitter.chunk_size = 20

        settings_large = Mock()
        settings_large.splitter = Mock()
        settings_large.splitter.provider = "recursive"
        settings_large.splitter.chunk_size = 100

        with patch('src.ingestion.chunking.document_chunker.SplitterFactory') as mock_factory:
            mock_factory.create.side_effect = [
                FakeSplitter(chunk_size=20),
                FakeSplitter(chunk_size=100)
            ]

            chunker_small = DocumentChunker(settings_small)
            chunks_small = chunker_small.split_document(sample_document)

            chunker_large = DocumentChunker(settings_large)
            chunks_large = chunker_large.split_document(sample_document)

            assert len(chunks_small) > len(chunks_large)
