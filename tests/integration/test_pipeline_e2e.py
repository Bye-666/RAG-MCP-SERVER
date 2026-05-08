"""
End-to-end integration tests for ingestion pipeline.

Tests the complete flow: integrity → load → split → transform → encode → store
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.core.types import Chunk, ChunkRecord


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    temp_root = tempfile.mkdtemp()
    db_dir = Path(temp_root) / "db"
    images_dir = Path(temp_root) / "images"
    db_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    yield {
        "root": temp_root,
        "db": str(db_dir),
        "images": str(images_dir)
    }

    shutil.rmtree(temp_root)


@pytest.fixture
def sample_pdf(temp_dirs):
    """Create a sample PDF file for testing"""
    pdf_path = Path(temp_dirs["root"]) / "test_doc.pdf"
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000304 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
398
%%EOF
"""
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


@pytest.fixture
def pipeline_components(temp_dirs):
    """Create all pipeline components"""
    # Integrity checker
    integrity_checker = SQLiteIntegrityChecker(db_path=f"{temp_dirs['db']}/integrity.db")

    # Loader
    loader = PdfLoader()

    # Chunker - create mock settings with proper structure
    mock_settings = Mock()
    mock_settings.splitter = {
        "splitter": {
            "provider": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
    }
    chunker = DocumentChunker(settings=mock_settings)

    return {
        "integrity_checker": integrity_checker,
        "loader": loader,
        "chunker": chunker
    }


def test_pipeline_basic_flow(sample_pdf, pipeline_components):
    """Test basic pipeline flow with minimal stages"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    result = pipeline.ingest_file(sample_pdf)

    assert result["skipped"] is False
    assert result["file_hash"] is not None
    assert result["chunk_count"] > 0
    assert result["error"] is None


def test_pipeline_skip_already_processed(sample_pdf, pipeline_components):
    """Test that pipeline skips already processed files"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    # First ingestion
    result1 = pipeline.ingest_file(sample_pdf)
    assert result1["skipped"] is False

    # Second ingestion should skip
    result2 = pipeline.ingest_file(sample_pdf)
    assert result2["skipped"] is True
    assert result2["file_hash"] == result1["file_hash"]


def test_pipeline_force_reprocess(sample_pdf, pipeline_components):
    """Test force reprocess flag"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    # First ingestion
    result1 = pipeline.ingest_file(sample_pdf)
    assert result1["skipped"] is False

    # Force reprocess
    config = PipelineConfig(force_reprocess=True)
    result2 = pipeline.ingest_file(sample_pdf, config=config)
    assert result2["skipped"] is False


def test_pipeline_without_transforms(sample_pdf, pipeline_components):
    """Test pipeline without transform stage"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    config = PipelineConfig(enable_transforms=False)
    result = pipeline.ingest_file(sample_pdf, config=config)

    assert result["error"] is None
    assert result["chunk_count"] > 0


def test_pipeline_without_encoding(sample_pdf, pipeline_components):
    """Test pipeline without encoding stage"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    result = pipeline.ingest_file(sample_pdf)

    assert result["error"] is None
    assert result["chunk_count"] > 0


def test_pipeline_progress_callback(sample_pdf, pipeline_components):
    """Test progress callback is called"""
    progress_events = []

    def on_progress(stage, details):
        progress_events.append({"stage": stage, "details": details})

    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"],
        on_progress=on_progress
    )

    pipeline.ingest_file(sample_pdf)

    # Check that progress events were recorded
    assert len(progress_events) > 0
    stages = [e["stage"] for e in progress_events]
    assert "integrity_check" in stages
    assert "load" in stages
    assert "split" in stages
    assert "completed" in stages


def test_pipeline_file_not_found(pipeline_components):
    """Test error handling for non-existent file"""
    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"]
    )

    with pytest.raises(FileNotFoundError):
        pipeline.ingest_file("/nonexistent/file.pdf")


def test_pipeline_error_handling(sample_pdf, pipeline_components):
    """Test error handling during processing"""
    # Create a loader that raises an error
    bad_loader = Mock()
    bad_loader.load.side_effect = Exception("Load failed")

    pipeline = IngestionPipeline(
        integrity_checker=pipeline_components["integrity_checker"],
        loader=bad_loader,
        chunker=pipeline_components["chunker"]
    )

    with pytest.raises(Exception, match="Load failed"):
        pipeline.ingest_file(sample_pdf)

    # Check that failure was recorded
    file_hash = pipeline_components["integrity_checker"].compute_sha256(sample_pdf)
    assert pipeline_components["integrity_checker"].should_skip(file_hash) is False
