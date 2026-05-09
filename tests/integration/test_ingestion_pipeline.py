"""
Integration tests for IngestionPipeline with trace support.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.core.types import Document, Chunk, ChunkRecord
from src.core.trace import TraceContext
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.loader.base_loader import BaseLoader
from src.ingestion.chunking.document_chunker import DocumentChunker


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary test file"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for ingestion pipeline.")
    return str(test_file)


@pytest.fixture
def mock_integrity_checker(tmp_path):
    """Create mock integrity checker"""
    checker = Mock(spec=FileIntegrityChecker)
    checker.compute_sha256 = Mock(return_value="test_hash_123")
    checker.should_skip = Mock(return_value=False)
    checker.mark_success = Mock()
    checker.mark_failed = Mock()
    return checker


@pytest.fixture
def mock_loader():
    """Create mock loader"""
    loader = Mock(spec=BaseLoader)
    loader.load = Mock(return_value=Document(
        id="doc_123",
        text="This is a test document for ingestion pipeline.",
        metadata={"source_path": "/path/to/test.txt"}
    ))
    return loader


@pytest.fixture
def mock_chunker():
    """Create mock chunker"""
    chunker = Mock(spec=DocumentChunker)
    chunker.split_document = Mock(return_value=[
        Chunk(id="chunk_1", text="This is a test", metadata={}),
        Chunk(id="chunk_2", text="document for ingestion", metadata={}),
        Chunk(id="chunk_3", text="pipeline.", metadata={}),
    ])
    return chunker


@pytest.fixture
def mock_batch_processor():
    """Create mock batch processor"""
    processor = Mock()
    processor.process = Mock(return_value=[
        ChunkRecord(id="chunk_1", text="This is a test", metadata={}, dense_vector=[0.1]*768),
        ChunkRecord(id="chunk_2", text="document for ingestion", metadata={}, dense_vector=[0.2]*768),
        ChunkRecord(id="chunk_3", text="pipeline.", metadata={}, dense_vector=[0.3]*768),
    ])
    return processor


@pytest.fixture
def mock_vector_upserter():
    """Create mock vector upserter"""
    upserter = Mock()
    upserter.upsert = Mock(return_value=["chunk_1", "chunk_2", "chunk_3"])
    return upserter


class TestIngestionPipelineWithTrace:
    """Test IngestionPipeline with trace support"""

    def test_ingest_with_trace(self, temp_file, mock_integrity_checker, mock_loader,
                                mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test ingestion with trace context"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Verify ingestion succeeded
        assert not result["skipped"]
        assert result["chunk_count"] == 3
        assert result["error"] is None

        # Verify trace type
        trace_dict = trace.to_dict()
        assert trace_dict["trace_type"] == "ingestion"

    def test_trace_contains_all_stages(self, temp_file, mock_integrity_checker, mock_loader,
                                       mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that trace contains all required stages"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Verify all required stages exist
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "load" in stage_names
        assert "split" in stage_names
        assert "embed" in stage_names
        assert "upsert" in stage_names

    def test_trace_stages_have_method_field(self, temp_file, mock_integrity_checker, mock_loader,
                                            mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that each stage records method field"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Verify each stage has method field
        for stage in trace.stages:
            assert "method" in stage.metadata, f"Stage {stage.stage_name} missing method field"

    def test_trace_stages_have_elapsed_ms(self, temp_file, mock_integrity_checker, mock_loader,
                                          mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that each stage records elapsed time"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Verify each stage has duration
        for stage in trace.stages:
            assert stage.duration_ms is not None, f"Stage {stage.stage_name} missing duration"
            assert stage.duration_ms >= 0

    def test_trace_serialization(self, temp_file, mock_integrity_checker, mock_loader,
                                 mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that trace can be serialized to dict"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
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

    def test_trace_with_transforms(self, temp_file, mock_integrity_checker, mock_loader,
                                   mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test trace with transform stages"""
        # Mock transform
        mock_transform = Mock()
        mock_transform.transform = Mock(side_effect=lambda chunks, trace: chunks)

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            transforms=[mock_transform],
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        config = PipelineConfig(enable_transforms=True)
        result = pipeline.ingest_file(temp_file, config=config, trace=trace)
        trace.finish()

        # Verify transform stage exists
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "transform" in stage_names

    def test_trace_without_batch_processor(self, temp_file, mock_integrity_checker, mock_loader,
                                          mock_chunker, mock_vector_upserter):
        """Test trace when batch processor is not available"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=None,  # No batch processor
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Should not have embed stage
        stage_names = [stage.stage_name for stage in trace.stages]
        assert "embed" not in stage_names

    def test_ingest_without_trace(self, temp_file, mock_integrity_checker, mock_loader,
                                  mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that ingestion works without trace"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        result = pipeline.ingest_file(temp_file, trace=None)

        # Should work normally
        assert not result["skipped"]
        assert result["chunk_count"] == 3

    def test_trace_records_processing_details(self, temp_file, mock_integrity_checker, mock_loader,
                                              mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that trace records processing details"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter
        )

        trace = TraceContext(trace_type="ingestion")
        result = pipeline.ingest_file(temp_file, trace=trace)
        trace.finish()

        # Check load stage details
        load_stage = next(s for s in trace.stages if s.stage_name == "load")
        assert "doc_id" in load_stage.metadata

        # Check split stage details
        split_stage = next(s for s in trace.stages if s.stage_name == "split")
        assert "chunk_count" in split_stage.metadata

        # Check embed stage details
        embed_stage = next(s for s in trace.stages if s.stage_name == "embed")
        assert "record_count" in embed_stage.metadata
