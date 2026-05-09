"""
Unit tests for IngestionPipeline progress callback functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, call

from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.core.types import Document, Chunk, ChunkRecord
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.loader.base_loader import BaseLoader
from src.ingestion.chunking.document_chunker import DocumentChunker


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary test file"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for progress callback testing.")
    return str(test_file)


@pytest.fixture
def mock_integrity_checker():
    """Create mock integrity checker"""
    checker = Mock(spec=FileIntegrityChecker)
    checker.compute_sha256 = Mock(return_value="test_hash_456")
    checker.should_skip = Mock(return_value=False)
    checker.mark_success = Mock()
    checker.mark_failed = Mock()
    return checker


@pytest.fixture
def mock_loader():
    """Create mock loader"""
    loader = Mock(spec=BaseLoader)
    loader.load = Mock(return_value=Document(
        id="doc_456",
        text="This is a test document for progress callback testing.",
        metadata={"source_path": "/path/to/test.txt"}
    ))
    return loader


@pytest.fixture
def mock_chunker():
    """Create mock chunker"""
    chunker = Mock(spec=DocumentChunker)
    chunker.split_document = Mock(return_value=[
        Chunk(id="chunk_1", text="This is a test", metadata={}),
        Chunk(id="chunk_2", text="document for progress", metadata={}),
        Chunk(id="chunk_3", text="callback testing.", metadata={}),
    ])
    return chunker


@pytest.fixture
def mock_batch_processor():
    """Create mock batch processor"""
    processor = Mock()
    processor.process = Mock(return_value=[
        ChunkRecord(id="chunk_1", text="This is a test", metadata={}, dense_vector=[0.1]*768),
        ChunkRecord(id="chunk_2", text="document for progress", metadata={}, dense_vector=[0.2]*768),
        ChunkRecord(id="chunk_3", text="callback testing.", metadata={}, dense_vector=[0.3]*768),
    ])
    return processor


@pytest.fixture
def mock_vector_upserter():
    """Create mock vector upserter"""
    upserter = Mock()
    upserter.upsert = Mock(return_value=["chunk_1", "chunk_2", "chunk_3"])
    return upserter


class TestPipelineProgressCallback:
    """Test IngestionPipeline progress callback functionality"""

    def test_progress_callback_called(self, temp_file, mock_integrity_checker, mock_loader,
                                     mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that progress callback is called during ingestion"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Verify callback was called
        assert progress_callback.call_count > 0

    def test_progress_callback_stages(self, temp_file, mock_integrity_checker, mock_loader,
                                     mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that progress callback is called for all stages"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Extract stage names from calls
        stage_names = [call[0][0] for call in progress_callback.call_args_list]

        # Verify key stages are present
        assert "integrity_check" in stage_names
        assert "load" in stage_names
        assert "split" in stage_names
        assert "encode" in stage_names
        assert "upsert" in stage_names
        assert "completed" in stage_names

    def test_progress_callback_signature(self, temp_file, mock_integrity_checker, mock_loader,
                                        mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that progress callback receives correct arguments"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Verify each call has 3 arguments: stage_name, current, total
        for call_args in progress_callback.call_args_list:
            args = call_args[0]
            assert len(args) == 3, f"Expected 3 args, got {len(args)}"
            stage_name, current, total = args
            assert isinstance(stage_name, str)
            assert isinstance(current, int)
            assert isinstance(total, int)

    def test_progress_callback_current_total_values(self, temp_file, mock_integrity_checker, mock_loader,
                                                   mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that current and total values are reasonable"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Verify current <= total for all calls
        for call_args in progress_callback.call_args_list:
            stage_name, current, total = call_args[0]
            assert current >= 0, f"Stage {stage_name}: current should be >= 0"
            assert total >= 0, f"Stage {stage_name}: total should be >= 0"
            assert current <= total, f"Stage {stage_name}: current ({current}) should be <= total ({total})"

    def test_progress_callback_with_transforms(self, temp_file, mock_integrity_checker, mock_loader,
                                              mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test progress callback with transform stages"""
        progress_callback = Mock()

        # Mock transforms
        mock_transform1 = Mock()
        mock_transform1.transform = Mock(side_effect=lambda chunks, trace: chunks)
        mock_transform2 = Mock()
        mock_transform2.transform = Mock(side_effect=lambda chunks, trace: chunks)

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            transforms=[mock_transform1, mock_transform2],
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        config = PipelineConfig(enable_transforms=True)
        result = pipeline.ingest_file(temp_file, config=config)

        # Extract transform stage calls
        transform_calls = [call for call in progress_callback.call_args_list
                          if call[0][0] == "transform"]

        # Should have 2 transforms * 2 calls (start + end) = 4 calls
        assert len(transform_calls) >= 2, "Should have at least 2 transform progress calls"

    def test_progress_callback_none_does_not_break(self, temp_file, mock_integrity_checker, mock_loader,
                                                   mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that pipeline works normally when on_progress is None"""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=None  # No callback
        )

        result = pipeline.ingest_file(temp_file)

        # Should work normally
        assert not result["skipped"]
        assert result["chunk_count"] == 3
        assert result["error"] is None

    def test_progress_callback_on_skip(self, temp_file, mock_integrity_checker, mock_loader,
                                      mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test progress callback when file is skipped"""
        progress_callback = Mock()

        # Configure to skip file
        mock_integrity_checker.should_skip = Mock(return_value=True)

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Should have integrity_check calls
        stage_names = [call[0][0] for call in progress_callback.call_args_list]
        assert "integrity_check" in stage_names
        assert result["skipped"]

    def test_progress_callback_on_error(self, temp_file, mock_integrity_checker, mock_loader,
                                       mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test progress callback when error occurs"""
        progress_callback = Mock()

        # Configure loader to raise error
        mock_loader.load = Mock(side_effect=Exception("Test error"))

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        with pytest.raises(Exception, match="Test error"):
            pipeline.ingest_file(temp_file)

        # Should have failed stage
        stage_names = [call[0][0] for call in progress_callback.call_args_list]
        assert "failed" in stage_names

    def test_progress_callback_encode_stage_counts(self, temp_file, mock_integrity_checker, mock_loader,
                                                  mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that encode stage reports correct chunk counts"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Find encode stage calls
        encode_calls = [call for call in progress_callback.call_args_list
                       if call[0][0] == "encode"]

        # Should have start (0, 3) and end (3, 3) calls
        assert len(encode_calls) >= 2
        # Last encode call should be (3, 3)
        last_encode = encode_calls[-1][0]
        assert last_encode[1] == 3  # current
        assert last_encode[2] == 3  # total

    def test_progress_callback_upsert_stage_counts(self, temp_file, mock_integrity_checker, mock_loader,
                                                  mock_chunker, mock_batch_processor, mock_vector_upserter):
        """Test that upsert stage reports correct record counts"""
        progress_callback = Mock()

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            on_progress=progress_callback
        )

        result = pipeline.ingest_file(temp_file)

        # Find upsert stage calls
        upsert_calls = [call for call in progress_callback.call_args_list
                       if call[0][0] == "upsert"]

        # Should have start (0, 3) and end (3, 3) calls
        assert len(upsert_calls) >= 2
        # Last upsert call should be (3, 3)
        last_upsert = upsert_calls[-1][0]
        assert last_upsert[1] == 3  # current
        assert last_upsert[2] == 3  # total
