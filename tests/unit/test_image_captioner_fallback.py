"""
Unit tests for ImageCaptioner.

Tests cover:
- Vision LLM mode with image captioning
- Fallback mode when Vision LLM is disabled
- Error handling and graceful degradation
- Chunks without images (pass-through)
- Configuration switches
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_vision_llm import ChatResponse


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_settings_vision_disabled():
    """Settings with Vision LLM disabled"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.image_captioner = Mock()
    settings.ingestion.image_captioner.use_vision = False
    return settings


@pytest.fixture
def mock_settings_vision_enabled():
    """Settings with Vision LLM enabled"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.image_captioner = Mock()
    settings.ingestion.image_captioner.use_vision = True
    return settings


@pytest.fixture
def chunk_without_images():
    """Chunk with no image references"""
    return Chunk(
        id="chunk_001",
        text="This is a text-only chunk with no images.",
        metadata={"source_path": "test.pdf", "chunk_index": 0}
    )


@pytest.fixture
def chunk_with_images():
    """Chunk with image references"""
    return Chunk(
        id="chunk_002",
        text="This chunk has an image: [IMAGE: img_001]",
        metadata={
            "source_path": "test.pdf",
            "chunk_index": 1,
            "image_refs": ["img_001"],
            "images": [
                {
                    "image_id": "img_001",
                    "path": "data/images/test_doc/img_001.png"
                }
            ]
        }
    )


@pytest.fixture
def chunk_with_multiple_images():
    """Chunk with multiple image references"""
    return Chunk(
        id="chunk_003",
        text="Multiple images: [IMAGE: img_001] and [IMAGE: img_002]",
        metadata={
            "source_path": "test.pdf",
            "chunk_index": 2,
            "image_refs": ["img_001", "img_002"],
            "images": [
                {
                    "image_id": "img_001",
                    "path": "data/images/test_doc/img_001.png"
                },
                {
                    "image_id": "img_002",
                    "path": "data/images/test_doc/img_002.png"
                }
            ]
        }
    )


# ============================================================================
# Pass-through tests (no images)
# ============================================================================

def test_chunk_without_images_passthrough(mock_settings_vision_disabled, chunk_without_images):
    """Test that chunks without images pass through unchanged"""
    captioner = ImageCaptioner(mock_settings_vision_disabled)

    result = captioner.transform([chunk_without_images])

    assert len(result) == 1
    assert result[0].id == chunk_without_images.id
    assert result[0].text == chunk_without_images.text
    assert "image_captions" not in result[0].metadata
    assert "has_unprocessed_images" not in result[0].metadata


# ============================================================================
# Vision disabled mode tests
# ============================================================================

def test_vision_disabled_marks_unprocessed(mock_settings_vision_disabled, chunk_with_images):
    """Test that disabled Vision LLM marks images as unprocessed"""
    captioner = ImageCaptioner(mock_settings_vision_disabled)

    result = captioner.transform([chunk_with_images])

    assert len(result) == 1
    assert result[0].metadata["has_unprocessed_images"] is True
    assert result[0].metadata["unprocessed_image_refs"] == ["img_001"]
    assert "image_captions" not in result[0].metadata


def test_vision_disabled_multiple_images(mock_settings_vision_disabled, chunk_with_multiple_images):
    """Test that all images are marked as unprocessed when Vision is disabled"""
    captioner = ImageCaptioner(mock_settings_vision_disabled)

    result = captioner.transform([chunk_with_multiple_images])

    assert result[0].metadata["has_unprocessed_images"] is True
    assert set(result[0].metadata["unprocessed_image_refs"]) == {"img_001", "img_002"}


# ============================================================================
# Vision enabled mode tests
# ============================================================================

def test_vision_enabled_generates_captions(mock_settings_vision_enabled, chunk_with_images, tmp_path):
    """Test that Vision LLM generates captions for images"""
    # Create mock image file
    image_path = tmp_path / "data" / "images" / "test_doc"
    image_path.mkdir(parents=True)
    (image_path / "img_001.png").write_bytes(b"fake image data")

    # Update chunk metadata with correct path
    chunk_with_images.metadata["images"][0]["path"] = str(image_path / "img_001.png")

    # Mock Vision LLM
    mock_vision_llm = Mock()
    mock_vision_llm.chat_with_image.return_value = ChatResponse(
        content="A diagram showing the system architecture with multiple components."
    )

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    result = captioner.transform([chunk_with_images])

    assert len(result) == 1
    assert "image_captions" in result[0].metadata
    assert "img_001" in result[0].metadata["image_captions"]
    assert "system architecture" in result[0].metadata["image_captions"]["img_001"]
    assert result[0].metadata["captioned_by"] == "vision_llm"
    assert "has_unprocessed_images" not in result[0].metadata


def test_vision_enabled_multiple_images(mock_settings_vision_enabled, chunk_with_multiple_images, tmp_path):
    """Test captioning multiple images in one chunk"""
    # Create mock image files
    image_path = tmp_path / "data" / "images" / "test_doc"
    image_path.mkdir(parents=True)
    (image_path / "img_001.png").write_bytes(b"fake image 1")
    (image_path / "img_002.png").write_bytes(b"fake image 2")

    # Update chunk metadata
    chunk_with_multiple_images.metadata["images"][0]["path"] = str(image_path / "img_001.png")
    chunk_with_multiple_images.metadata["images"][1]["path"] = str(image_path / "img_002.png")

    # Mock Vision LLM with different responses
    mock_vision_llm = Mock()
    mock_vision_llm.chat_with_image.side_effect = [
        ChatResponse(content="First image caption"),
        ChatResponse(content="Second image caption")
    ]

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    result = captioner.transform([chunk_with_multiple_images])

    assert len(result[0].metadata["image_captions"]) == 2
    assert result[0].metadata["image_captions"]["img_001"] == "First image caption"
    assert result[0].metadata["image_captions"]["img_002"] == "Second image caption"
    assert mock_vision_llm.chat_with_image.call_count == 2


# ============================================================================
# Fallback and error handling tests
# ============================================================================

def test_vision_llm_initialization_failure(mock_settings_vision_enabled, chunk_with_images):
    """Test graceful fallback when Vision LLM initialization fails"""
    with patch('src.ingestion.transform.image_captioner.LLMFactory.create_vision_llm',
               side_effect=Exception("Init failed")):
        captioner = ImageCaptioner(mock_settings_vision_enabled)

        # Should fall back to disabled mode
        assert captioner.use_vision is False
        assert captioner.vision_llm is None

        result = captioner.transform([chunk_with_images])
        assert result[0].metadata["has_unprocessed_images"] is True


def test_vision_llm_call_failure_partial_fallback(mock_settings_vision_enabled, chunk_with_multiple_images, tmp_path):
    """Test partial fallback when some images fail to caption"""
    # Create mock image files
    image_path = tmp_path / "data" / "images" / "test_doc"
    image_path.mkdir(parents=True)
    (image_path / "img_001.png").write_bytes(b"fake image 1")
    (image_path / "img_002.png").write_bytes(b"fake image 2")

    chunk_with_multiple_images.metadata["images"][0]["path"] = str(image_path / "img_001.png")
    chunk_with_multiple_images.metadata["images"][1]["path"] = str(image_path / "img_002.png")

    # Mock Vision LLM: first succeeds, second fails
    mock_vision_llm = Mock()
    mock_vision_llm.chat_with_image.side_effect = [
        ChatResponse(content="First image caption"),
        Exception("API error")
    ]

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    result = captioner.transform([chunk_with_multiple_images])

    # Should have one caption and one unprocessed
    assert "image_captions" in result[0].metadata
    assert "img_001" in result[0].metadata["image_captions"]
    assert result[0].metadata["has_unprocessed_images"] is True
    assert "img_002" in result[0].metadata["unprocessed_image_refs"]


def test_missing_image_file(mock_settings_vision_enabled, chunk_with_images):
    """Test handling when image file doesn't exist"""
    # Don't create the image file
    mock_vision_llm = Mock()

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    result = captioner.transform([chunk_with_images])

    # Should mark as unprocessed
    assert result[0].metadata["has_unprocessed_images"] is True
    assert "img_001" in result[0].metadata["unprocessed_image_refs"]
    assert "image_captions" not in result[0].metadata
    # Vision LLM should not be called
    mock_vision_llm.chat_with_image.assert_not_called()


def test_chunk_processing_error_preserves_original(mock_settings_vision_enabled, chunk_with_images):
    """Test that chunk processing errors preserve original chunk"""
    mock_vision_llm = Mock()

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    # Force an error in _process_chunk
    with patch.object(captioner, '_process_chunk', side_effect=Exception("Processing error")):
        result = captioner.transform([chunk_with_images])

        assert len(result) == 1
        assert result[0].id == chunk_with_images.id
        assert result[0].text == chunk_with_images.text
        assert "image_processing_error" in result[0].metadata
        assert result[0].metadata["has_unprocessed_images"] is True


# ============================================================================
# Configuration tests
# ============================================================================

def test_custom_prompt_loading(mock_settings_vision_disabled, tmp_path):
    """Test loading custom prompt template"""
    custom_prompt = tmp_path / "custom_prompt.txt"
    custom_prompt.write_text("Custom image description prompt")

    captioner = ImageCaptioner(mock_settings_vision_disabled, prompt_path=str(custom_prompt))

    assert "Custom image description prompt" in captioner.prompt_template


def test_prompt_loading_fallback(mock_settings_vision_disabled):
    """Test fallback prompt when file doesn't exist"""
    captioner = ImageCaptioner(mock_settings_vision_disabled, prompt_path="nonexistent.txt")

    # Should use fallback prompt
    assert "Describe this image" in captioner.prompt_template


# ============================================================================
# Batch processing tests
# ============================================================================

def test_batch_processing_mixed_chunks(mock_settings_vision_enabled, chunk_without_images, chunk_with_images, tmp_path):
    """Test processing batch with mixed chunks (with and without images)"""
    # Create mock image file
    image_path = tmp_path / "data" / "images" / "test_doc"
    image_path.mkdir(parents=True)
    (image_path / "img_001.png").write_bytes(b"fake image")

    chunk_with_images.metadata["images"][0]["path"] = str(image_path / "img_001.png")

    mock_vision_llm = Mock()
    mock_vision_llm.chat_with_image.return_value = ChatResponse(content="Test caption")

    captioner = ImageCaptioner(mock_settings_vision_enabled, vision_llm=mock_vision_llm)

    result = captioner.transform([chunk_without_images, chunk_with_images])

    assert len(result) == 2
    # First chunk unchanged
    assert "image_captions" not in result[0].metadata
    # Second chunk captioned
    assert "image_captions" in result[1].metadata


# ============================================================================
# Trace context tests
# ============================================================================

def test_trace_context_recording(mock_settings_vision_disabled, chunk_without_images):
    """Test that trace context is properly recorded"""
    captioner = ImageCaptioner(mock_settings_vision_disabled)
    trace = TraceContext(trace_id="test_trace")

    result = captioner.transform([chunk_without_images], trace=trace)

    assert len(result) == 1
    # Trace should have recorded the stage
    stages = [s.stage_name for s in trace.stages]
    assert "image_captioner" in stages


# ============================================================================
# Integration contract tests
# ============================================================================

def test_original_chunk_data_preserved(mock_settings_vision_disabled, chunk_with_images):
    """Test that original chunk data is preserved"""
    captioner = ImageCaptioner(mock_settings_vision_disabled)

    original_text = chunk_with_images.text
    original_id = chunk_with_images.id
    original_source_ref = chunk_with_images.source_ref

    result = captioner.transform([chunk_with_images])

    processed = result[0]
    assert processed.text == original_text
    assert processed.id == original_id
    assert processed.source_ref == original_source_ref
    # Original metadata should be preserved
    assert processed.metadata["source_path"] == "test.pdf"
    assert processed.metadata["chunk_index"] == 1
    assert processed.metadata["image_refs"] == ["img_001"]
