"""
Unit tests for ChunkRefiner.

Tests cover:
- Rule-based refinement patterns
- LLM mode with mocking
- Fallback behavior
- Configuration switches
- Error handling
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


# Load test fixtures
@pytest.fixture
def noisy_chunks():
    """Load noisy chunk test cases from fixtures"""
    fixture_path = Path("tests/fixtures/noisy_chunks.json")
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def mock_settings_rule_only():
    """Settings with LLM disabled (rule-based only)"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = Mock()
    settings.ingestion.chunk_refiner.use_llm = False
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Settings with LLM enabled"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = Mock()
    settings.ingestion.chunk_refiner.use_llm = True
    settings.llm = Mock()
    settings.llm.provider = "openai"
    settings.llm.model = "gpt-4o"
    return settings


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing"""
    return Chunk(
        id="test_doc_0001_abc123",
        text="  Page 1 of 10  \n\nThis is test content.\n\n  Footer: Copyright 2024  ",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
        source_ref="test_doc"
    )


# ============================================================================
# Rule-based refinement tests
# ============================================================================

def test_rule_based_excessive_whitespace(mock_settings_rule_only, noisy_chunks):
    """Test removal of excessive whitespace"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["excessive_whitespace"]["input"]
    chunk = Chunk(id="test_001", text=input_text, metadata={})

    result = refiner.transform([chunk])

    assert len(result) == 1
    refined_text = result[0].text

    # Should collapse multiple spaces
    assert "    " not in refined_text
    # Should reduce excessive newlines
    assert "\n\n\n" not in refined_text
    # Should preserve content
    assert "This text has too many spaces" in refined_text


def test_rule_based_page_headers_footers(mock_settings_rule_only, noisy_chunks):
    """Test removal of page headers and footers"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["page_header_footer"]["input"]
    chunk = Chunk(id="test_002", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should remove header/footer patterns
    assert "Header:" not in refined_text
    assert "Footer:" not in refined_text
    # Should preserve main content
    assert "actual content starts here" in refined_text


def test_rule_based_html_comments(mock_settings_rule_only, noisy_chunks):
    """Test removal of HTML comments"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["format_markers"]["input"]
    chunk = Chunk(id="test_003", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should remove HTML comments
    assert "<!--" not in refined_text
    assert "-->" not in refined_text
    # Should preserve content
    assert "Some text with HTML" in refined_text


def test_rule_based_preserves_clean_text(mock_settings_rule_only, noisy_chunks):
    """Test that clean text is not over-processed"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["clean_text"]["input"]
    chunk = Chunk(id="test_004", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should be nearly identical (maybe just trimmed)
    assert "already clean text" in refined_text
    assert "proper formatting" in refined_text


def test_rule_based_preserves_code_blocks(mock_settings_rule_only, noisy_chunks):
    """Test that code block formatting is preserved"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["code_blocks"]["input"]
    chunk = Chunk(id="test_005", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should preserve code block markers
    assert "```python" in refined_text
    assert "```" in refined_text
    # Should preserve indentation in code
    assert "    print" in refined_text


def test_rule_based_typical_noise(mock_settings_rule_only, noisy_chunks):
    """Test handling of typical combined noise"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["typical_noise_scenario"]["input"]
    chunk = Chunk(id="test_006", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should remove page numbers
    assert "Page 1 of 10" not in refined_text
    # Should remove footer
    assert "Footer:" not in refined_text
    # Should preserve main content
    assert "main content of the document" in refined_text


def test_rule_based_mixed_noise(mock_settings_rule_only, noisy_chunks):
    """Test handling of mixed real-world noise"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    input_text = noisy_chunks["mixed_noise"]["input"]
    chunk = Chunk(id="test_007", text=input_text, metadata={})

    result = refiner.transform([chunk])
    refined_text = result[0].text

    # Should clean various noise types
    assert "Page 3 of 20" not in refined_text
    assert "<!--" not in refined_text
    assert "Footer Text" not in refined_text
    # Should preserve content
    assert "real content" in refined_text


def test_rule_based_empty_text(mock_settings_rule_only):
    """Test handling of empty or whitespace-only text that becomes empty after cleaning"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    # Use a chunk with minimal valid text that becomes empty after cleaning
    chunk = Chunk(id="test_008", text="Page 1 of 10", metadata={})

    result = refiner.transform([chunk])

    # Should handle gracefully - returns original chunk when refinement produces empty text
    assert len(result) == 1
    assert result[0].text == "Page 1 of 10"  # Original preserved
    assert result[0].id == chunk.id


# ============================================================================
# LLM mode tests (with mocking)
# ============================================================================

def test_llm_mode_successful_refinement(mock_settings_with_llm, sample_chunk):
    """Test LLM refinement when LLM call succeeds"""
    mock_llm = Mock()
    mock_llm.generate.return_value = "This is LLM-refined content."

    refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    assert len(result) == 1
    assert result[0].text == "This is LLM-refined content."
    assert result[0].metadata["refined_by"] == "llm"
    assert "fallback_reason" not in result[0].metadata

    # Verify LLM was called
    mock_llm.generate.assert_called_once()


def test_llm_mode_fallback_on_error(mock_settings_with_llm, sample_chunk):
    """Test fallback to rule-based when LLM fails"""
    mock_llm = Mock()
    mock_llm.generate.side_effect = Exception("LLM API error")

    refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    assert len(result) == 1
    # Should have rule-based result
    assert result[0].text != sample_chunk.text  # Was refined
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"


def test_llm_mode_fallback_on_empty_response(mock_settings_with_llm, sample_chunk):
    """Test fallback when LLM returns empty response"""
    mock_llm = Mock()
    mock_llm.generate.return_value = ""

    refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    assert len(result) == 1
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"


def test_llm_mode_fallback_on_none_response(mock_settings_with_llm, sample_chunk):
    """Test fallback when LLM returns None"""
    mock_llm = Mock()
    mock_llm.generate.return_value = None

    refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    assert len(result) == 1
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"


# ============================================================================
# Configuration and initialization tests
# ============================================================================

def test_config_switch_llm_disabled(mock_settings_rule_only, sample_chunk):
    """Test that LLM is not used when disabled in config"""
    mock_llm = Mock()

    refiner = ChunkRefiner(mock_settings_rule_only, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    # LLM should not be called
    mock_llm.generate.assert_not_called()
    assert result[0].metadata["refined_by"] == "rule"


def test_config_switch_llm_enabled(mock_settings_with_llm, sample_chunk):
    """Test that LLM is used when enabled in config"""
    mock_llm = Mock()
    mock_llm.generate.return_value = "LLM output"

    refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)

    result = refiner.transform([sample_chunk])

    # LLM should be called
    mock_llm.generate.assert_called_once()
    assert result[0].metadata["refined_by"] == "llm"


def test_prompt_loading_default(mock_settings_rule_only):
    """Test that default prompt is loaded"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    assert refiner.prompt_template is not None
    assert "{text}" in refiner.prompt_template


def test_prompt_loading_custom(mock_settings_rule_only, tmp_path):
    """Test loading custom prompt file"""
    custom_prompt = tmp_path / "custom_prompt.txt"
    custom_prompt.write_text("Custom prompt: {text}")

    refiner = ChunkRefiner(mock_settings_rule_only, prompt_path=str(custom_prompt))

    assert "Custom prompt" in refiner.prompt_template


def test_prompt_loading_fallback_on_missing_file(mock_settings_rule_only):
    """Test fallback when prompt file doesn't exist"""
    refiner = ChunkRefiner(mock_settings_rule_only, prompt_path="/nonexistent/path.txt")

    # Should use fallback prompt
    assert refiner.prompt_template is not None
    assert "{text}" in refiner.prompt_template


# ============================================================================
# Error handling and edge cases
# ============================================================================

def test_error_handling_single_chunk_failure(mock_settings_rule_only):
    """Test that single chunk error doesn't affect others"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    chunks = [
        Chunk(id="test_001", text="Good chunk 1", metadata={}),
        Chunk(id="test_002", text="Good chunk 2", metadata={}),
    ]

    # Mock _refine_chunk to fail on second chunk
    original_refine = refiner._refine_chunk
    def mock_refine(chunk, trace=None):
        if chunk.id == "test_002":
            raise Exception("Processing error")
        return original_refine(chunk, trace)

    refiner._refine_chunk = mock_refine

    result = refiner.transform(chunks)

    # Should return both chunks (second one unchanged due to error)
    assert len(result) == 2
    assert result[0].text == "Good chunk 1"
    assert result[1].text == "Good chunk 2"  # Original preserved


def test_batch_processing_multiple_chunks(mock_settings_rule_only, noisy_chunks):
    """Test processing multiple chunks in one call"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    chunks = [
        Chunk(id=f"test_{i:03d}", text=data["input"], metadata={})
        for i, (key, data) in enumerate(noisy_chunks.items())
    ]

    result = refiner.transform(chunks)

    assert len(result) == len(chunks)
    # All should be refined
    for chunk in result:
        assert "refined_by" in chunk.metadata


def test_metadata_preservation(mock_settings_rule_only, sample_chunk):
    """Test that original metadata is preserved"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    result = refiner.transform([sample_chunk])

    # Original metadata should be preserved
    assert result[0].metadata["source_path"] == "test.pdf"
    assert result[0].metadata["chunk_index"] == 0
    # New metadata should be added
    assert "refined_by" in result[0].metadata


def test_trace_context_recording(mock_settings_rule_only, sample_chunk):
    """Test that operations are recorded in trace context"""
    refiner = ChunkRefiner(mock_settings_rule_only)
    trace = TraceContext()

    refiner.transform([sample_chunk], trace=trace)

    # Should have recorded a stage
    assert len(trace.stages) > 0
    assert any(stage.stage_name == "chunk_refiner" for stage in trace.stages)


def test_source_ref_preservation(mock_settings_rule_only, sample_chunk):
    """Test that source_ref is preserved"""
    refiner = ChunkRefiner(mock_settings_rule_only)

    result = refiner.transform([sample_chunk])

    assert result[0].source_ref == sample_chunk.source_ref
    assert result[0].id == sample_chunk.id
