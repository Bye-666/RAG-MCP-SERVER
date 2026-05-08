"""
Unit tests for MetadataEnricher.

Tests cover:
- Rule-based metadata generation (title, summary, tags)
- LLM mode with mocking
- Fallback behavior
- Configuration switches
- Error handling
- JSON parsing from LLM responses
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.metadata_enricher import MetadataEnricher


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_settings_rule_only():
    """Settings with LLM disabled (rule-based only)"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = Mock()
    settings.ingestion.metadata_enricher.use_llm = False
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Settings with LLM enabled"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = Mock()
    settings.ingestion.metadata_enricher.use_llm = True

    # Make llm settings subscriptable like a dict
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o"
    }
    settings.llm = Mock()
    settings.llm.__getitem__ = lambda self, key: llm_config[key]
    settings.llm.get = lambda key, default=None: llm_config.get(key, default)

    return settings


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing"""
    return Chunk(
        id="test_doc_0001_abc123",
        text="# Introduction to RAG\n\nRetrieval-Augmented Generation (RAG) is a technique that combines retrieval and generation.",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
        source_ref="test_doc"
    )


@pytest.fixture
def technical_chunk():
    """Chunk with technical content"""
    return Chunk(
        id="test_doc_0002_def456",
        text="""# API Authentication

This section covers OAuth 2.0 authentication for the REST API.
The API uses JWT tokens for secure access. All requests must include
an Authorization header with a valid bearer token.""",
        metadata={"source_path": "api_docs.pdf", "chunk_index": 1},
        source_ref="api_doc"
    )


# ============================================================================
# Rule-based enrichment tests
# ============================================================================

def test_rule_based_title_extraction_from_heading(mock_settings_rule_only, sample_chunk):
    """Test title extraction from markdown heading"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    result = enricher.transform([sample_chunk])

    assert len(result) == 1
    assert "title" in result[0].metadata
    assert result[0].metadata["title"] == "Introduction to RAG"
    assert result[0].metadata["enriched_by"] == "rule"


def test_rule_based_title_from_first_line(mock_settings_rule_only):
    """Test title extraction when no heading present"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    chunk = Chunk(
        id="test_001",
        text="This is the first line without heading.\nSecond line here.",
        metadata={}
    )

    result = enricher.transform([chunk])

    assert result[0].metadata["title"] == "This is the first line without heading."


def test_rule_based_summary_generation(mock_settings_rule_only, sample_chunk):
    """Test summary generation"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    result = enricher.transform([sample_chunk])

    assert "summary" in result[0].metadata
    summary = result[0].metadata["summary"]
    assert len(summary) <= 200
    assert "Retrieval-Augmented Generation" in summary or "RAG" in summary


def test_rule_based_tags_extraction(mock_settings_rule_only, technical_chunk):
    """Test tag extraction from technical content"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    result = enricher.transform([technical_chunk])

    assert "tags" in result[0].metadata
    tags = result[0].metadata["tags"]
    assert isinstance(tags, list)
    assert len(tags) <= 5
    # Should extract technical keywords
    tags_lower = [tag.lower() for tag in tags]
    assert any(keyword in tags_lower for keyword in ["api", "oauth", "jwt", "rest", "authentication"])


def test_rule_based_minimal_chunk(mock_settings_rule_only):
    """Test handling of minimal chunk with very short text"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    chunk = Chunk(id="test_minimal", text="Short.", metadata={})

    result = enricher.transform([chunk])

    # Should still generate metadata
    assert "title" in result[0].metadata
    assert "summary" in result[0].metadata
    assert "tags" in result[0].metadata
    assert result[0].metadata["title"] == "Short."


def test_metadata_fields_always_present(mock_settings_rule_only, sample_chunk):
    """Test that all required metadata fields are always present"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    result = enricher.transform([sample_chunk])

    metadata = result[0].metadata
    assert "title" in metadata
    assert "summary" in metadata
    assert "tags" in metadata
    assert isinstance(metadata["title"], str)
    assert isinstance(metadata["summary"], str)
    assert isinstance(metadata["tags"], list)


# ============================================================================
# LLM mode tests
# ============================================================================

def test_llm_mode_successful_enrichment(mock_settings_with_llm, sample_chunk):
    """Test successful LLM-based enrichment"""
    # Create enricher with mock LLM directly injected
    mock_llm = Mock()
    llm_response = json.dumps({
        "title": "RAG: Retrieval-Augmented Generation Overview",
        "summary": "Introduction to RAG technique combining retrieval and generation for enhanced AI responses.",
        "tags": ["rag", "retrieval", "generation", "ai", "nlp"]
    })
    mock_llm.generate.return_value = llm_response

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    assert len(result) == 1
    metadata = result[0].metadata
    assert metadata["enriched_by"] == "llm"
    assert metadata["title"] == "RAG: Retrieval-Augmented Generation Overview"
    assert "RAG technique" in metadata["summary"]
    assert "rag" in metadata["tags"]
    assert len(metadata["tags"]) == 5


def test_llm_mode_with_markdown_wrapped_json(mock_settings_with_llm, sample_chunk):
    """Test parsing JSON wrapped in markdown code blocks"""
    # Mock LLM response with markdown wrapper
    mock_llm = Mock()
    llm_response = """```json
{
    "title": "Test Title",
    "summary": "Test summary content",
    "tags": ["test", "metadata"]
}
```"""
    mock_llm.generate.return_value = llm_response

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    assert result[0].metadata["enriched_by"] == "llm"
    assert result[0].metadata["title"] == "Test Title"


def test_llm_mode_fallback_on_invalid_json(mock_settings_with_llm, sample_chunk):
    """Test fallback to rule-based when LLM returns invalid JSON"""
    # Mock LLM with invalid response
    mock_llm = Mock()
    mock_llm.generate.return_value = "This is not valid JSON"

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    # Should fall back to rule-based
    assert result[0].metadata["enriched_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"
    assert "title" in result[0].metadata
    assert "summary" in result[0].metadata
    assert "tags" in result[0].metadata


def test_llm_mode_fallback_on_missing_fields(mock_settings_with_llm, sample_chunk):
    """Test fallback when LLM response missing required fields"""
    # Mock LLM with incomplete response
    mock_llm = Mock()
    llm_response = json.dumps({
        "title": "Test Title"
        # Missing summary and tags
    })
    mock_llm.generate.return_value = llm_response

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    assert result[0].metadata["enriched_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"


def test_llm_mode_fallback_on_exception(mock_settings_with_llm, sample_chunk):
    """Test fallback when LLM raises exception"""
    # Mock LLM that raises exception
    mock_llm = Mock()
    mock_llm.generate.side_effect = Exception("LLM API error")

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    assert result[0].metadata["enriched_by"] == "rule"
    assert result[0].metadata["fallback_reason"] == "llm_failed"


def test_llm_mode_truncates_long_fields(mock_settings_with_llm, sample_chunk):
    """Test that LLM responses are truncated to max lengths"""
    # Mock LLM with overly long fields
    mock_llm = Mock()
    llm_response = json.dumps({
        "title": "A" * 200,  # 200 chars, should be truncated to 100
        "summary": "B" * 300,  # 300 chars, should be truncated to 200
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"]  # 7 tags, should be truncated to 5
    })
    mock_llm.generate.return_value = llm_response

    enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

    result = enricher.transform([sample_chunk])

    metadata = result[0].metadata
    assert len(metadata["title"]) == 100
    assert len(metadata["summary"]) == 200
    assert len(metadata["tags"]) == 5


# ============================================================================
# Configuration and initialization tests
# ============================================================================

def test_llm_disabled_in_settings(mock_settings_rule_only, sample_chunk):
    """Test that LLM is not used when disabled in settings"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    assert enricher.use_llm is False
    assert enricher.llm is None

    result = enricher.transform([sample_chunk])
    assert result[0].metadata["enriched_by"] == "rule"


def test_llm_initialization_failure_falls_back(mock_settings_with_llm, sample_chunk):
    """Test that LLM initialization failure doesn't break enrichment"""
    with patch('src.ingestion.transform.metadata_enricher.LLMFactory.create', side_effect=Exception("Init failed")):
        enricher = MetadataEnricher(mock_settings_with_llm)

        # Should fall back to rule-based
        assert enricher.use_llm is False
        assert enricher.llm is None

        result = enricher.transform([sample_chunk])
        assert result[0].metadata["enriched_by"] == "rule"


def test_custom_prompt_loading(mock_settings_rule_only, tmp_path):
    """Test loading custom prompt template"""
    # Create custom prompt file
    custom_prompt = tmp_path / "custom_prompt.txt"
    custom_prompt.write_text("Custom prompt: {text}")

    enricher = MetadataEnricher(mock_settings_rule_only, prompt_path=str(custom_prompt))

    assert "Custom prompt:" in enricher.prompt_template


def test_prompt_loading_fallback_on_missing_file(mock_settings_rule_only):
    """Test fallback prompt when file doesn't exist"""
    enricher = MetadataEnricher(mock_settings_rule_only, prompt_path="nonexistent.txt")

    # Should use fallback prompt
    assert "{text}" in enricher.prompt_template
    assert "JSON" in enricher.prompt_template


# ============================================================================
# Batch processing tests
# ============================================================================

def test_batch_processing_multiple_chunks(mock_settings_rule_only):
    """Test processing multiple chunks in batch"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    chunks = [
        Chunk(id=f"chunk_{i}", text=f"# Heading {i}\n\nContent {i}", metadata={})
        for i in range(5)
    ]

    result = enricher.transform(chunks)

    assert len(result) == 5
    for i, chunk in enumerate(result):
        assert "title" in chunk.metadata
        assert "summary" in chunk.metadata
        assert "tags" in chunk.metadata
        assert f"Heading {i}" in chunk.metadata["title"]


def test_error_handling_preserves_other_chunks(mock_settings_rule_only):
    """Test that error in one chunk doesn't affect others"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    chunks = [
        Chunk(id="chunk_1", text="# Valid Chunk 1\n\nContent", metadata={}),
        Chunk(id="chunk_2", text="# Valid Chunk 2\n\nContent", metadata={}),
    ]

    # Mock _enrich_chunk to fail on second chunk
    original_enrich = enricher._enrich_chunk
    def mock_enrich(chunk, trace=None):
        if chunk.id == "chunk_2":
            raise Exception("Processing error")
        return original_enrich(chunk, trace)

    enricher._enrich_chunk = mock_enrich

    result = enricher.transform(chunks)

    # Should still return 2 chunks
    assert len(result) == 2
    # First chunk should be enriched normally
    assert result[0].metadata["enriched_by"] == "rule"
    # Second chunk should have error metadata
    assert "enrichment_error" in result[1].metadata


# ============================================================================
# Trace context tests
# ============================================================================

def test_trace_context_recording(mock_settings_rule_only, sample_chunk):
    """Test that trace context is properly recorded"""
    enricher = MetadataEnricher(mock_settings_rule_only)
    trace = TraceContext(trace_id="test_trace")

    result = enricher.transform([sample_chunk], trace=trace)

    assert len(result) == 1
    # Trace should have recorded the stage
    stages = [s.stage_name for s in trace.stages]
    assert "metadata_enricher" in stages


# ============================================================================
# Integration tests
# ============================================================================

def test_original_chunk_data_preserved(mock_settings_rule_only, sample_chunk):
    """Test that original chunk data is preserved"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    original_text = sample_chunk.text
    original_id = sample_chunk.id
    original_source_ref = sample_chunk.source_ref

    result = enricher.transform([sample_chunk])

    enriched = result[0]
    assert enriched.text == original_text
    assert enriched.id == original_id
    assert enriched.source_ref == original_source_ref
    # Original metadata should be preserved
    assert enriched.metadata["source_path"] == "test.pdf"
    assert enriched.metadata["chunk_index"] == 0


def test_metadata_enrichment_contract(mock_settings_rule_only):
    """Test that enrichment satisfies the contract: always produces title/summary/tags"""
    enricher = MetadataEnricher(mock_settings_rule_only)

    # Test various edge cases
    test_cases = [
        Chunk(id="normal", text="# Normal\n\nContent here", metadata={}),
        Chunk(id="whitespace", text="   Some text   \n\n   ", metadata={}),
        Chunk(id="no_heading", text="Just plain text without structure", metadata={}),
        Chunk(id="code_only", text="```python\nprint('hello')\n```", metadata={}),
    ]

    for chunk in test_cases:
        result = enricher.transform([chunk])
        metadata = result[0].metadata

        # Contract: must have these fields
        assert "title" in metadata, f"Missing title for {chunk.id}"
        assert "summary" in metadata, f"Missing summary for {chunk.id}"
        assert "tags" in metadata, f"Missing tags for {chunk.id}"

        # Contract: must be correct types
        assert isinstance(metadata["title"], str), f"Title not string for {chunk.id}"
        assert isinstance(metadata["summary"], str), f"Summary not string for {chunk.id}"
        assert isinstance(metadata["tags"], list), f"Tags not list for {chunk.id}"

        # Contract: must be non-None
        assert metadata["title"] is not None, f"Title is None for {chunk.id}"
        assert metadata["summary"] is not None, f"Summary is None for {chunk.id}"
        assert metadata["tags"] is not None, f"Tags is None for {chunk.id}"
