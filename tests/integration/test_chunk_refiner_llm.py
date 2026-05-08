"""
Integration tests for ChunkRefiner with real LLM.

These tests verify:
1. Real LLM configuration is correct
2. LLM refinement produces quality output
3. Fallback mechanism works with invalid configuration

⚠️ WARNING: Tests marked with @pytest.mark.skip require real LLM API keys.
"""

import pytest
from unittest.mock import Mock

from src.core.types import Chunk
from src.core.settings import load_settings
from src.ingestion.transform.chunk_refiner import ChunkRefiner


@pytest.fixture
def base_settings():
    """Create mock settings for testing"""
    settings = Mock()

    # Mock LLM settings
    settings.llm = Mock()
    settings.llm.provider = "openai"
    settings.llm.model = "gpt-4o"

    # Mock ingestion configuration
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = Mock()
    settings.ingestion.chunk_refiner.use_llm = False  # Default to rule-based

    return settings


@pytest.fixture
def noisy_chunk():
    """Create a chunk with typical noise for testing"""
    return Chunk(
        id="integration_test_001",
        text="""
        Page 5 of 20

        <!-- HTML comment -->

        This is   a   test   document   with   excessive   spacing.

        It contains    multiple    types    of    noise    that    need    cleaning.

        Header: Document Title

        The actual content should be preserved and cleaned properly.

        Footer: Copyright 2024
        """,
        metadata={"source_path": "test.pdf", "chunk_index": 0}
    )


@pytest.mark.integration
def test_rule_based_refinement_integration(base_settings, noisy_chunk):
    """Test that rule-based refinement works in integration context"""
    from src.core.trace import TraceContext

    base_settings.ingestion.chunk_refiner.use_llm = False

    refiner = ChunkRefiner(base_settings)
    trace = TraceContext()

    result = refiner.transform([noisy_chunk], trace)

    assert len(result) == 1
    refined_chunk = result[0]

    # Verify noise is removed by rules
    assert "Page 5 of 20" not in refined_chunk.text
    assert "<!-- HTML comment -->" not in refined_chunk.text

    # Verify content is preserved
    assert "actual content" in refined_chunk.text.lower()

    # Verify original metadata is preserved (plus refined_by added)
    assert refined_chunk.metadata["source_path"] == noisy_chunk.metadata["source_path"]
    assert refined_chunk.metadata["chunk_index"] == noisy_chunk.metadata["chunk_index"]
    assert refined_chunk.metadata["refined_by"] == "rule"
    assert refined_chunk.id == noisy_chunk.id


@pytest.mark.integration
@pytest.mark.skip(reason="Requires real LLM API key - run manually with: pytest -m integration --run-llm")
def test_llm_refinement_with_real_config(base_settings, noisy_chunk):
    """Test that ChunkRefiner works with real LLM configuration"""
    from src.core.trace import TraceContext

    base_settings.ingestion.chunk_refiner.use_llm = True

    refiner = ChunkRefiner(base_settings)
    trace = TraceContext()

    result = refiner.transform([noisy_chunk], trace)

    assert len(result) == 1
    refined_chunk = result[0]

    # Verify noise is removed
    assert "Page 5 of 20" not in refined_chunk.text
    assert "<!-- HTML comment -->" not in refined_chunk.text
    assert "Footer: Copyright 2024" not in refined_chunk.text

    # Verify content is preserved
    assert "actual content" in refined_chunk.text.lower()

    # Verify original metadata is preserved
    assert refined_chunk.metadata["source_path"] == noisy_chunk.metadata["source_path"]
    assert refined_chunk.metadata["chunk_index"] == noisy_chunk.metadata["chunk_index"]
    assert refined_chunk.id == noisy_chunk.id


@pytest.mark.integration
def test_fallback_on_invalid_llm_config(noisy_chunk):
    """Test that ChunkRefiner falls back to rule-based when LLM config is invalid"""
    from src.core.trace import TraceContext

    # Create settings with invalid LLM config
    invalid_settings = Mock()
    invalid_settings.llm = Mock()
    invalid_settings.llm.provider = "invalid_provider"
    invalid_settings.ingestion = Mock()
    invalid_settings.ingestion.chunk_refiner = Mock()
    invalid_settings.ingestion.chunk_refiner.use_llm = True

    refiner = ChunkRefiner(invalid_settings)
    trace = TraceContext()

    # Should not raise exception, should fall back to rule-based
    result = refiner.transform([noisy_chunk], trace)

    assert len(result) == 1
    # Rule-based cleaning should still work
    assert "Page 5 of 20" not in result[0].text


@pytest.mark.integration
def test_batch_refinement(base_settings):
    """Test refining multiple chunks in batch"""
    from src.core.trace import TraceContext

    base_settings.ingestion.chunk_refiner.use_llm = False

    chunks = [
        Chunk(id=f"batch_{i}", text=f"Page {i}\n\nContent {i}", metadata={})
        for i in range(5)
    ]

    refiner = ChunkRefiner(base_settings)
    trace = TraceContext()

    result = refiner.transform(chunks, trace)

    assert len(result) == 5
    for i, chunk in enumerate(result):
        assert chunk.id == f"batch_{i}"
        assert f"Page {i}" not in chunk.text  # Noise removed
        assert f"Content {i}" in chunk.text  # Content preserved


@pytest.mark.integration
def test_empty_chunk_handling(base_settings):
    """Test that empty chunks are handled gracefully"""
    from src.core.trace import TraceContext

    base_settings.ingestion.chunk_refiner.use_llm = False

    chunk = Chunk(
        id="empty_test",
        text="Page 1 of 10",  # Only noise, no real content
        metadata={}
    )

    refiner = ChunkRefiner(base_settings)
    trace = TraceContext()

    result = refiner.transform([chunk], trace)

    assert len(result) == 1
    # Should return original chunk if refinement produces empty text
    assert result[0].text == chunk.text
