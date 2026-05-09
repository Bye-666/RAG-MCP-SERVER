"""
Unit tests for QueryProcessor.

Tests keyword extraction, stopword filtering, and filter parsing.
"""

import pytest
from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery


class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass"""

    def test_processed_query_creation(self):
        """Test creating ProcessedQuery with all fields"""
        pq = ProcessedQuery(
            raw_query="test query",
            keywords=["test", "query"],
            filters={"collection": "docs"}
        )
        assert pq.raw_query == "test query"
        assert pq.keywords == ["test", "query"]
        assert pq.filters == {"collection": "docs"}

    def test_processed_query_default_filters(self):
        """Test ProcessedQuery with default empty filters"""
        pq = ProcessedQuery(
            raw_query="test",
            keywords=["test"]
        )
        assert pq.filters == {}


class TestQueryProcessor:
    """Tests for QueryProcessor"""

    def test_basic_keyword_extraction(self):
        """Test basic keyword extraction from simple query"""
        processor = QueryProcessor()
        result = processor.process("How to configure Azure OpenAI?")

        assert result.raw_query == "How to configure Azure OpenAI?"
        assert "configure" in result.keywords
        assert "azure" in result.keywords
        assert "openai" in result.keywords
        # Stopwords should be filtered
        assert "how" not in result.keywords
        assert "to" not in result.keywords

    def test_stopword_filtering(self):
        """Test that stopwords are filtered out"""
        processor = QueryProcessor()
        result = processor.process("the quick brown fox")

        assert "quick" in result.keywords
        assert "brown" in result.keywords
        assert "fox" in result.keywords
        assert "the" not in result.keywords

    def test_min_keyword_length(self):
        """Test minimum keyword length filtering"""
        processor = QueryProcessor(min_keyword_length=3)
        result = processor.process("a big cat is here")

        assert "big" in result.keywords
        assert "cat" in result.keywords
        # "is" is stopword, "a" is too short
        assert "is" not in result.keywords
        assert "a" not in result.keywords

    def test_custom_stopwords(self):
        """Test using custom stopword set"""
        custom_stopwords = {"test", "example"}
        processor = QueryProcessor(stopwords=custom_stopwords)
        result = processor.process("test example query")

        assert "query" in result.keywords
        assert "test" not in result.keywords
        assert "example" not in result.keywords

    def test_max_keywords_limit(self):
        """Test limiting maximum number of keywords"""
        processor = QueryProcessor(max_keywords=3)
        result = processor.process("one two three four five six")

        assert len(result.keywords) == 3
        assert result.keywords == ["one", "two", "three"]

    def test_case_insensitive(self):
        """Test that keyword extraction is case-insensitive"""
        processor = QueryProcessor()
        result = processor.process("Azure OPENAI Configuration")

        assert "azure" in result.keywords
        assert "openai" in result.keywords
        assert "configuration" in result.keywords

    def test_special_characters_removed(self):
        """Test that special characters are handled correctly"""
        processor = QueryProcessor()
        result = processor.process("What's the API-key configuration?")

        assert "api" in result.keywords
        assert "key" in result.keywords
        assert "configuration" in result.keywords

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError"""
        processor = QueryProcessor()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            processor.process("")

    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises ValueError"""
        processor = QueryProcessor()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            processor.process("   ")

    def test_filters_passthrough(self):
        """Test that filters are passed through correctly"""
        processor = QueryProcessor()
        filters = {"collection": "docs", "doc_type": "pdf"}
        result = processor.process("test query", filters=filters)

        assert result.filters == filters

    def test_no_filters_returns_empty_dict(self):
        """Test that no filters returns empty dict"""
        processor = QueryProcessor()
        result = processor.process("test query")

        assert result.filters == {}

    def test_all_stopwords_query(self):
        """Test query with only stopwords returns empty keywords"""
        processor = QueryProcessor()
        result = processor.process("the and or but")

        assert result.keywords == []

    def test_numeric_keywords(self):
        """Test that numeric tokens are included"""
        processor = QueryProcessor()
        result = processor.process("Python 3.9 configuration")

        assert "python" in result.keywords
        assert "configuration" in result.keywords
        # Single digit numbers are filtered by min_keyword_length=2
        # Only multi-character tokens pass through

    def test_unicode_characters(self):
        """Test handling of unicode characters"""
        processor = QueryProcessor()
        result = processor.process("配置 Azure 服务")

        # Chinese characters should be tokenized
        assert "配置" in result.keywords
        assert "azure" in result.keywords
        assert "服务" in result.keywords

    def test_hyphenated_words(self):
        """Test handling of hyphenated words"""
        processor = QueryProcessor()
        result = processor.process("multi-modal AI system")

        # Hyphenated words are split
        assert "multi" in result.keywords
        assert "modal" in result.keywords
        assert "system" in result.keywords
