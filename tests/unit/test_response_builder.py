"""Tests for ResponseBuilder."""

import pytest
import json
from src.core.response.response_builder import ResponseBuilder, MCPResponse
from src.core.response.citation_generator import CitationGenerator


class TestResponseBuilder:
    """Test ResponseBuilder functionality."""

    @pytest.fixture
    def response_builder(self):
        """Create ResponseBuilder instance."""
        return ResponseBuilder()

    @pytest.fixture
    def sample_results(self):
        """Sample retrieval results."""
        return [
            {
                'chunk_id': 'chunk1',
                'score': 0.95,
                'text': 'This is the first relevant document about machine learning.',
                'metadata': {'source': 'ml_guide.pdf', 'page': 1}
            },
            {
                'chunk_id': 'chunk2',
                'score': 0.87,
                'text': 'Neural networks are a key component of deep learning systems.',
                'metadata': {'source': 'dl_intro.pdf', 'page': 5}
            }
        ]

    def test_build_with_results(self, response_builder, sample_results):
        """Test building response with results."""
        response = response_builder.build(sample_results, "machine learning")

        assert isinstance(response, MCPResponse)
        assert response.isError is False
        assert len(response.content) == 2

        # Check text content
        text_content = response.content[0]
        assert text_content['type'] == 'text'
        assert 'machine learning' in text_content['text']
        assert '[1]' in text_content['text']
        assert '[2]' in text_content['text']

        # Check resource content
        resource_content = response.content[1]
        assert resource_content['type'] == 'resource'
        assert resource_content['resource']['mimeType'] == 'application/json'

        # Parse citations
        citations_data = json.loads(resource_content['resource']['text'])
        assert 'citations' in citations_data
        assert len(citations_data['citations']) == 2

    def test_build_empty_results(self, response_builder):
        """Test building response with empty results."""
        response = response_builder.build([], "test query")

        assert isinstance(response, MCPResponse)
        assert response.isError is False
        assert len(response.content) == 1

        text_content = response.content[0]
        assert text_content['type'] == 'text'
        assert 'No relevant documents found' in text_content['text']
        assert 'test query' in text_content['text']

    def test_markdown_format(self, response_builder, sample_results):
        """Test markdown formatting."""
        response = response_builder.build(sample_results, "test")

        text = response.content[0]['text']

        # Check headers
        assert '# Query Results: test' in text
        assert '## [1]' in text
        assert '## [2]' in text

        # Check metadata
        assert 'ml_guide.pdf' in text
        assert 'Page 1' in text
        assert 'Relevance Score:' in text

    def test_structured_citations(self, response_builder, sample_results):
        """Test structured citations format."""
        response = response_builder.build(sample_results, "test")

        citations_json = response.content[1]['resource']['text']
        citations_data = json.loads(citations_json)

        assert len(citations_data['citations']) == 2

        # Check first citation
        citation1 = citations_data['citations'][0]
        assert citation1['index'] == 1
        assert citation1['chunk_id'] == 'chunk1'
        assert citation1['source'] == 'ml_guide.pdf'
        assert citation1['page'] == 1
        assert citation1['score'] == 0.95

    def test_custom_citation_generator(self):
        """Test with custom citation generator."""
        custom_generator = CitationGenerator(snippet_length=50)
        builder = ResponseBuilder(citation_generator=custom_generator)

        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': 'A' * 100,
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        response = builder.build(results, "test")
        text = response.content[0]['text']

        # Snippet should be truncated
        assert '...' in text

    def test_missing_metadata(self, response_builder):
        """Test handling missing metadata."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.8,
                'text': 'Test text',
                'metadata': {}
            }
        ]

        response = response_builder.build(results, "test")

        citations_json = response.content[1]['resource']['text']
        citations_data = json.loads(citations_json)

        citation = citations_data['citations'][0]
        assert citation['source'] == 'Unknown'
        assert citation['page'] == 0

    def test_multiple_results(self, response_builder):
        """Test with multiple results."""
        results = [
            {
                'chunk_id': f'chunk{i}',
                'score': 0.9 - i * 0.1,
                'text': f'Document {i}',
                'metadata': {'source': f'doc{i}.pdf', 'page': i}
            }
            for i in range(5)
        ]

        response = response_builder.build(results, "test")

        citations_json = response.content[1]['resource']['text']
        citations_data = json.loads(citations_json)

        assert len(citations_data['citations']) == 5

        # Check indices are sequential
        for i, citation in enumerate(citations_data['citations'], start=1):
            assert citation['index'] == i

    def test_special_characters_in_query(self, response_builder, sample_results):
        """Test query with special characters."""
        query = "What is \"machine learning\"?"
        response = response_builder.build(sample_results, query)

        text = response.content[0]['text']
        assert query in text

    def test_empty_query(self, response_builder, sample_results):
        """Test with empty query string."""
        response = response_builder.build(sample_results, "")

        assert response.isError is False
        assert len(response.content) == 2

    def test_long_text_snippet(self, response_builder):
        """Test text snippet truncation."""
        long_text = "word " * 100  # 500 characters
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': long_text,
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        response = response_builder.build(results, "test")
        text = response.content[0]['text']

        # Should be truncated with ellipsis
        assert '...' in text
        assert len(text) < len(long_text) + 200  # Some overhead for formatting

    def test_score_formatting(self, response_builder):
        """Test score formatting in markdown."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.123456789,
                'text': 'Test',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        response = response_builder.build(results, "test")
        text = response.content[0]['text']

        # Score should be formatted to 4 decimal places
        assert '0.1235' in text

    def test_resource_uri(self, response_builder, sample_results):
        """Test resource URI format."""
        response = response_builder.build(sample_results, "test")

        resource = response.content[1]['resource']
        assert resource['uri'] == 'citations://query-results'
        assert resource['mimeType'] == 'application/json'
