"""Tests for CitationGenerator."""

import pytest
from src.core.response.citation_generator import CitationGenerator, Citation


class TestCitationGenerator:
    """Test CitationGenerator functionality."""

    @pytest.fixture
    def generator(self):
        """Create CitationGenerator instance."""
        return CitationGenerator()

    @pytest.fixture
    def sample_results(self):
        """Sample retrieval results."""
        return [
            {
                'chunk_id': 'chunk1',
                'score': 0.95,
                'text': 'This is a test document about machine learning.',
                'metadata': {'source': 'ml_guide.pdf', 'page': 1, 'title': 'ML Guide'}
            },
            {
                'chunk_id': 'chunk2',
                'score': 0.87,
                'text': 'Neural networks are powerful models.',
                'metadata': {'source': 'nn_intro.pdf', 'page': 5, 'title': 'Neural Networks'}
            }
        ]

    def test_generate_citations(self, generator, sample_results):
        """Test generating citations from results."""
        citations = generator.generate(sample_results)

        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)

    def test_citation_fields(self, generator, sample_results):
        """Test citation fields are populated correctly."""
        citations = generator.generate(sample_results)

        citation = citations[0]
        assert citation.chunk_id == 'chunk1'
        assert citation.source == 'ml_guide.pdf'
        assert citation.page == 1
        assert citation.title == 'ML Guide'
        assert citation.score == 0.95
        assert citation.snippet == 'This is a test document about machine learning.'

    def test_snippet_truncation(self):
        """Test snippet truncation."""
        generator = CitationGenerator(snippet_length=20)
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': 'This is a very long text that should be truncated.',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert len(citations[0].snippet) <= 23  # 20 + '...'
        assert citations[0].snippet.endswith('...')

    def test_missing_metadata(self, generator):
        """Test handling missing metadata fields."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.8,
                'text': 'Test text',
                'metadata': {}
            }
        ]

        citations = generator.generate(results)
        citation = citations[0]

        assert citation.source == 'Unknown'
        assert citation.page == 0
        assert citation.title == 'Untitled'

    def test_partial_metadata(self, generator):
        """Test handling partial metadata."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.8,
                'text': 'Test text',
                'metadata': {'source': 'test.pdf'}
            }
        ]

        citations = generator.generate(results)
        citation = citations[0]

        assert citation.source == 'test.pdf'
        assert citation.page == 0
        assert citation.title == 'Untitled'

    def test_empty_results(self, generator):
        """Test with empty results list."""
        citations = generator.generate([])
        assert citations == []

    def test_custom_snippet_length(self):
        """Test custom snippet length."""
        generator = CitationGenerator(snippet_length=50)
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': 'A' * 100,
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert len(citations[0].snippet) <= 53  # 50 + '...'

    def test_short_text_no_truncation(self, generator):
        """Test short text is not truncated."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': 'Short text',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert citations[0].snippet == 'Short text'
        assert not citations[0].snippet.endswith('...')

    def test_multiple_results(self, generator):
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

        citations = generator.generate(results)
        assert len(citations) == 5

        for i, citation in enumerate(citations):
            assert citation.chunk_id == f'chunk{i}'
            assert citation.source == f'doc{i}.pdf'
            assert citation.page == i

    def test_special_characters_in_text(self, generator):
        """Test handling special characters."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': 'Text with "quotes" and <tags>',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert citations[0].snippet == 'Text with "quotes" and <tags>'

    def test_unicode_text(self, generator):
        """Test handling unicode characters."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.9,
                'text': '中文测试 Unicode test 日本語',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert '中文测试' in citations[0].snippet

    def test_citation_to_dict(self, generator, sample_results):
        """Test Citation to_dict method."""
        citations = generator.generate(sample_results)
        citation_dict = citations[0].to_dict()

        assert isinstance(citation_dict, dict)
        assert citation_dict['chunk_id'] == 'chunk1'
        assert citation_dict['source'] == 'ml_guide.pdf'
        assert citation_dict['page'] == 1
        assert citation_dict['title'] == 'ML Guide'
        assert citation_dict['score'] == 0.95
        assert 'snippet' in citation_dict

    def test_score_precision(self, generator):
        """Test score is preserved with precision."""
        results = [
            {
                'chunk_id': 'chunk1',
                'score': 0.123456789,
                'text': 'Test',
                'metadata': {'source': 'test.pdf', 'page': 1}
            }
        ]

        citations = generator.generate(results)
        assert citations[0].score == 0.123456789
