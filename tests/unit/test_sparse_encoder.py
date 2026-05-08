"""Unit tests for SparseEncoder

Tests cover:
- Normal encoding with single and multiple chunks
- Empty chunks list handling
- Tokenization and stopword filtering
- Term weight calculation
- Empty text handling
- Term frequency normalization
"""

import pytest
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.types import Chunk, ChunkRecord


class TestSparseEncoder:
    """Test cases for SparseEncoder class"""

    def test_initialization_default_stopwords(self):
        """Test initialization with default stopwords"""
        encoder = SparseEncoder()
        assert encoder.stopwords is not None
        assert len(encoder.stopwords) > 0
        assert 'the' in encoder.stopwords
        assert 'and' in encoder.stopwords

    def test_initialization_custom_stopwords(self):
        """Test initialization with custom stopwords"""
        custom_stopwords = {'custom', 'stop', 'words'}
        encoder = SparseEncoder(stopwords=custom_stopwords)
        assert encoder.stopwords == custom_stopwords

    def test_initialization_min_term_length(self):
        """Test initialization with custom min_term_length"""
        encoder = SparseEncoder(min_term_length=3)
        assert encoder.min_term_length == 3

    def test_tokenize_basic(self):
        """Test basic tokenization"""
        encoder = SparseEncoder(stopwords=set())
        tokens = encoder._tokenize("Hello world")
        assert tokens == ['hello', 'world']

    def test_tokenize_with_stopwords(self):
        """Test tokenization with stopword filtering"""
        encoder = SparseEncoder(stopwords={'the', 'is'})
        tokens = encoder._tokenize("The cat is here")
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'cat' in tokens
        assert 'here' in tokens

    def test_tokenize_min_length_filter(self):
        """Test tokenization with minimum length filter"""
        encoder = SparseEncoder(stopwords=set(), min_term_length=3)
        tokens = encoder._tokenize("I am a developer")
        # 'I' (1 char) and 'am' (2 chars) should be filtered out
        assert 'i' not in tokens
        assert 'am' not in tokens
        assert 'developer' in tokens

    def test_tokenize_punctuation_removal(self):
        """Test that punctuation is removed during tokenization"""
        encoder = SparseEncoder(stopwords=set())
        tokens = encoder._tokenize("Hello, world! How are you?")
        assert tokens == ['hello', 'world', 'how', 'are', 'you']

    def test_tokenize_empty_text(self):
        """Test tokenization of empty text"""
        encoder = SparseEncoder()
        tokens = encoder._tokenize("")
        assert tokens == []

    def test_compute_term_weights_basic(self):
        """Test basic term weight computation"""
        encoder = SparseEncoder()
        tokens = ['cat', 'dog', 'cat']
        weights = encoder._compute_term_weights(tokens)

        # cat appears 2 times out of 3 tokens: 2/3 ≈ 0.667
        # dog appears 1 time out of 3 tokens: 1/3 ≈ 0.333
        assert abs(weights['cat'] - 2/3) < 1e-6
        assert abs(weights['dog'] - 1/3) < 1e-6

    def test_compute_term_weights_empty_tokens(self):
        """Test term weight computation with empty token list"""
        encoder = SparseEncoder()
        weights = encoder._compute_term_weights([])
        assert weights == {}

    def test_compute_term_weights_single_token(self):
        """Test term weight computation with single token"""
        encoder = SparseEncoder()
        tokens = ['hello']
        weights = encoder._compute_term_weights(tokens)
        assert weights == {'hello': 1.0}

    def test_encode_single_chunk(self):
        """Test encoding a single chunk"""
        encoder = SparseEncoder(stopwords={'the'})

        chunk = Chunk(
            id="test_001",
            text="The cat sat on the mat",
            metadata={"source": "test.pdf"}
        )

        records = encoder.encode([chunk])

        assert len(records) == 1
        assert isinstance(records[0], ChunkRecord)
        assert records[0].id == "test_001"
        assert records[0].text == "The cat sat on the mat"
        assert records[0].sparse_vector is not None
        assert isinstance(records[0].sparse_vector, dict)

        # 'the' should be filtered out
        assert 'the' not in records[0].sparse_vector
        # Other terms should be present
        assert 'cat' in records[0].sparse_vector
        assert 'sat' in records[0].sparse_vector
        assert 'mat' in records[0].sparse_vector

    def test_encode_multiple_chunks(self):
        """Test encoding multiple chunks"""
        encoder = SparseEncoder(stopwords=set())

        chunks = [
            Chunk(id="c1", text="machine learning", metadata={}),
            Chunk(id="c2", text="deep learning neural networks", metadata={}),
            Chunk(id="c3", text="artificial intelligence", metadata={})
        ]

        records = encoder.encode(chunks)

        assert len(records) == 3
        assert records[0].id == "c1"
        assert records[1].id == "c2"
        assert records[2].id == "c3"

        # Each should have different sparse vectors
        assert 'machine' in records[0].sparse_vector
        assert 'learning' in records[0].sparse_vector
        assert 'deep' in records[1].sparse_vector
        assert 'neural' in records[1].sparse_vector
        assert 'artificial' in records[2].sparse_vector

    def test_encode_empty_chunks_list(self):
        """Test that empty chunks list raises ValueError"""
        encoder = SparseEncoder()

        with pytest.raises(ValueError, match="chunks list cannot be empty"):
            encoder.encode([])

    def test_encode_chunk_with_only_stopwords(self):
        """Test encoding chunk with only stopwords (results in empty sparse vector)"""
        encoder = SparseEncoder(stopwords={'the', 'and', 'is', 'a'})

        chunk = Chunk(
            id="stopwords_001",
            text="the and is a",  # Only stopwords
            metadata={}
        )

        records = encoder.encode([chunk])

        assert len(records) == 1
        # All terms are stopwords, so sparse vector should be empty
        assert records[0].sparse_vector == {}

    def test_encode_preserves_chunk_metadata(self):
        """Test that chunk metadata is preserved in ChunkRecord"""
        encoder = SparseEncoder()

        chunk = Chunk(
            id="test_001",
            text="Test document",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "page": 5,
                "title": "Section 1"
            }
        )

        records = encoder.encode([chunk])

        assert records[0].metadata == chunk.metadata
        assert records[0].metadata["source_path"] == "/path/to/doc.pdf"
        assert records[0].metadata["page"] == 5

    def test_encode_dense_vector_remains_none(self):
        """Test that dense_vector field remains None after sparse encoding"""
        encoder = SparseEncoder()

        chunk = Chunk(id="c1", text="Test text", metadata={})
        records = encoder.encode([chunk])

        assert records[0].sparse_vector is not None
        assert records[0].dense_vector is None

    def test_encode_term_frequency_normalization(self):
        """Test that term frequencies are normalized by document length"""
        encoder = SparseEncoder(stopwords=set())

        chunk = Chunk(
            id="c1",
            text="cat cat dog",  # 3 tokens: cat(2), dog(1)
            metadata={}
        )

        records = encoder.encode([chunk])
        weights = records[0].sparse_vector

        # Verify normalization: sum of all term frequencies should equal 1.0
        # cat: 2/3, dog: 1/3
        assert abs(weights['cat'] - 2/3) < 1e-6
        assert abs(weights['dog'] - 1/3) < 1e-6

    def test_encode_case_insensitive(self):
        """Test that encoding is case-insensitive"""
        encoder = SparseEncoder(stopwords=set())

        chunk = Chunk(
            id="c1",
            text="Python PYTHON python",
            metadata={}
        )

        records = encoder.encode([chunk])
        weights = records[0].sparse_vector

        # All variants should be counted as 'python'
        assert 'python' in weights
        assert weights['python'] == 1.0  # 3/3 = 1.0
        assert 'Python' not in weights
        assert 'PYTHON' not in weights

    def test_encode_with_numbers(self):
        """Test encoding text with numbers"""
        encoder = SparseEncoder(stopwords=set(), min_term_length=1)

        chunk = Chunk(
            id="c1",
            text="Python 3 and version 27",
            metadata={}
        )

        records = encoder.encode([chunk])
        weights = records[0].sparse_vector

        # With min_term_length=1, single digit numbers should be included
        assert 'python' in weights
        assert '3' in weights
        assert '27' in weights
        assert 'version' in weights

    def test_encode_output_structure_for_bm25(self):
        """Test that output structure is suitable for BM25 indexer"""
        encoder = SparseEncoder()

        chunks = [
            Chunk(id="c1", text="information retrieval", metadata={}),
            Chunk(id="c2", text="information extraction", metadata={})
        ]

        records = encoder.encode(chunks)

        # Verify structure is suitable for BM25 indexer
        for record in records:
            assert isinstance(record, ChunkRecord)
            assert isinstance(record.sparse_vector, dict)
            # All keys should be strings (terms)
            assert all(isinstance(k, str) for k in record.sparse_vector.keys())
            # All values should be floats (weights)
            assert all(isinstance(v, float) for v in record.sparse_vector.values())
            # All weights should be positive
            assert all(v > 0 for v in record.sparse_vector.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
