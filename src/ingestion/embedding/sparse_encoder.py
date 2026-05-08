"""Sparse vector encoder for text chunks using BM25 statistics

This module provides the SparseEncoder class that converts text chunks into
sparse vector representations (term weights) suitable for BM25 indexing.
"""

from typing import List, Dict, Optional
from collections import Counter
import re
from src.core.types import Chunk, ChunkRecord
from src.core.trace import TraceContext


class SparseEncoder:
    """Encodes text chunks into sparse vectors (term weights) for BM25

    This encoder takes a list of Chunk objects and produces ChunkRecord objects
    with sparse_vector populated. The sparse vector is a dictionary mapping
    terms to their term frequency (TF) values.

    Attributes:
        stopwords: Set of stopwords to filter out during tokenization
        min_term_length: Minimum length for a term to be included
    """

    # Common English stopwords
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }

    def __init__(
        self,
        stopwords: Optional[set] = None,
        min_term_length: int = 2
    ):
        """Initialize the sparse encoder

        Args:
            stopwords: Set of stopwords to filter out (uses default if None)
            min_term_length: Minimum length for a term to be included
        """
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_term_length = min_term_length

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms

        Args:
            text: Input text to tokenize

        Returns:
            List of normalized terms (lowercase, filtered)
        """
        # Convert to lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Filter by length and stopwords
        filtered = [
            token for token in tokens
            if len(token) >= self.min_term_length and token not in self.stopwords
        ]

        return filtered

    def _compute_term_weights(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency weights for tokens

        Args:
            tokens: List of tokens from a chunk

        Returns:
            Dictionary mapping terms to their TF values
        """
        if not tokens:
            return {}

        # Count term frequencies
        term_counts = Counter(tokens)
        doc_length = len(tokens)

        # Compute normalized term frequency (TF)
        # TF = (term_count / doc_length)
        term_weights = {
            term: count / doc_length
            for term, count in term_counts.items()
        }

        return term_weights

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """Encode chunks into sparse vectors (term weights)

        Args:
            chunks: List of Chunk objects to encode
            trace: Optional trace context for observability

        Returns:
            List of ChunkRecord objects with sparse_vector populated

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        records = []
        for chunk in chunks:
            # Tokenize the chunk text
            tokens = self._tokenize(chunk.text)

            # Compute term weights
            term_weights = self._compute_term_weights(tokens)

            # Create ChunkRecord with sparse vector
            record = ChunkRecord.from_chunk(
                chunk=chunk,
                sparse_vector=term_weights
            )
            records.append(record)

        return records
