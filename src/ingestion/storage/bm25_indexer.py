"""BM25 Indexer for building and persisting inverted indices

This module provides the BM25Indexer class that builds inverted indices
from sparse-encoded chunks, calculates IDF scores, and persists the index
to disk for retrieval.
"""

import os
import pickle
import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from src.core.types import ChunkRecord


@dataclass
class PostingEntry:
    """A single posting in the inverted index

    Attributes:
        chunk_id: ID of the chunk containing this term
        tf: Term frequency (normalized by document length)
        doc_length: Total number of terms in the document
    """
    chunk_id: str
    tf: float
    doc_length: int


@dataclass
class TermIndex:
    """Index entry for a single term

    Attributes:
        term: The term string
        idf: Inverse document frequency score
        postings: List of posting entries for this term
    """
    term: str
    idf: float
    postings: List[PostingEntry] = field(default_factory=list)


class BM25Indexer:
    """Builds and manages BM25 inverted index

    This indexer takes ChunkRecords with sparse vectors (term weights),
    builds an inverted index, calculates IDF scores, and persists the
    index to disk.

    Attributes:
        index_dir: Directory where index files are stored
        index: Dictionary mapping terms to TermIndex objects
        num_documents: Total number of documents in the index
    """

    def __init__(self, index_dir=None, settings=None):
        """Initialize the BM25 indexer

        Args:
            index_dir: Directory to store index files (string) or Settings object for backward compatibility
            settings: Optional Settings object (for keyword argument compatibility)
        """
        # Determine index_dir based on arguments
        final_index_dir = "data/db/bm25"  # default

        # Handle different calling patterns
        if settings is not None:
            # Called with settings=settings keyword argument
            # Use default index_dir (could extract from settings in future)
            pass
        elif index_dir is not None:
            # Check if it's a Settings object (has __dict__ or is not a string)
            if isinstance(index_dir, str):
                # It's an index_dir string
                final_index_dir = index_dir
            else:
                # It's a Settings object passed as first positional arg
                # Use default index_dir
                pass

        self.index_dir = final_index_dir
        self.index: Dict[str, TermIndex] = {}
        self.num_documents: int = 0

        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

    def build(self, records: List[ChunkRecord]) -> None:
        """Build inverted index from chunk records

        Args:
            records: List of ChunkRecord objects with sparse_vector populated

        Raises:
            ValueError: If records list is empty or sparse vectors are missing
        """
        if not records:
            raise ValueError("records list cannot be empty")

        # Verify all records have sparse vectors
        for record in records:
            if record.sparse_vector is None:
                raise ValueError(f"Record {record.id} missing sparse_vector")

        # Clear existing index
        self.index = {}
        self.num_documents = len(records)

        # Build inverted index structure
        term_doc_freq: Dict[str, int] = defaultdict(int)
        term_postings: Dict[str, List[PostingEntry]] = defaultdict(list)

        for record in records:
            sparse_vector = record.sparse_vector
            doc_length = len(sparse_vector)

            # Track which terms appear in this document
            doc_terms = set(sparse_vector.keys())

            # Update document frequency for each unique term
            for term in doc_terms:
                term_doc_freq[term] += 1

            # Create posting entries
            for term, tf in sparse_vector.items():
                posting = PostingEntry(
                    chunk_id=record.id,
                    tf=tf,
                    doc_length=doc_length
                )
                term_postings[term].append(posting)

        # Calculate IDF and build final index
        for term, df in term_doc_freq.items():
            idf = self._calculate_idf(df, self.num_documents)
            self.index[term] = TermIndex(
                term=term,
                idf=idf,
                postings=term_postings[term]
            )

    def _calculate_idf(self, df: int, num_docs: int) -> float:
        """Calculate IDF score for a term

        Uses the BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))

        Args:
            df: Document frequency (number of documents containing the term)
            num_docs: Total number of documents

        Returns:
            IDF score
        """
        return math.log((num_docs - df + 0.5) / (df + 0.5))

    def save(self, filename: str = "index.pkl") -> None:
        """Save index to disk

        Args:
            filename: Name of the index file
        """
        filepath = os.path.join(self.index_dir, filename)

        # Prepare data for serialization
        index_data = {
            'index': self.index,
            'num_documents': self.num_documents
        }

        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)

    def load(self, filename: str = "index.pkl") -> None:
        """Load index from disk

        Args:
            filename: Name of the index file

        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        filepath = os.path.join(self.index_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        self.index = index_data['index']
        self.num_documents = index_data['num_documents']

    def query(self, terms: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Query the index for relevant chunks

        Args:
            terms: List of query terms
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        if not terms:
            return []

        # Accumulate scores for each chunk
        chunk_scores: Dict[str, float] = defaultdict(float)

        for term in terms:
            term_lower = term.lower()
            if term_lower not in self.index:
                continue

            term_index = self.index[term_lower]

            # Add IDF-weighted scores for each posting
            for posting in term_index.postings:
                # Simple scoring: TF * IDF
                score = posting.tf * term_index.idf
                chunk_scores[posting.chunk_id] += score

        # Sort by score descending and return top_k
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:top_k]

    def get_term_info(self, term: str) -> Optional[TermIndex]:
        """Get index information for a specific term

        Args:
            term: The term to look up

        Returns:
            TermIndex object if term exists, None otherwise
        """
        return self.index.get(term.lower())

    def get_stats(self) -> Dict[str, any]:
        """Get index statistics

        Returns:
            Dictionary with index statistics
        """
        return {
            'num_documents': self.num_documents,
            'num_terms': len(self.index),
            'total_postings': sum(len(term_idx.postings) for term_idx in self.index.values())
        }

    def remove_document(self, chunk_ids: List[str]) -> int:
        """Remove documents from the index by chunk IDs

        Args:
            chunk_ids: List of chunk IDs to remove

        Returns:
            Number of postings removed
        """
        if not chunk_ids:
            return 0

        chunk_id_set = set(chunk_ids)
        removed_count = 0
        terms_to_remove = []

        # Remove postings for the specified chunk IDs
        for term, term_index in self.index.items():
            original_count = len(term_index.postings)
            term_index.postings = [
                posting for posting in term_index.postings
                if posting.chunk_id not in chunk_id_set
            ]
            removed_count += original_count - len(term_index.postings)

            # Mark terms with no postings for removal
            if not term_index.postings:
                terms_to_remove.append(term)

        # Remove empty terms
        for term in terms_to_remove:
            del self.index[term]

        # Update document count (approximate - assumes each chunk is a document)
        self.num_documents = max(0, self.num_documents - len(chunk_ids))

        return removed_count
