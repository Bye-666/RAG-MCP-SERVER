"""Citation generator for retrieval results."""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Citation:
    """Citation information for a retrieval result."""
    chunk_id: str
    source: str
    page: int
    title: str
    score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'source': self.source,
            'page': self.page,
            'title': self.title,
            'score': self.score,
            'snippet': self.snippet
        }


class CitationGenerator:
    """Generates citation information from retrieval results."""

    def __init__(self, snippet_length: int = 150):
        """
        Initialize citation generator.

        Args:
            snippet_length: Maximum length of text snippet in citation
        """
        self.snippet_length = snippet_length

    def generate(self, retrieval_results: List[Dict[str, Any]]) -> List[Citation]:
        """
        Generate citations from retrieval results.

        Args:
            retrieval_results: List of retrieval results with chunk_id, score, text, metadata

        Returns:
            List of Citation objects with source, page, title, etc.
        """
        citations = []

        for result in retrieval_results:
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 0)
            title = metadata.get('title', 'Untitled')

            # Create text snippet
            text = result.get('text', '')
            snippet = self._create_snippet(text)

            citation = Citation(
                chunk_id=result.get('chunk_id', ''),
                source=source,
                page=page,
                title=title,
                score=result.get('score', 0.0),
                snippet=snippet
            )
            citations.append(citation)

        return citations

    def _create_snippet(self, text: str) -> str:
        """Create a text snippet with ellipsis if needed."""
        if len(text) <= self.snippet_length:
            return text
        return text[:self.snippet_length].rsplit(' ', 1)[0] + '...'
