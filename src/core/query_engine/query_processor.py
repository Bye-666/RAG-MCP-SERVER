"""
Query Processor for keyword extraction and filter parsing.

Processes raw user queries into structured format for retrieval:
- Extracts keywords using tokenization and stopword filtering
- Parses metadata filters (currently placeholder implementation)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ProcessedQuery:
    """Structured representation of a processed query

    Attributes:
        raw_query: Original user query string
        keywords: Extracted keywords for retrieval
        filters: Metadata filters (dict format, can be empty)
    """
    raw_query: str
    keywords: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)


class QueryProcessor:
    """
    Processes raw queries into structured format for retrieval.

    Current implementation:
    - Simple tokenization with regex
    - Basic stopword filtering
    - Placeholder filter parsing (returns empty dict)

    Future enhancements:
    - Query expansion (synonyms/aliases)
    - Advanced filter parsing from natural language
    - Language-specific tokenization
    """

    # Basic English stopwords
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or'
    }

    def __init__(
        self,
        stopwords: Optional[set] = None,
        min_keyword_length: int = 2,
        max_keywords: Optional[int] = None
    ):
        """
        Initialize QueryProcessor.

        Args:
            stopwords: Custom stopword set (uses DEFAULT_STOPWORDS if None)
            min_keyword_length: Minimum length for keywords (default: 2)
            max_keywords: Maximum number of keywords to return (None = unlimited)
        """
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_keyword_length = min_keyword_length
        self.max_keywords = max_keywords

    def process(self, query: str, filters: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """
        Process raw query into structured format.

        Args:
            query: Raw user query string
            filters: Optional metadata filters (passed through as-is)

        Returns:
            ProcessedQuery with extracted keywords and filters

        Raises:
            ValueError: If query is empty or whitespace-only
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Use provided filters or empty dict
        processed_filters = filters if filters is not None else {}

        return ProcessedQuery(
            raw_query=query,
            keywords=keywords,
            filters=processed_filters
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query using tokenization and filtering.

        Strategy:
        1. Lowercase the query
        2. Tokenize using regex (alphanumeric sequences)
        3. Filter by length and stopwords
        4. Limit to max_keywords if specified

        Args:
            query: Raw query string

        Returns:
            List of extracted keywords
        """
        # Lowercase and tokenize
        query_lower = query.lower()
        tokens = re.findall(r'\b\w+\b', query_lower)

        # Filter tokens
        keywords = [
            token for token in tokens
            if len(token) >= self.min_keyword_length
            and token not in self.stopwords
        ]

        # Apply max_keywords limit if specified
        if self.max_keywords is not None and len(keywords) > self.max_keywords:
            keywords = keywords[:self.max_keywords]

        return keywords
