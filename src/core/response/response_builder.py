"""MCP response builder for query results."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from .citation_generator import CitationGenerator, Citation


@dataclass
class MCPResponse:
    """MCP tool response format."""
    content: List[Dict[str, Any]]
    isError: bool = False


class ResponseBuilder:
    """Builds MCP-formatted responses from retrieval results."""

    def __init__(self, citation_generator: Optional[CitationGenerator] = None):
        """
        Initialize response builder.

        Args:
            citation_generator: Citation generator instance (creates default if None)
        """
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: List[Dict[str, Any]], query: str) -> MCPResponse:
        """
        Build MCP response from retrieval results.

        Args:
            retrieval_results: List of retrieval results
            query: Original query text

        Returns:
            MCPResponse with markdown text and structured citations
        """
        if not retrieval_results:
            return self._build_empty_response(query)

        # Generate citations
        citations = self.citation_generator.generate(retrieval_results)

        # Build markdown text with citation markers
        markdown_text = self._build_markdown(query, retrieval_results, citations)

        # Build structured citations
        structured_citations = self._build_structured_citations(citations)

        # Construct MCP response
        content = [
            {
                "type": "text",
                "text": markdown_text
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "citations://query-results",
                    "mimeType": "application/json",
                    "text": structured_citations
                }
            }
        ]

        return MCPResponse(content=content, isError=False)

    def _build_empty_response(self, query: str) -> MCPResponse:
        """Build response for empty results."""
        message = (
            f"No relevant documents found for query: \"{query}\"\n\n"
            "Suggestions:\n"
            "- Try different keywords or rephrase your query\n"
            "- Check if documents have been ingested using list_collections tool\n"
            "- Verify the collection filter if specified"
        )

        content = [{"type": "text", "text": message}]
        return MCPResponse(content=content, isError=False)

    def _build_markdown(
        self,
        query: str,
        results: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """Build markdown text with citation markers."""
        lines = [
            f"# Query Results: {query}",
            "",
            f"Found {len(results)} relevant documents:",
            ""
        ]

        for idx, citation in enumerate(citations, start=1):
            lines.append(f"## [{idx}] {citation.title}")
            lines.append(f"**Source:** {citation.source} (Page {citation.page})")
            lines.append(f"**Relevance Score:** {citation.score:.4f}")
            lines.append("")
            lines.append(citation.snippet)
            lines.append("")

        return "\n".join(lines)

    def _build_structured_citations(self, citations: List[Citation]) -> str:
        """Build structured citations as JSON string."""
        import json

        citations_data = {
            "citations": [
                {
                    "index": idx,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page": c.page,
                    "title": c.title,
                    "score": c.score
                }
                for idx, c in enumerate(citations, start=1)
            ]
        }

        return json.dumps(citations_data, indent=2)
