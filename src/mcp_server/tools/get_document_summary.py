"""Get document summary tool implementation."""

from typing import Dict, Any, Optional
from pathlib import Path


def get_tool_schema() -> Dict[str, Any]:
    """Get the tool schema for get_document_summary."""
    return {
        "name": "get_document_summary",
        "description": "Get summary information for a document by its ID, including title, summary, and tags",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID (SHA256 hash of document content)"
                }
            },
            "required": ["doc_id"]
        }
    }


class GetDocumentSummary:
    """Get document summary tool handler."""

    def __init__(self, vector_store=None):
        """
        Initialize get document summary tool.

        Args:
            vector_store: Vector store instance for querying document metadata
        """
        self.vector_store = vector_store

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the get_document_summary tool.

        Args:
            arguments: Tool arguments with doc_id

        Returns:
            MCP tool result with document summary
        """
        try:
            doc_id = arguments.get("doc_id")
            if not doc_id:
                return self._build_error_response("Missing required parameter: doc_id")

            # Get document metadata
            doc_info = self._get_document_info(doc_id)

            if not doc_info:
                return self._build_error_response(
                    f"Document not found: {doc_id}",
                    suggestion="Check if the document has been ingested using list_collections tool"
                )

            # Build response
            return self._build_success_response(doc_info)

        except Exception as e:
            return self._build_error_response(f"Error retrieving document summary: {str(e)}")

    def _get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document information from vector store.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with document info, or None if not found
        """
        if not self.vector_store:
            # Fallback: return mock data for testing
            return None

        try:
            # Query chunks belonging to this document
            # Chunk IDs follow format: {doc_id}_{index:04d}_{hash}
            # We'll get the first chunk which typically contains document-level metadata
            chunk_id_prefix = f"{doc_id}_0000"

            # Try to get chunks by ID pattern
            # Note: This is a simplified implementation
            # In production, you might want to add a metadata filter for doc_id
            results = self.vector_store.get_by_ids([chunk_id_prefix])

            if not results:
                return None

            # Extract metadata from first chunk
            first_chunk = results[0]
            metadata = first_chunk.get("metadata", {})

            return {
                "doc_id": doc_id,
                "title": metadata.get("title", "Untitled"),
                "summary": metadata.get("summary", "No summary available"),
                "tags": metadata.get("tags", []),
                "source_path": metadata.get("source_path", "Unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "collection": metadata.get("collection", "default")
            }

        except Exception:
            return None

    def _build_success_response(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build success response with document info."""
        # Build markdown text
        lines = [
            f"# Document Summary: {doc_info['title']}",
            "",
            f"**Document ID:** `{doc_info['doc_id']}`",
            f"**Source:** {doc_info['source_path']}",
            f"**Type:** {doc_info['doc_type']}",
            f"**Collection:** {doc_info['collection']}",
            "",
            "## Summary",
            doc_info['summary'],
            "",
            "## Tags",
            ", ".join(f"`{tag}`" for tag in doc_info['tags']) if doc_info['tags'] else "No tags",
            ""
        ]

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(lines)
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": f"document://{doc_info['doc_id']}/summary",
                        "mimeType": "application/json",
                        "text": self._format_json(doc_info)
                    }
                }
            ],
            "isError": False
        }

    def _build_error_response(self, message: str, suggestion: str = "") -> Dict[str, Any]:
        """Build error response."""
        text = f"Error: {message}"
        if suggestion:
            text += f"\n\nSuggestion: {suggestion}"

        return {
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": True
        }

    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format data as JSON string."""
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)


# Global instance
_tool_instance: GetDocumentSummary = None


def get_document_summary(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get document summary tool entry point.

    Args:
        arguments: Tool arguments from MCP client

    Returns:
        MCP tool result
    """
    global _tool_instance

    if _tool_instance is None:
        _tool_instance = GetDocumentSummary()

    return _tool_instance.execute(arguments)
