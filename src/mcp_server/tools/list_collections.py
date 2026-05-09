"""List collections tool implementation."""

import os
from typing import Dict, Any, List
from pathlib import Path


def get_tool_schema() -> Dict[str, Any]:
    """Get the tool schema for list_collections."""
    return {
        "name": "list_collections",
        "description": "List all available document collections in the knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }


class ListCollections:
    """List collections tool handler."""

    def __init__(self, documents_dir: str = "data/documents"):
        """
        Initialize list collections tool.

        Args:
            documents_dir: Path to documents directory
        """
        self.documents_dir = Path(documents_dir)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the list_collections tool.

        Args:
            arguments: Tool arguments (none required)

        Returns:
            MCP tool result with collection list
        """
        try:
            collections = self._get_collections()

            if not collections:
                return {
                    "content": [{
                        "type": "text",
                        "text": "No collections found. Please ingest documents first."
                    }],
                    "isError": False
                }

            # Build response text
            lines = ["# Available Collections", ""]
            for collection in collections:
                lines.append(f"- **{collection['name']}**")
                if collection.get('document_count'):
                    lines.append(f"  - Documents: {collection['document_count']}")
                lines.append("")

            return {
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(lines)
                    },
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "collections://list",
                            "mimeType": "application/json",
                            "text": self._format_collections_json(collections)
                        }
                    }
                ],
                "isError": False
            }

        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing collections: {str(e)}"
                }],
                "isError": True
            }

    def _get_collections(self) -> List[Dict[str, Any]]:
        """
        Get list of collections from documents directory.

        Returns:
            List of collection info dictionaries
        """
        collections = []

        if not self.documents_dir.exists():
            return collections

        # List subdirectories in documents directory
        for item in self.documents_dir.iterdir():
            if item.is_dir():
                collection_info = {
                    "name": item.name,
                    "path": str(item),
                    "document_count": self._count_documents(item)
                }
                collections.append(collection_info)

        return sorted(collections, key=lambda x: x['name'])

    def _count_documents(self, collection_path: Path) -> int:
        """
        Count documents in a collection directory.

        Args:
            collection_path: Path to collection directory

        Returns:
            Number of documents
        """
        count = 0
        for item in collection_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.pdf', '.md', '.txt']:
                count += 1
        return count

    def _format_collections_json(self, collections: List[Dict[str, Any]]) -> str:
        """Format collections as JSON string."""
        import json
        return json.dumps({"collections": collections}, indent=2)


# Global instance
_tool_instance: ListCollections = None


def list_collections(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    List collections tool entry point.

    Args:
        arguments: Tool arguments from MCP client

    Returns:
        MCP tool result
    """
    global _tool_instance

    if _tool_instance is None:
        _tool_instance = ListCollections()

    return _tool_instance.execute(arguments)
