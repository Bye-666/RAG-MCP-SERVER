"""Tests for get_document_summary tool."""

import pytest
from unittest.mock import Mock
from src.mcp_server.tools.get_document_summary import (
    GetDocumentSummary,
    get_tool_schema,
    get_document_summary
)


class TestGetDocumentSummary:
    """Test GetDocumentSummary functionality."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Sample document content",
                "metadata": {
                    "title": "Test Document",
                    "summary": "This is a test document about machine learning.",
                    "tags": ["machine-learning", "ai", "test"],
                    "source_path": "/path/to/test.pdf",
                    "doc_type": "pdf",
                    "collection": "test_collection"
                }
            }
        ]
        return store

    @pytest.fixture
    def tool(self, mock_vector_store):
        """Create GetDocumentSummary instance with mock store."""
        return GetDocumentSummary(vector_store=mock_vector_store)

    def test_get_tool_schema(self):
        """Test tool schema structure."""
        schema = get_tool_schema()

        assert schema["name"] == "get_document_summary"
        assert "description" in schema
        assert schema["inputSchema"]["type"] == "object"
        assert "doc_id" in schema["inputSchema"]["properties"]
        assert schema["inputSchema"]["required"] == ["doc_id"]

    def test_get_document_summary_success(self, tool):
        """Test successful document summary retrieval."""
        result = tool.execute({"doc_id": "abc123"})

        assert result["isError"] is False
        assert len(result["content"]) == 2

        # Check text content
        text_content = result["content"][0]
        assert text_content["type"] == "text"
        assert "Test Document" in text_content["text"]
        assert "machine learning" in text_content["text"]

        # Check resource content
        resource_content = result["content"][1]
        assert resource_content["type"] == "resource"
        assert "document://abc123/summary" in resource_content["resource"]["uri"]
        assert resource_content["resource"]["mimeType"] == "application/json"

    def test_missing_doc_id(self, tool):
        """Test error when doc_id is missing."""
        result = tool.execute({})

        assert result["isError"] is True
        assert "Missing required parameter" in result["content"][0]["text"]

    def test_document_not_found(self, tool, mock_vector_store):
        """Test error when document is not found."""
        mock_vector_store.get_by_ids.return_value = []

        result = tool.execute({"doc_id": "nonexistent"})

        assert result["isError"] is True
        assert "Document not found" in result["content"][0]["text"]
        assert "nonexistent" in result["content"][0]["text"]

    def test_document_info_extraction(self, tool):
        """Test document info extraction from vector store."""
        doc_info = tool._get_document_info("abc123")

        assert doc_info is not None
        assert doc_info["doc_id"] == "abc123"
        assert doc_info["title"] == "Test Document"
        assert doc_info["summary"] == "This is a test document about machine learning."
        assert doc_info["tags"] == ["machine-learning", "ai", "test"]
        assert doc_info["source_path"] == "/path/to/test.pdf"
        assert doc_info["doc_type"] == "pdf"
        assert doc_info["collection"] == "test_collection"

    def test_missing_metadata_fields(self, tool, mock_vector_store):
        """Test handling of missing metadata fields."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Content",
                "metadata": {}
            }
        ]

        doc_info = tool._get_document_info("abc123")

        assert doc_info is not None
        assert doc_info["title"] == "Untitled"
        assert doc_info["summary"] == "No summary available"
        assert doc_info["tags"] == []
        assert doc_info["source_path"] == "Unknown"
        assert doc_info["doc_type"] == "unknown"

    def test_partial_metadata(self, tool, mock_vector_store):
        """Test handling of partial metadata."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Content",
                "metadata": {
                    "title": "Partial Doc",
                    "source_path": "/path/to/doc.pdf"
                }
            }
        ]

        doc_info = tool._get_document_info("abc123")

        assert doc_info["title"] == "Partial Doc"
        assert doc_info["source_path"] == "/path/to/doc.pdf"
        assert doc_info["summary"] == "No summary available"
        assert doc_info["tags"] == []

    def test_vector_store_exception(self, tool, mock_vector_store):
        """Test handling of vector store exceptions."""
        mock_vector_store.get_by_ids.side_effect = Exception("Database error")

        doc_info = tool._get_document_info("abc123")

        assert doc_info is None

    def test_no_vector_store(self):
        """Test behavior when no vector store is provided."""
        tool = GetDocumentSummary(vector_store=None)
        doc_info = tool._get_document_info("abc123")

        assert doc_info is None

    def test_execute_with_vector_store_error(self, tool, mock_vector_store):
        """Test execute method with vector store error."""
        mock_vector_store.get_by_ids.side_effect = Exception("Connection failed")

        result = tool.execute({"doc_id": "abc123"})

        assert result["isError"] is True
        assert "Document not found" in result["content"][0]["text"]

    def test_json_format(self, tool):
        """Test JSON formatting in resource."""
        import json

        result = tool.execute({"doc_id": "abc123"})
        resource_text = result["content"][1]["resource"]["text"]
        data = json.loads(resource_text)

        assert "doc_id" in data
        assert "title" in data
        assert "summary" in data
        assert "tags" in data
        assert data["doc_id"] == "abc123"

    def test_markdown_formatting(self, tool):
        """Test markdown formatting in text response."""
        result = tool.execute({"doc_id": "abc123"})
        text = result["content"][0]["text"]

        # Check markdown elements
        assert text.startswith("# Document Summary:")
        assert "**Document ID:**" in text
        assert "**Source:**" in text
        assert "## Summary" in text
        assert "## Tags" in text

    def test_tags_display(self, tool):
        """Test tags are displayed correctly."""
        result = tool.execute({"doc_id": "abc123"})
        text = result["content"][0]["text"]

        assert "`machine-learning`" in text
        assert "`ai`" in text
        assert "`test`" in text

    def test_empty_tags_display(self, tool, mock_vector_store):
        """Test display when tags are empty."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Content",
                "metadata": {
                    "title": "No Tags Doc",
                    "tags": []
                }
            }
        ]

        result = tool.execute({"doc_id": "abc123"})
        text = result["content"][0]["text"]

        assert "No tags" in text

    def test_global_function_entry_point(self, mock_vector_store):
        """Test global get_document_summary function."""
        result = get_document_summary({"doc_id": "test123"})

        # Should work even without vector store (returns error)
        assert "isError" in result

    def test_empty_doc_id(self, tool):
        """Test with empty doc_id string."""
        result = tool.execute({"doc_id": ""})

        assert result["isError"] is True
        assert "Missing required parameter" in result["content"][0]["text"]

    def test_special_characters_in_metadata(self, tool, mock_vector_store):
        """Test handling special characters in metadata."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Content",
                "metadata": {
                    "title": "Document with \"quotes\" and <tags>",
                    "summary": "Summary with special chars: & < > \"",
                    "tags": ["tag-with-dash", "tag_with_underscore"],
                    "source_path": "/path/with spaces/doc.pdf"
                }
            }
        ]

        result = tool.execute({"doc_id": "abc123"})

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "quotes" in text
        assert "special chars" in text

    def test_unicode_in_metadata(self, tool, mock_vector_store):
        """Test handling unicode characters in metadata."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "Content",
                "metadata": {
                    "title": "中文标题 Unicode Title",
                    "summary": "摘要内容 Summary content 日本語",
                    "tags": ["中文", "日本語", "한국어"],
                    "source_path": "/path/文档.pdf"
                }
            }
        ]

        result = tool.execute({"doc_id": "abc123"})

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "中文标题" in text
        assert "摘要内容" in text

    def test_long_doc_id(self, tool, mock_vector_store):
        """Test with very long doc_id."""
        mock_vector_store.get_by_ids.return_value = []

        long_id = "a" * 100
        result = tool.execute({"doc_id": long_id})

        # Should handle gracefully (not found)
        assert result["isError"] is True

    def test_doc_id_with_special_chars(self, tool, mock_vector_store):
        """Test doc_id with special characters."""
        special_id = "doc-123_abc.xyz"
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": f"{special_id}_0000_xyz",
                "text": "Content",
                "metadata": {"title": "Special ID Doc"}
            }
        ]

        result = tool.execute({"doc_id": special_id})

        assert result["isError"] is False

    def test_suggestion_in_error_response(self, tool, mock_vector_store):
        """Test that error response includes helpful suggestion."""
        mock_vector_store.get_by_ids.return_value = []

        result = tool.execute({"doc_id": "missing123"})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "Suggestion:" in text
        assert "list_collections" in text

    def test_multiple_chunks_returns_first(self, tool, mock_vector_store):
        """Test that when multiple chunks exist, first one is used."""
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "abc123_0000_xyz",
                "text": "First chunk",
                "metadata": {
                    "title": "First Title",
                    "summary": "First summary"
                }
            },
            {
                "id": "abc123_0001_xyz",
                "text": "Second chunk",
                "metadata": {
                    "title": "Second Title",
                    "summary": "Second summary"
                }
            }
        ]

        doc_info = tool._get_document_info("abc123")

        # Should use first chunk
        assert doc_info["title"] == "First Title"
        assert doc_info["summary"] == "First summary"
