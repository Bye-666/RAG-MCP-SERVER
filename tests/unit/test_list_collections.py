"""Tests for list_collections tool."""

import pytest
from pathlib import Path
from src.mcp_server.tools.list_collections import (
    ListCollections,
    get_tool_schema,
    list_collections
)


class TestListCollections:
    """Test ListCollections functionality."""

    @pytest.fixture
    def temp_docs_dir(self, tmp_path):
        """Create temporary documents directory structure."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create collection1 with 2 PDFs
        col1 = docs_dir / "collection1"
        col1.mkdir()
        (col1 / "doc1.pdf").write_text("PDF content")
        (col1 / "doc2.pdf").write_text("PDF content")

        # Create collection2 with mixed files
        col2 = docs_dir / "collection2"
        col2.mkdir()
        (col2 / "doc1.md").write_text("Markdown")
        (col2 / "doc2.txt").write_text("Text")
        (col2 / "ignored.json").write_text("{}")  # Should be ignored

        # Create empty collection3
        col3 = docs_dir / "collection3"
        col3.mkdir()

        return docs_dir

    @pytest.fixture
    def tool(self, temp_docs_dir):
        """Create ListCollections instance with temp directory."""
        return ListCollections(documents_dir=str(temp_docs_dir))

    def test_get_tool_schema(self):
        """Test tool schema structure."""
        schema = get_tool_schema()

        assert schema["name"] == "list_collections"
        assert "description" in schema
        assert schema["inputSchema"]["type"] == "object"
        assert schema["inputSchema"]["properties"] == {}
        assert schema["inputSchema"]["required"] == []

    def test_list_collections_success(self, tool):
        """Test listing collections successfully."""
        result = tool.execute({})

        assert result["isError"] is False
        assert len(result["content"]) == 2

        # Check text content
        text_content = result["content"][0]
        assert text_content["type"] == "text"
        assert "Available Collections" in text_content["text"]
        assert "collection1" in text_content["text"]
        assert "collection2" in text_content["text"]
        assert "collection3" in text_content["text"]

        # Check resource content
        resource_content = result["content"][1]
        assert resource_content["type"] == "resource"
        assert resource_content["resource"]["uri"] == "collections://list"
        assert resource_content["resource"]["mimeType"] == "application/json"

    def test_document_count_accuracy(self, tool):
        """Test document counting is accurate."""
        collections = tool._get_collections()

        col1 = next(c for c in collections if c["name"] == "collection1")
        col2 = next(c for c in collections if c["name"] == "collection2")
        col3 = next(c for c in collections if c["name"] == "collection3")

        assert col1["document_count"] == 2  # 2 PDFs
        assert col2["document_count"] == 2  # 1 MD + 1 TXT (JSON ignored)
        assert col3["document_count"] == 0  # Empty

    def test_collections_sorted_by_name(self, tool):
        """Test collections are sorted alphabetically."""
        collections = tool._get_collections()
        names = [c["name"] for c in collections]

        assert names == sorted(names)

    def test_empty_documents_directory(self, tmp_path):
        """Test with empty documents directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = ListCollections(documents_dir=str(empty_dir))
        result = tool.execute({})

        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert "No collections found" in result["content"][0]["text"]

    def test_missing_documents_directory(self, tmp_path):
        """Test with non-existent documents directory."""
        missing_dir = tmp_path / "missing"

        tool = ListCollections(documents_dir=str(missing_dir))
        result = tool.execute({})

        assert result["isError"] is False
        assert "No collections found" in result["content"][0]["text"]

    def test_only_files_no_directories(self, tmp_path):
        """Test directory with only files, no subdirectories."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "file1.pdf").write_text("content")
        (docs_dir / "file2.txt").write_text("content")

        tool = ListCollections(documents_dir=str(docs_dir))
        result = tool.execute({})

        assert result["isError"] is False
        assert "No collections found" in result["content"][0]["text"]

    def test_supported_file_extensions(self, tmp_path):
        """Test only supported file extensions are counted."""
        docs_dir = tmp_path / "documents"
        col = docs_dir / "test_col"
        col.mkdir(parents=True)

        # Supported extensions
        (col / "doc1.pdf").write_text("content")
        (col / "doc2.PDF").write_text("content")  # Case insensitive
        (col / "doc3.md").write_text("content")
        (col / "doc4.txt").write_text("content")

        # Unsupported extensions
        (col / "doc.docx").write_text("content")
        (col / "doc.json").write_text("content")
        (col / "doc.py").write_text("content")

        tool = ListCollections(documents_dir=str(docs_dir))
        collections = tool._get_collections()

        assert len(collections) == 1
        assert collections[0]["document_count"] == 4  # Only PDF, MD, TXT

    def test_json_format(self, tool):
        """Test JSON resource format."""
        import json

        result = tool.execute({})
        resource_text = result["content"][1]["resource"]["text"]
        data = json.loads(resource_text)

        assert "collections" in data
        assert isinstance(data["collections"], list)
        assert len(data["collections"]) == 3

        for collection in data["collections"]:
            assert "name" in collection
            assert "path" in collection
            assert "document_count" in collection

    def test_collection_path_included(self, tool):
        """Test collection path is included in results."""
        collections = tool._get_collections()

        for collection in collections:
            assert "path" in collection
            assert Path(collection["path"]).exists()
            assert Path(collection["path"]).is_dir()

    def test_global_function_entry_point(self, temp_docs_dir):
        """Test global list_collections function."""
        # First call creates instance
        result1 = list_collections({})
        assert result1["isError"] is False

        # Second call reuses instance
        result2 = list_collections({})
        assert result2["isError"] is False

    def test_error_handling(self, tmp_path):
        """Test error handling for invalid directory."""
        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")

        tool = ListCollections(documents_dir=str(file_path))
        result = tool.execute({})

        # Should handle gracefully
        assert "isError" in result

    def test_special_characters_in_collection_name(self, tmp_path):
        """Test collections with special characters in names."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create collection with special chars
        col = docs_dir / "collection-with_special.chars"
        col.mkdir()
        (col / "doc.pdf").write_text("content")

        tool = ListCollections(documents_dir=str(docs_dir))
        collections = tool._get_collections()

        assert len(collections) == 1
        assert collections[0]["name"] == "collection-with_special.chars"

    def test_nested_subdirectories_not_counted(self, tmp_path):
        """Test nested subdirectories are not counted as collections."""
        docs_dir = tmp_path / "documents"
        col = docs_dir / "collection1"
        nested = col / "nested"
        nested.mkdir(parents=True)

        (col / "doc.pdf").write_text("content")
        (nested / "nested_doc.pdf").write_text("content")

        tool = ListCollections(documents_dir=str(docs_dir))
        collections = tool._get_collections()

        # Only top-level collection should be listed
        assert len(collections) == 1
        assert collections[0]["name"] == "collection1"
        # Document count should only include files in collection root
        assert collections[0]["document_count"] == 1

    def test_markdown_formatting(self, tool):
        """Test markdown formatting in text response."""
        result = tool.execute({})
        text = result["content"][0]["text"]

        # Check markdown elements
        assert text.startswith("# Available Collections")
        assert "- **collection1**" in text
        assert "Documents:" in text

    def test_empty_arguments(self, tool):
        """Test with empty arguments dictionary."""
        result = tool.execute({})
        assert result["isError"] is False

    def test_arguments_ignored(self, tool):
        """Test that any arguments are ignored gracefully."""
        result = tool.execute({"unexpected": "argument"})
        assert result["isError"] is False
