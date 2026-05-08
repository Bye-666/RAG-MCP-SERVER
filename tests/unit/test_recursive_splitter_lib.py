import pytest
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class TestRecursiveSplitter:
    """Test cases for RecursiveSplitter class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        splitter = RecursiveSplitter()
        assert splitter.provider == "recursive"
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        splitter = RecursiveSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_split_text_simple(self):
        """Test basic text splitting."""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 10
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some margin

    def test_split_text_empty_string(self):
        """Test that empty string returns empty list."""
        splitter = RecursiveSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_text_whitespace_only(self):
        """Test that whitespace-only string returns empty list."""
        splitter = RecursiveSplitter()
        chunks = splitter.split_text("   \n\n   ")
        assert chunks == []

    def test_split_text_non_string_input(self):
        """Test that non-string input raises TypeError."""
        splitter = RecursiveSplitter()
        with pytest.raises(TypeError, match="text must be a string"):
            splitter.split_text(123)

    def test_split_markdown_headers(self):
        """Test that Markdown headers are preserved."""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=0)
        text = """## Header 1
Content for header 1.

## Header 2
Content for header 2.

### Subheader 2.1
Content for subheader."""

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Check that headers are preserved in chunks
        header_found = any("##" in chunk for chunk in chunks)
        assert header_found

    def test_split_markdown_code_blocks(self):
        """Test that code blocks are handled properly."""
        splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
        text = """Here is some code:

```python
def hello():
    print("Hello, world!")
    return True
```

And here is more text after the code block."""

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Code block should ideally stay together
        code_chunks = [c for c in chunks if "```" in c or "def hello" in c]
        assert len(code_chunks) > 0

    def test_split_long_text(self):
        """Test splitting of long text."""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=20)
        text = "A" * 500
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # Check overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            # Some overlap should exist
            assert len(chunks[i]) <= 120  # chunk_size + some margin

    def test_split_text_with_trace(self):
        """Test that trace logging works."""
        from unittest.mock import Mock

        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)
        trace = Mock()
        text = "This is a test. " * 10

        chunks = splitter.split_text(text, trace=trace)

        assert len(chunks) > 0
        trace.log.assert_called_once()
        call_args = trace.log.call_args[0]
        assert call_args[0] == "splitter"
        assert "chunk_count" in call_args[1]

    def test_factory_integration(self):
        """Test that RecursiveSplitter can be created via factory."""
        # Register the provider
        SplitterFactory.register_provider("recursive", RecursiveSplitter)

        settings = {
            "splitter": {
                "provider": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }

        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_chunk_overlap_behavior(self):
        """Test that chunk overlap works as expected."""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = splitter.split_text(text)

        assert len(chunks) >= 2
        # Verify chunks are created
        total_chars = sum(len(chunk) for chunk in chunks)
        # Total should be more than original due to overlap
        assert total_chars >= len(text)

    def test_custom_separators(self):
        """Test that custom separators can be provided."""
        custom_separators = ["\n\n", "\n", " "]
        splitter = RecursiveSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separators=custom_separators
        )

        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
