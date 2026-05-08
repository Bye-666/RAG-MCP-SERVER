"""
Unit tests for PDF Loader.
"""
import os
import tempfile
import pytest
from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


class TestBaseLoader:
    """Test cases for BaseLoader abstract interface."""

    def test_base_loader_is_abstract(self):
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()


class TestPdfLoader:
    """Test cases for PdfLoader."""

    @pytest.fixture
    def simple_pdf(self, tmp_path):
        """Create a simple text-only PDF for testing."""
        # Create a simple text file and convert to PDF using a simple method
        # For testing purposes, we'll create a minimal PDF manually
        pdf_path = tmp_path / "simple.pdf"

        # Minimal PDF content (PDF 1.4 format)
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path

    def test_loader_initialization(self):
        """Test PdfLoader can be instantiated."""
        loader = PdfLoader()
        assert isinstance(loader, BaseLoader)

    def test_load_simple_pdf(self, simple_pdf):
        """Test loading a simple text-only PDF."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        # Verify Document structure
        assert isinstance(doc, Document)
        assert doc.id is not None
        assert len(doc.id) > 0
        assert doc.text is not None
        assert len(doc.text) > 0

        # Verify metadata
        assert "source_path" in doc.metadata
        assert doc.metadata["source_path"] == str(simple_pdf)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.pdf")

    def test_load_invalid_pdf(self, tmp_path):
        """Test that loading invalid PDF raises error."""
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF file")

        loader = PdfLoader()
        with pytest.raises(Exception):  # Should raise some parsing error
            loader.load(str(invalid_pdf))

    def test_document_id_consistency(self, simple_pdf):
        """Test that same file produces same document ID."""
        loader = PdfLoader()
        doc1 = loader.load(str(simple_pdf))
        doc2 = loader.load(str(simple_pdf))

        # Same file should produce same ID (based on content hash)
        assert doc1.id == doc2.id

    def test_metadata_contains_required_fields(self, simple_pdf):
        """Test that metadata contains all required fields."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        # Required field
        assert "source_path" in doc.metadata

        # Optional but expected fields
        assert "doc_type" in doc.metadata
        assert doc.metadata["doc_type"] == "pdf"

    def test_images_metadata_empty_for_text_only_pdf(self, simple_pdf):
        """Test that text-only PDF has empty or absent images metadata."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        # Images should be empty list or absent
        if "images" in doc.metadata:
            assert isinstance(doc.metadata["images"], list)
            assert len(doc.metadata["images"]) == 0

    def test_text_is_not_empty(self, simple_pdf):
        """Test that extracted text is not empty."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        assert doc.text is not None
        assert len(doc.text.strip()) > 0

    def test_text_format_is_markdown(self, simple_pdf):
        """Test that extracted text is in Markdown format."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        # Text should be string (Markdown format)
        assert isinstance(doc.text, str)

    def test_loader_handles_different_pdf_paths(self, tmp_path):
        """Test loader works with different path formats."""
        # Create PDF in subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        pdf_path = subdir / "test.pdf"

        # Minimal PDF
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 20 >>
stream
BT
/F1 12 Tf
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000062 00000 n
0000000121 00000 n
0000000217 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
286
%%EOF
"""
        pdf_path.write_bytes(pdf_content)

        loader = PdfLoader()

        # Test with string path
        doc1 = loader.load(str(pdf_path))
        assert doc1 is not None

        # Test with Path object
        doc2 = loader.load(pdf_path)
        assert doc2 is not None

    def test_document_id_is_hash_based(self, simple_pdf):
        """Test that document ID is based on content hash."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf))

        # ID should be a hash (hex string)
        assert len(doc.id) >= 8  # At least 8 characters
        assert all(c in "0123456789abcdef" for c in doc.id.lower())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
