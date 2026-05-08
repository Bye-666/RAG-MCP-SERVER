"""
PDF Loader implementation using markitdown.
"""
import hashlib
from pathlib import Path
from typing import Union

from markitdown import MarkItDown

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    """
    PDF document loader using markitdown library.

    Extracts text content from PDF files and converts to Markdown format.
    """

    def __init__(self):
        """Initialize the PDF loader."""
        self._converter = MarkItDown()

    def load(self, path: Union[str, Path]) -> Document:
        """
        Load a PDF document from the given file path.

        Args:
            path: Path to the PDF file

        Returns:
            Document object with extracted content and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid PDF
            Exception: If the PDF cannot be parsed
        """
        # Convert to Path object
        file_path = Path(path)

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate PDF format
        if not self._is_valid_pdf(file_path):
            raise ValueError(f"File is not a valid PDF: {file_path}")

        # Compute document ID based on file content hash
        doc_id = self._compute_file_hash(file_path)

        # Extract text using markitdown
        try:
            result = self._converter.convert(str(file_path))
            text_content = result.text_content
        except Exception as e:
            raise Exception(f"Failed to parse PDF: {e}") from e

        # Build metadata
        metadata = {
            "source_path": str(file_path),
            "doc_type": "pdf",
        }

        # Create Document
        doc = Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

        return doc

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _is_valid_pdf(self, file_path: Path) -> bool:
        """
        Check if file is a valid PDF by checking the header.

        Args:
            file_path: Path to the file

        Returns:
            True if file starts with PDF header, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                return header == b"%PDF-"
        except Exception:
            return False
