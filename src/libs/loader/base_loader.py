"""
Base Loader abstract interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.core.types import Document


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    Loaders are responsible for:
    1. Reading files from disk
    2. Extracting text content
    3. Extracting images (if applicable)
    4. Creating Document objects with proper metadata
    """

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Document:
        """
        Load a document from the given file path.

        Args:
            path: Path to the file to load

        Returns:
            Document object with extracted content and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If the file cannot be parsed
        """
        pass
