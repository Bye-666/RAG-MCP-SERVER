"""
DocumentChunker: Adapter layer between libs.splitter and core.types.

Converts Document objects to List[Chunk] with business logic:
- Chunk ID generation
- Metadata inheritance
- Image reference distribution
- Source reference tracking
"""
import hashlib
import re
from typing import List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """
    Adapter that converts Document objects to Chunk objects using libs.splitter.

    Responsibilities:
    1. Generate unique and deterministic Chunk IDs
    2. Inherit metadata from parent Document
    3. Add chunk_index for ordering
    4. Establish source_ref to parent Document
    5. Distribute image references to chunks that reference them
    6. Convert List[str] from splitter to List[Chunk]
    """

    def __init__(self, settings):
        """
        Initialize DocumentChunker with settings.

        Args:
            settings: Settings object with splitter configuration
        """
        self.settings = settings
        self.splitter = SplitterFactory.create(settings.splitter)

    def split_document(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunks.

        Args:
            document: Document object to split

        Returns:
            List of Chunk objects with metadata and references
        """
        # Use splitter to get text chunks
        text_chunks = self.splitter.split_text(document.text)

        # Convert to Chunk objects with business logic
        chunks = []
        for idx, text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(document.id, idx, text)
            metadata = self._inherit_metadata(document, idx, text)

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                source_ref=document.id
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        Generate unique and deterministic chunk ID.

        Format: {doc_id}_{index:04d}_{hash_8chars}

        Args:
            doc_id: Parent document ID
            index: Chunk index in document
            text: Chunk text content

        Returns:
            Chunk ID string
        """
        # Compute hash of text for uniqueness
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict:
        """
        Inherit metadata from document and add chunk-specific fields.

        Also distributes image references to chunks that contain image placeholders.

        Args:
            document: Parent document
            chunk_index: Index of this chunk
            chunk_text: Text content of this chunk

        Returns:
            Metadata dictionary for chunk
        """
        # Copy all document metadata
        metadata = document.metadata.copy()

        # Add chunk_index
        metadata["chunk_index"] = chunk_index

        # Distribute image references
        self._distribute_images(metadata, chunk_text)

        return metadata

    def _distribute_images(self, metadata: dict, chunk_text: str) -> None:
        """
        Distribute image references to chunk metadata based on placeholders.

        Scans chunk_text for [IMAGE: {id}] placeholders and extracts matching
        images from document-level metadata["images"].

        Modifies metadata in-place:
        - Adds metadata["images"]: list of ImageRef dicts for this chunk
        - Adds metadata["image_refs"]: list of image_id strings
        - Removes document-level "images" if no placeholders found

        Args:
            metadata: Chunk metadata dict (will be modified in-place)
            chunk_text: Text content to scan for placeholders
        """
        # Extract image IDs from placeholders in chunk text
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, chunk_text)
        image_ids = [m.strip() for m in matches]

        if not image_ids:
            # No image placeholders in this chunk, remove images field
            metadata.pop("images", None)
            return

        # Get document-level images
        doc_images = metadata.get("images", [])
        if not doc_images:
            # Document has no images metadata
            return

        # Filter images that are referenced in this chunk
        chunk_images = []
        for img in doc_images:
            if img.get("image_id") in image_ids:
                chunk_images.append(img)

        # Update metadata with chunk-specific images
        if chunk_images:
            metadata["images"] = chunk_images
            metadata["image_refs"] = image_ids
        else:
            # No matching images found, remove field
            metadata.pop("images", None)
