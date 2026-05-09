"""Multimodal content assembler for MCP responses.

Handles assembly of text and image content from retrieval results.
"""

import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes


class MultimodalAssembler:
    """Assembles multimodal content (text + images) for MCP responses."""

    def __init__(self, images_base_dir: str = "data/images"):
        """
        Initialize multimodal assembler.

        Args:
            images_base_dir: Base directory for image storage
        """
        self.images_base_dir = Path(images_base_dir)

    def assemble(self, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assemble multimodal content from retrieval results.

        Args:
            retrieval_results: List of retrieval results with text and metadata

        Returns:
            List of MCP content items (text and image)
        """
        content_items = []

        for result in retrieval_results:
            # Add text content
            text = result.get("text", "")
            if text:
                content_items.append({
                    "type": "text",
                    "text": text
                })

            # Check for images in metadata
            metadata = result.get("metadata", {})
            images = metadata.get("images", [])

            # Add image content
            for image_info in images:
                image_content = self._load_image(image_info)
                if image_content:
                    content_items.append(image_content)

        return content_items

    def _load_image(self, image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load image from file and convert to base64.

        Args:
            image_info: Image metadata with path, id, etc.

        Returns:
            MCP image content dict, or None if loading fails
        """
        try:
            image_path = image_info.get("path")
            if not image_path:
                return None

            # Resolve path (can be relative or absolute)
            if not Path(image_path).is_absolute():
                image_path = self.images_base_dir / image_path
            else:
                image_path = Path(image_path)

            # Check if file exists
            if not image_path.exists():
                return None

            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Convert to base64
            base64_data = base64.b64encode(image_data).decode("utf-8")

            # Determine MIME type
            mime_type = self._get_mime_type(image_path)

            return {
                "type": "image",
                "data": base64_data,
                "mimeType": mime_type
            }

        except Exception:
            # Silently skip images that fail to load
            return None

    def _get_mime_type(self, file_path: Path) -> str:
        """
        Get MIME type for image file.

        Args:
            file_path: Path to image file

        Returns:
            MIME type string (e.g., "image/png")
        """
        # Try to guess from extension
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type and mime_type.startswith("image/"):
            return mime_type

        # Default to common image types based on extension
        ext = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml"
        }

        return mime_map.get(ext, "image/png")  # Default to PNG
