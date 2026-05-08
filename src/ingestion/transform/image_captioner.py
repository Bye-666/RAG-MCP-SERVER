"""
ImageCaptioner: Transform that generates captions for images in chunks.

Provides two modes:
1. Enabled mode: Uses Vision LLM to generate captions for images
2. Disabled/fallback mode: Marks chunks with unprocessed images

Falls back gracefully when Vision LLM is unavailable or fails.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ImageCaptioner(BaseTransform):
    """
    Generates captions for images referenced in chunks.

    Two-stage processing:
    1. Vision LLM mode (if enabled and available)
    2. Fallback mode (marks unprocessed images)

    Graceful degradation: Vision LLM failures don't block ingestion.
    """

    def __init__(
        self,
        settings,
        vision_llm=None,
        prompt_path: Optional[str] = None
    ):
        """
        Initialize ImageCaptioner.

        Args:
            settings: Settings object with vision_llm configuration
            vision_llm: Optional Vision LLM instance (if None, will create from settings)
            prompt_path: Optional path to prompt template file
        """
        self.settings = settings
        self.use_vision = getattr(settings.ingestion.image_captioner, 'use_vision', False)

        # Initialize Vision LLM if enabled
        self.vision_llm = None
        if self.use_vision:
            if vision_llm is not None:
                self.vision_llm = vision_llm
            else:
                try:
                    self.vision_llm = LLMFactory.create_vision_llm(settings)
                except Exception as e:
                    logger.warning(f"Failed to initialize Vision LLM for image captioning: {e}")
                    self.use_vision = False

        # Load prompt template
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        Load prompt template from file.

        Args:
            prompt_path: Optional custom prompt path

        Returns:
            Prompt template string
        """
        if prompt_path is None:
            prompt_path = "config/prompts/image_captioning.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_path}: {e}")

        # Fallback prompt
        return "Describe this image in detail. Focus on the main content, objects, text, and any important visual elements."

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        Transform chunks by generating captions for images.

        Args:
            chunks: List of chunks to process
            trace: Optional trace context

        Returns:
            List of chunks with image captions (if applicable)
        """
        stage = None
        if trace:
            stage = trace.record_stage("image_captioner", {"use_vision": self.use_vision})

        processed_chunks = []
        images_processed = 0
        images_failed = 0

        for chunk in chunks:
            try:
                processed_chunk = self._process_chunk(chunk, trace)
                processed_chunks.append(processed_chunk)

                # Track statistics
                if "image_captions" in processed_chunk.metadata:
                    images_processed += len(processed_chunk.metadata["image_captions"])
                if processed_chunk.metadata.get("has_unprocessed_images"):
                    images_failed += 1

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")
                # On error, keep original chunk but mark as unprocessed
                error_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={**chunk.metadata, "image_processing_error": str(e)},
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref
                )
                if "image_refs" in chunk.metadata:
                    error_chunk.metadata["has_unprocessed_images"] = True
                processed_chunks.append(error_chunk)

        if trace and stage:
            trace.finish_stage(stage, {
                "chunks_processed": len(processed_chunks),
                "images_captioned": images_processed,
                "images_failed": images_failed
            })

        return processed_chunks

    def _process_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """
        Process a single chunk to generate image captions.

        Args:
            chunk: Chunk to process
            trace: Optional trace context

        Returns:
            Processed chunk with captions or fallback markers
        """
        # Check if chunk has image references
        image_refs = chunk.metadata.get("image_refs", [])
        if not image_refs:
            # No images, return as-is
            return chunk

        # Try to generate captions if Vision LLM is available
        captions = {}
        unprocessed = []

        if self.use_vision and self.vision_llm:
            for image_ref in image_refs:
                caption = self._generate_caption(image_ref, chunk, trace)
                if caption:
                    captions[image_ref] = caption
                else:
                    unprocessed.append(image_ref)
        else:
            # Vision disabled, mark all as unprocessed
            unprocessed = image_refs

        # Build updated metadata
        updated_metadata = {**chunk.metadata}

        if captions:
            updated_metadata["image_captions"] = captions
            updated_metadata["captioned_by"] = "vision_llm"

        if unprocessed:
            updated_metadata["has_unprocessed_images"] = True
            updated_metadata["unprocessed_image_refs"] = unprocessed

        # Create updated chunk
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=updated_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )

    def _generate_caption(
        self,
        image_ref: str,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """
        Generate caption for a single image.

        Args:
            image_ref: Image reference ID
            chunk: Parent chunk (for context)
            trace: Optional trace context

        Returns:
            Caption string, or None on failure
        """
        if not self.vision_llm:
            return None

        try:
            # Get image path from metadata
            image_path = self._resolve_image_path(image_ref, chunk)
            if not image_path:
                logger.warning(f"Could not resolve image path for {image_ref}")
                return None

            # Check if image file exists
            if not Path(image_path).exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            # Generate caption using Vision LLM
            response = self.vision_llm.chat_with_image(
                text=self.prompt_template,
                image=image_path,
                trace=trace
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate caption for {image_ref}: {e}")
            return None

    def _resolve_image_path(self, image_ref: str, chunk: Chunk) -> Optional[str]:
        """
        Resolve image reference to file path.

        Args:
            image_ref: Image reference ID
            chunk: Parent chunk

        Returns:
            Image file path, or None if not found
        """
        # Check if chunk metadata has images list with path info
        images = chunk.metadata.get("images", [])

        for img in images:
            if isinstance(img, dict) and img.get("image_id") == image_ref:
                return img.get("path")

        # Fallback: construct path from image_ref
        # Assuming format: data/images/{doc_hash}/{image_id}.{ext}
        # This is a heuristic and may need adjustment based on actual storage
        logger.debug(f"Could not find image path in metadata for {image_ref}, using fallback")
        return None
