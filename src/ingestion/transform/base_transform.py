"""
BaseTransform: Abstract base class for chunk transformation operations.

All transform operations in the ingestion pipeline should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace import TraceContext


class BaseTransform(ABC):
    """
    Abstract base class for chunk transformations.

    Transform operations modify chunks during the ingestion pipeline,
    such as:
    - ChunkRefiner: Clean and refine chunk text
    - MetadataEnricher: Add metadata like title, summary, tags
    - ImageCaptioner: Generate captions for images

    All transforms should:
    1. Accept a list of chunks and optional trace context
    2. Return a modified list of chunks
    3. Handle errors gracefully without blocking the pipeline
    4. Record their operations in the trace context
    """

    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        Transform a list of chunks.

        Args:
            chunks: List of Chunk objects to transform
            trace: Optional trace context for recording operations

        Returns:
            List of transformed Chunk objects

        Raises:
            Should not raise exceptions that block the pipeline.
            Errors should be logged and handled gracefully.
        """
        pass
