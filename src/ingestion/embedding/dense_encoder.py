"""Dense vector encoder for text chunks

This module provides the DenseEncoder class that converts text chunks into
dense embedding vectors using the configured embedding provider.
"""

from typing import List, Optional
from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.trace import TraceContext


class DenseEncoder:
    """Encodes text chunks into dense embedding vectors

    This encoder takes a list of Chunk objects and produces ChunkRecord objects
    with dense_vector populated. It delegates the actual embedding computation
    to a BaseEmbedding implementation (e.g., OpenAI, Azure, Ollama).

    Attributes:
        embedding_model: The embedding provider instance
    """

    def __init__(self, embedding_model: BaseEmbedding):
        """Initialize the dense encoder

        Args:
            embedding_model: An instance of BaseEmbedding for computing embeddings
        """
        if not isinstance(embedding_model, BaseEmbedding):
            raise TypeError("embedding_model must be an instance of BaseEmbedding")
        self.embedding_model = embedding_model

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """Encode chunks into dense vectors

        Args:
            chunks: List of Chunk objects to encode
            trace: Optional trace context for observability

        Returns:
            List of ChunkRecord objects with dense_vector populated

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        # Extract text from all chunks
        texts = [chunk.text for chunk in chunks]

        # Call embedding model to get vectors
        vectors = self.embedding_model.embed(texts, trace=trace)

        # Validate output dimensions
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors "
                f"but expected {len(chunks)}"
            )

        # Create ChunkRecord objects with dense vectors
        records = []
        for chunk, vector in zip(chunks, vectors):
            record = ChunkRecord.from_chunk(
                chunk=chunk,
                dense_vector=vector
            )
            records.append(record)

        return records
