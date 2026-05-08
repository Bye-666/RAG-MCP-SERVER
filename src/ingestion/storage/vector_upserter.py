"""Vector upserter for persisting chunk embeddings to vector store

This module provides the VectorUpserter class that generates stable chunk IDs
and writes chunk records with embeddings to a vector store backend.
"""

import hashlib
from typing import List, Optional
from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.core.trace import TraceContext


class VectorUpserter:
    """Upserts chunk records with embeddings to vector store

    This upserter generates deterministic chunk IDs based on content and
    metadata, then writes records to a vector store backend with idempotent
    behavior (same content produces same ID).

    Attributes:
        vector_store: The vector store backend instance
    """

    def __init__(self, vector_store: BaseVectorStore):
        """Initialize the vector upserter

        Args:
            vector_store: An instance of BaseVectorStore for storage

        Raises:
            TypeError: If vector_store is not a BaseVectorStore instance
        """
        if not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store must be an instance of BaseVectorStore")
        self.vector_store = vector_store

    def _generate_chunk_id(self, record: ChunkRecord) -> str:
        """Generate deterministic chunk ID

        The ID is generated as: hash(source_path + chunk_index + content_hash[:8])
        This ensures:
        - Same content in same location = same ID (idempotent)
        - Content changes = different ID
        - Different locations = different ID

        Args:
            record: ChunkRecord to generate ID for

        Returns:
            Deterministic chunk ID string
        """
        # Extract source path from metadata
        source_path = record.metadata.get('source_path', '')

        # Extract chunk index from existing ID if available
        # Assuming ID format: {doc_id}_{index:04d}_{hash}
        chunk_index = ''
        if record.id:
            parts = record.id.split('_')
            if len(parts) >= 2:
                chunk_index = parts[1]

        # Generate content hash
        content_hash = hashlib.sha256(record.text.encode('utf-8')).hexdigest()[:8]

        # Combine components
        id_components = f"{source_path}_{chunk_index}_{content_hash}"

        # Generate final ID hash
        final_id = hashlib.sha256(id_components.encode('utf-8')).hexdigest()[:16]

        return final_id

    def upsert(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """Upsert chunk records to vector store

        Args:
            records: List of ChunkRecord objects with dense_vector populated
            trace: Optional trace context for observability

        Returns:
            List of chunk IDs that were upserted

        Raises:
            ValueError: If records list is empty or dense vectors are missing
        """
        if not records:
            raise ValueError("records list cannot be empty")

        # Verify all records have dense vectors
        for record in records:
            if record.dense_vector is None:
                raise ValueError(f"Record {record.id} missing dense_vector")

        # Generate stable IDs and prepare records for vector store
        upsert_records = []
        generated_ids = []

        for record in records:
            # Generate deterministic ID
            chunk_id = self._generate_chunk_id(record)
            generated_ids.append(chunk_id)

            # Prepare record for vector store
            # Vector stores expect dict format with specific fields
            store_record = {
                'id': chunk_id,
                'vector': record.dense_vector,
                'text': record.text,
                'metadata': record.metadata.copy()
            }

            # Add sparse vector if available
            if record.sparse_vector is not None:
                store_record['metadata']['sparse_vector'] = record.sparse_vector

            upsert_records.append(store_record)

        # Call vector store upsert
        self.vector_store.upsert(upsert_records, trace=trace)

        return generated_ids

    def upsert_single(
        self,
        record: ChunkRecord,
        trace: Optional[TraceContext] = None
    ) -> str:
        """Upsert a single chunk record

        Convenience method for upserting a single record.

        Args:
            record: ChunkRecord to upsert
            trace: Optional trace context

        Returns:
            The generated chunk ID
        """
        ids = self.upsert([record], trace=trace)
        return ids[0]
