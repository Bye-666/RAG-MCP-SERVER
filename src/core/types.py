"""Core data types and contracts for the RAG system

This module defines the core data structures used across ingestion, retrieval,
and MCP tools to ensure consistency and avoid coupling between modules.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class Document:
    """Represents a source document loaded from a file

    Attributes:
        id: Unique document identifier (typically file hash or path-based)
        text: Full text content of the document
        metadata: Document metadata including source_path and optional fields
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required metadata fields"""
        if 'source_path' not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create Document from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Chunk:
    """Represents a text chunk split from a document

    Attributes:
        id: Unique chunk identifier (format: {doc_id}_{index:04d}_{hash})
        text: Chunk text content
        metadata: Chunk metadata (inherited from document + chunk-specific)
        start_offset: Character offset where chunk starts in original document
        end_offset: Character offset where chunk ends in original document
        source_ref: Optional reference to source document ID for traceability
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def __post_init__(self):
        """Validate chunk data"""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create Chunk from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Chunk':
        """Create Chunk from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ChunkRecord:
    """Represents a chunk with embeddings for storage and retrieval

    This is the storage/retrieval carrier that extends Chunk with vector data.
    Fields evolve as the pipeline progresses (C8-C12).

    Attributes:
        id: Unique chunk identifier (same as Chunk.id)
        text: Chunk text content
        metadata: Chunk metadata
        dense_vector: Optional dense embedding vector (from embedding models)
        sparse_vector: Optional sparse embedding vector (from BM25/SPLADE)
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate chunk record data"""
        if not self.text or not self.text.strip():
            raise ValueError("ChunkRecord text cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkRecord':
        """Create ChunkRecord from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkRecord':
        """Create ChunkRecord from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> 'ChunkRecord':
        """Create ChunkRecord from Chunk with optional vectors

        Args:
            chunk: Source Chunk instance
            dense_vector: Optional dense embedding vector
            sparse_vector: Optional sparse embedding vector

        Returns:
            ChunkRecord instance
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# Metadata field specifications for multimodal support
class ImageMetadata:
    """Specification for image metadata structure

    This is not a dataclass but a documentation of the expected structure
    for metadata.images field.

    Structure: List[Dict] where each dict contains:
        - id (str): Global unique image identifier (format: {doc_hash}_{page}_{seq})
        - path (str): Image file storage path (convention: data/images/{collection}/{image_id}.png)
        - page (int): Page number in original document (optional, for PDFs)
        - text_offset (int): Character position of placeholder in Document.text (0-indexed)
        - text_length (int): Length of placeholder string (e.g., len("[IMAGE: {image_id}]"))
        - position (dict): Physical position info in original document (optional, e.g., PDF coordinates)

    Example:
        metadata = {
            "source_path": "/path/to/doc.pdf",
            "images": [
                {
                    "id": "abc123_1_0",
                    "path": "data/images/default/abc123_1_0.png",
                    "page": 1,
                    "text_offset": 150,
                    "text_length": 20,
                    "position": {"x": 100, "y": 200, "width": 400, "height": 300}
                }
            ]
        }
    """
    pass
