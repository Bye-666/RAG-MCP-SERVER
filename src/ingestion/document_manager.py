"""
Document Manager for cross-storage document lifecycle management.

Coordinates operations across ChromaStore, BM25Indexer, ImageStorage, and FileIntegrity.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityChecker


@dataclass
class DocumentInfo:
    """Information about a document"""
    source_path: str
    file_hash: str
    chunk_count: int
    image_count: int
    created_at: str
    updated_at: str
    status: str


@dataclass
class DocumentDetail:
    """Detailed information about a document"""
    source_path: str
    file_hash: str
    chunks: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class DeleteResult:
    """Result of document deletion"""
    success: bool
    chunks_deleted: int
    images_deleted: int
    bm25_postings_deleted: int
    integrity_record_deleted: bool
    error: Optional[str] = None


@dataclass
class CollectionStats:
    """Statistics for a collection"""
    total_documents: int
    total_chunks: int
    total_images: int
    vector_store_count: int
    bm25_stats: Dict[str, Any]


class DocumentManager:
    """Manages document lifecycle across multiple storage systems"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker
    ):
        """
        Initialize document manager.

        Args:
            chroma_store: Vector store instance
            bm25_indexer: BM25 index instance
            image_storage: Image storage instance
            file_integrity: File integrity checker instance
        """
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """
        List all processed documents.

        Args:
            collection: Optional collection filter (not used in current implementation)

        Returns:
            List of DocumentInfo objects
        """
        # Get all successfully processed files from integrity checker
        processed_files = self.file_integrity.list_processed(status="success")

        documents = []
        for file_record in processed_files:
            metadata = file_record.get("metadata", {})
            documents.append(DocumentInfo(
                source_path=file_record.get("file_path", "unknown"),
                file_hash=file_record["file_hash"],
                chunk_count=metadata.get("chunk_count", 0),
                image_count=metadata.get("image_count", 0),
                created_at=file_record["created_at"],
                updated_at=file_record["updated_at"],
                status=file_record["status"]
            ))

        return documents

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        Get detailed information about a document.

        Args:
            source_path: Source path of the document

        Returns:
            DocumentDetail object or None if not found
        """
        # Query chunks from vector store by source_path
        chunks = self.chroma_store.get_by_metadata({"source_path": source_path})

        if not chunks:
            return None

        # Get images for this document
        images = []
        try:
            doc_images = self.image_storage.list_images(source_path)
            images = doc_images
        except Exception:
            pass  # Images might not exist

        # Extract metadata from first chunk
        metadata = chunks[0].get("metadata", {}) if chunks else {}

        return DocumentDetail(
            source_path=source_path,
            file_hash=metadata.get("file_hash", "unknown"),
            chunks=chunks,
            images=images,
            metadata=metadata
        )

    def delete_document(self, source_path: str, collection: Optional[str] = None) -> DeleteResult:
        """
        Delete a document from all storage systems.

        Args:
            source_path: Source path of the document
            collection: Optional collection name (not used in current implementation)

        Returns:
            DeleteResult with deletion statistics
        """
        result = DeleteResult(
            success=False,
            chunks_deleted=0,
            images_deleted=0,
            bm25_postings_deleted=0,
            integrity_record_deleted=False
        )

        try:
            # 1. Get all chunks for this document from vector store
            chunks = self.chroma_store.get_by_metadata({"source_path": source_path})

            if not chunks:
                result.error = f"No chunks found for source_path: {source_path}"
                return result

            chunk_ids = [chunk["id"] for chunk in chunks]

            # 2. Delete from vector store
            chunks_deleted = self.chroma_store.delete_by_metadata({"source_path": source_path})
            result.chunks_deleted = chunks_deleted

            # 3. Delete from BM25 index
            bm25_deleted = self.bm25_indexer.remove_document(chunk_ids)
            result.bm25_postings_deleted = bm25_deleted

            # Save updated BM25 index
            self.bm25_indexer.save()

            # 4. Delete images
            try:
                images_deleted = self.image_storage.delete_images(source_path)
                result.images_deleted = images_deleted
            except Exception:
                pass  # Images might not exist

            # 5. Remove from integrity checker
            # Get file_hash from metadata
            file_hash = chunks[0].get("metadata", {}).get("file_hash")
            if file_hash:
                integrity_deleted = self.file_integrity.remove_record(file_hash)
                result.integrity_record_deleted = integrity_deleted

            result.success = True
            return result

        except Exception as e:
            result.error = str(e)
            return result

    def get_collection_stats(self, collection: Optional[str] = None) -> CollectionStats:
        """
        Get statistics for a collection.

        Args:
            collection: Optional collection name (not used in current implementation)

        Returns:
            CollectionStats object
        """
        # Get document count from integrity checker
        documents = self.list_documents(collection)
        total_documents = len(documents)
        total_chunks = sum(doc.chunk_count for doc in documents)
        total_images = sum(doc.image_count for doc in documents)

        # Get vector store count
        vector_store_count = self.chroma_store.count()

        # Get BM25 stats
        bm25_stats = self.bm25_indexer.get_stats()

        return CollectionStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_images=total_images,
            vector_store_count=vector_store_count,
            bm25_stats=bm25_stats
        )
