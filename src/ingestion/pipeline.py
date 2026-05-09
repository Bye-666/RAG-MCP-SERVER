"""
Ingestion Pipeline for document processing.

Orchestrates the complete ingestion flow:
integrity → load → split → transform → encode → store
"""

from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass

from src.core.types import Document, Chunk, ChunkRecord
from src.core.trace import TraceContext
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.loader.base_loader import BaseLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.image_storage import ImageStorage


@dataclass
class PipelineConfig:
    """Configuration for ingestion pipeline"""
    collection: str = "default"
    force_reprocess: bool = False
    enable_transforms: bool = True
    enable_dense_encoding: bool = True
    enable_sparse_encoding: bool = True


class IngestionPipeline:
    """
    Orchestrates the complete document ingestion pipeline.

    Pipeline stages:
    1. Integrity check (skip if already processed)
    2. Load document
    3. Split into chunks
    4. Transform (refine, enrich, caption)
    5. Encode (dense + sparse vectors)
    6. Store (vector store + BM25 index + images)
    """

    def __init__(
        self,
        integrity_checker: FileIntegrityChecker,
        loader: BaseLoader,
        chunker: DocumentChunker,
        transforms: Optional[List[BaseTransform]] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize ingestion pipeline.

        Args:
            integrity_checker: File integrity checker
            loader: Document loader
            chunker: Document chunker
            transforms: List of transform operations (optional)
            batch_processor: Batch processor for encoding (optional)
            vector_upserter: Vector store upserter (optional)
            bm25_indexer: BM25 indexer (optional)
            image_storage: Image storage (optional)
            on_progress: Progress callback function(stage_name, current, total)
        """
        self.integrity_checker = integrity_checker
        self.loader = loader
        self.chunker = chunker
        self.transforms = transforms or []
        self.batch_processor = batch_processor
        self.vector_upserter = vector_upserter
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.on_progress = on_progress

    def _report_progress(self, stage_name: str, current: int, total: int):
        """Report progress to callback if available"""
        if self.on_progress:
            self.on_progress(stage_name, current, total)

    def ingest_file(
        self,
        file_path: str,
        config: Optional[PipelineConfig] = None,
        trace: Optional[TraceContext] = None
    ) -> dict:
        """
        Ingest a single file through the complete pipeline.

        Args:
            file_path: Path to file to ingest
            config: Pipeline configuration
            trace: Optional trace context

        Returns:
            Dictionary with ingestion results:
            - skipped: bool
            - file_hash: str
            - chunk_count: int
            - image_count: int
            - error: str (if failed)

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For other processing errors
        """
        if config is None:
            config = PipelineConfig()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = {
            "file_path": str(file_path),
            "skipped": False,
            "file_hash": None,
            "chunk_count": 0,
            "image_count": 0,
            "error": None
        }

        try:
            # Stage 1: Integrity check
            self._report_progress("integrity_check", 0, 1)
            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            result["file_hash"] = file_hash

            if not config.force_reprocess and self.integrity_checker.should_skip(file_hash):
                result["skipped"] = True
                self._report_progress("integrity_check", 1, 1)
                return result

            # Stage 2: Load document
            if trace:
                stage = trace.record_stage("load", {
                    "file": str(file_path),
                    "method": self.loader.__class__.__name__
                })

            self._report_progress("load", 0, 1)
            document = self.loader.load(str(file_path))
            self._report_progress("load", 1, 1)

            if trace:
                trace.finish_stage(stage, {
                    "doc_id": document.id,
                    "text_length": len(document.text)
                })

            # Stage 3: Split into chunks
            if trace:
                stage = trace.record_stage("split", {
                    "doc_id": document.id,
                    "method": self.chunker.__class__.__name__
                })

            self._report_progress("split", 0, 1)
            chunks = self.chunker.split_document(document)
            result["chunk_count"] = len(chunks)
            self._report_progress("split", 1, 1)

            if trace:
                trace.finish_stage(stage, {"chunk_count": len(chunks)})

            # Stage 4: Transform (optional)
            if config.enable_transforms and self.transforms:
                for idx, transform in enumerate(self.transforms):
                    if trace:
                        stage = trace.record_stage("transform", {
                            "method": transform.__class__.__name__,
                            "chunk_count": len(chunks)
                        })

                    self._report_progress("transform", idx, len(self.transforms))
                    chunks = transform.transform(chunks, trace=trace)
                    self._report_progress("transform", idx + 1, len(self.transforms))

                    if trace:
                        trace.finish_stage(stage, {"output_count": len(chunks)})

            # Stage 5: Encode
            records = []
            if self.batch_processor:
                if trace:
                    stage = trace.record_stage("embed", {
                        "chunk_count": len(chunks),
                        "method": self.batch_processor.__class__.__name__
                    })

                self._report_progress("encode", 0, len(chunks))
                records = self.batch_processor.process(chunks, trace=trace)
                self._report_progress("encode", len(chunks), len(chunks))

                if trace:
                    trace.finish_stage(stage, {
                        "record_count": len(records),
                        "dense_count": sum(1 for r in records if r.dense_vector is not None),
                        "sparse_count": sum(1 for r in records if r.sparse_vector is not None)
                    })
            else:
                # No encoding, convert chunks to records
                records = [ChunkRecord.from_chunk(chunk) for chunk in chunks]

            # Stage 6: Store
            # 6a. Vector store (if dense vectors available)
            if self.vector_upserter and config.enable_dense_encoding:
                dense_records = [r for r in records if r.dense_vector is not None]
                if dense_records:
                    if trace:
                        stage = trace.record_stage("upsert", {
                            "count": len(dense_records),
                            "method": self.vector_upserter.__class__.__name__,
                            "store_type": "vector"
                        })

                    self._report_progress("upsert", 0, len(dense_records))
                    self.vector_upserter.upsert(dense_records, trace=trace)
                    self._report_progress("upsert", len(dense_records), len(dense_records))

                    if trace:
                        trace.finish_stage(stage, {"success": True})

            # 6b. BM25 index (if sparse vectors available)
            if self.bm25_indexer and config.enable_sparse_encoding:
                sparse_records = [r for r in records if r.sparse_vector is not None]
                if sparse_records:
                    self._report_progress("bm25_index", 0, len(sparse_records))
                    self.bm25_indexer.build(sparse_records)
                    self.bm25_indexer.save()
                    self._report_progress("bm25_index", len(sparse_records), len(sparse_records))

            # 6c. Images (if any)
            if self.image_storage and "images" in document.metadata:
                images = document.metadata.get("images", [])
                result["image_count"] = len(images)
                if images:
                    self._report_progress("store_images", 0, len(images))
                    # Images are already saved by loader, just record in index
                    # (This is a placeholder - actual implementation depends on loader behavior)
                    self._report_progress("store_images", len(images), len(images))

            # Mark as successfully processed
            self.integrity_checker.mark_success(
                file_hash=file_hash,
                file_path=str(file_path),
                chunk_count=result["chunk_count"],
                image_count=result["image_count"]
            )

            self._report_progress("completed", 1, 1)
            return result

        except Exception as e:
            result["error"] = str(e)
            self.integrity_checker.mark_failed(file_hash, str(e))
            self._report_progress("failed", 0, 1)
            raise

    def ingest_directory(
        self,
        directory_path: str,
        pattern: str = "*.pdf",
        config: Optional[PipelineConfig] = None,
        trace: Optional[TraceContext] = None
    ) -> List[dict]:
        """
        Ingest all files in a directory matching pattern.

        Args:
            directory_path: Path to directory
            pattern: File pattern (default: *.pdf)
            config: Pipeline configuration
            trace: Optional trace context

        Returns:
            List of ingestion results for each file
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files = list(directory.glob(pattern))
        results = []

        for file_path in files:
            try:
                result = self.ingest_file(str(file_path), config=config, trace=trace)
                results.append(result)
            except Exception as e:
                results.append({
                    "file_path": str(file_path),
                    "skipped": False,
                    "error": str(e)
                })

        return results
