"""
文档处理的数据摄取管道。

协调完整的数据摄取流程：
完整性检查 → 加载 → 分割 → 转换 → 编码 → 存储
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
    """数据摄取管道的配置"""
    collection: str = "default"
    force_reprocess: bool = False
    enable_transforms: bool = True
    enable_dense_encoding: bool = True
    enable_sparse_encoding: bool = True


class IngestionPipeline:
    """
    协调完整的文档数据摄取管道。

    管道阶段：
    1. 完整性检查（如果已处理则跳过）
    2. 加载文档
    3. 分割为块
    4. 转换（精炼、增强、标题生成）
    5. 编码（密集 + 稀疏向量）
    6. 存储（向量存储 + BM25 索引 + 图像）
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
        初始化数据摄取管道。

        Args:
            integrity_checker: 文件完整性检查器
            loader: 文档加载器
            chunker: 文档分块器
            transforms: 转换操作列表（可选）
            batch_processor: 用于编码的批处理器（可选）
            vector_upserter: 向量存储上传器（可选）
            bm25_indexer: BM25 索引器（可选）
            image_storage: 图像存储（可选）
            on_progress: 进度回调函数(stage_name, current, total)
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
        """如果可用，向回调报告进度"""
        if self.on_progress:
            self.on_progress(stage_name, current, total)

    def ingest_file(
        self,
        file_path: str,
        config: Optional[PipelineConfig] = None,
        trace: Optional[TraceContext] = None
    ) -> dict:
        """
        通过完整管道摄取单个文件。

        Args:
            file_path: 要摄取的文件路径
            config: 管道配置
            trace: 可选的跟踪上下文

        Returns:
            包含摄取结果的字典：
            - skipped: bool
            - file_hash: str
            - chunk_count: int
            - image_count: int
            - error: str（如果失败）

        Raises:
            FileNotFoundError: 如果文件不存在
            Exception: 其他处理错误
        """
        if config is None:
            config = PipelineConfig()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        result = {
            "file_path": str(file_path),
            "skipped": False,
            "file_hash": None,
            "chunk_count": 0,
            "image_count": 0,
            "error": None
        }

        try:
            # 阶段 1: 完整性检查
            self._report_progress("integrity_check", 0, 1)
            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            result["file_hash"] = file_hash

            if not config.force_reprocess and self.integrity_checker.should_skip(file_hash):
                result["skipped"] = True
                self._report_progress("integrity_check", 1, 1)
                return result

            # 阶段 2: 加载文档
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

            # 阶段 3: 分割为块
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

            # 阶段 4: 转换（可选）
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

            # 阶段 5: 编码
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
                # 无编码，将块转换为记录
                records = [ChunkRecord.from_chunk(chunk) for chunk in chunks]

            # 阶段 6: 存储
            # 6a. 向量存储（如果密集向量可用）
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

            # 6b. BM25 索引（如果稀疏向量可用）
            if self.bm25_indexer and config.enable_sparse_encoding:
                sparse_records = [r for r in records if r.sparse_vector is not None]
                if sparse_records:
                    self._report_progress("bm25_index", 0, len(sparse_records))
                    self.bm25_indexer.build(sparse_records)
                    self.bm25_indexer.save()
                    self._report_progress("bm25_index", len(sparse_records), len(sparse_records))

            # 6c. 图像（如果有）
            if self.image_storage and "images" in document.metadata:
                images = document.metadata.get("images", [])
                result["image_count"] = len(images)
                if images:
                    self._report_progress("store_images", 0, len(images))
                    # 图像已由加载器保存，只需在索引中记录
                    # （这是一个占位符 - 实际实现取决于加载器行为）
                    self._report_progress("store_images", len(images), len(images))

            # 标记为成功处理
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
        摄取目录中所有匹配模式的文件。

        Args:
            directory_path: 目录路径
            pattern: 文件模式（默认: *.pdf）
            config: 管道配置
            trace: 可选的跟踪上下文

        Returns:
            每个文件的摄取结果列表
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"目录未找到: {directory_path}")

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
