"""
跨存储文档生命周期管理的文档管理器。

协调 ChromaStore、BM25Indexer、ImageStorage 和 FileIntegrity 之间的操作。
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityChecker


@dataclass
class DocumentInfo:
    """关于文档的信息"""
    source_path: str
    file_hash: str
    chunk_count: int
    image_count: int
    created_at: str
    updated_at: str
    status: str


@dataclass
class DocumentDetail:
    """关于文档的详细信息"""
    source_path: str
    file_hash: str
    chunks: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class DeleteResult:
    """文档删除结果"""
    success: bool
    chunks_deleted: int
    images_deleted: int
    bm25_postings_deleted: int
    integrity_record_deleted: bool
    error: Optional[str] = None


@dataclass
class CollectionStats:
    """集合的统计信息"""
    total_documents: int
    total_chunks: int
    total_images: int
    vector_store_count: int
    bm25_stats: Dict[str, Any]


class DocumentManager:
    """跨多个存储系统管理文档生命周期"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker
    ):
        """
        初始化文档管理器。

        Args:
            chroma_store: 向量存储实例
            bm25_indexer: BM25 索引实例
            image_storage: 图像存储实例
            file_integrity: 文件完整性检查器实例
        """
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """
        列出所有已处理的文档。

        Args:
            collection: 可选的集合过滤器（当前实现中未使用）

        Returns:
            DocumentInfo 对象列表
        """
        # 从完整性检查器获取所有成功处理的文件
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
        获取文档的详细信息。

        Args:
            source_path: 文档的源路径

        Returns:
            DocumentDetail 对象，如果未找到则返回 None
        """
        # 通过 source_path 从向量存储查询块
        chunks = self.chroma_store.get_by_metadata({"source_path": source_path})

        if not chunks:
            return None

        # 获取此文档的图像
        images = []
        try:
            doc_images = self.image_storage.list_images(source_path)
            images = doc_images
        except Exception:
            pass  # 图像可能不存在

        # 从第一个块提取元数据
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
        从所有存储系统中删除文档。

        Args:
            source_path: 文档的源路径
            collection: 可选的集合名称（当前实现中未使用）

        Returns:
            包含删除统计信息的 DeleteResult
        """
        result = DeleteResult(
            success=False,
            chunks_deleted=0,
            images_deleted=0,
            bm25_postings_deleted=0,
            integrity_record_deleted=False
        )

        try:
            # 1. 首先从完整性检查器获取 file_hash（最可靠的来源）
            file_hash = None
            processed_files = self.file_integrity.list_processed()
            for file_record in processed_files:
                if file_record.get("file_path") == source_path:
                    file_hash = file_record.get("file_hash")
                    break

            # 2. 从向量存储获取此文档的所有块
            chunks = self.chroma_store.get_by_metadata({"source_path": source_path})

            # 3. 如果块存在，从向量存储和 BM25 中删除
            if chunks:
                chunk_ids = [chunk["id"] for chunk in chunks]

                # 从向量存储删除
                chunks_deleted = self.chroma_store.delete_by_metadata({"source_path": source_path})
                result.chunks_deleted = chunks_deleted

                # 从 BM25 索引删除
                bm25_deleted = self.bm25_indexer.remove_document(chunk_ids)
                result.bm25_postings_deleted = bm25_deleted

                # 保存更新的 BM25 索引
                self.bm25_indexer.save()

                # 如果之前没有从完整性检查器获取到 file_hash，尝试从 metadata 获取
                if not file_hash:
                    file_hash = chunks[0].get("metadata", {}).get("file_hash")

            # 3. 删除图像
            try:
                images_deleted = self.image_storage.delete_images(source_path)
                result.images_deleted = images_deleted
            except Exception:
                pass  # 图像可能不存在

            # 4. 从完整性检查器中删除
            if file_hash:
                integrity_deleted = self.file_integrity.remove_record(file_hash)
                result.integrity_record_deleted = integrity_deleted

            # 如果我们删除了任何内容或删除了完整性记录，则认为成功
            result.success = (result.chunks_deleted > 0 or
                            result.images_deleted > 0 or
                            result.integrity_record_deleted)

            if not result.success and not chunks:
                result.error = f"未找到 source_path 的数据: {source_path}"

            return result

        except Exception as e:
            result.error = str(e)
            return result

    def get_collection_stats(self, collection: Optional[str] = None) -> CollectionStats:
        """
        获取集合的统计信息。

        Args:
            collection: 可选的集合名称（当前实现中未使用）

        Returns:
            CollectionStats 对象
        """
        # 从完整性检查器获取文档计数
        documents = self.list_documents(collection)
        total_documents = len(documents)
        total_chunks = sum(doc.chunk_count for doc in documents)
        total_images = sum(doc.image_count for doc in documents)

        # 获取向量存储计数
        vector_store_count = self.chroma_store.count()

        # 获取 BM25 统计信息
        bm25_stats = self.bm25_indexer.get_stats()

        return CollectionStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_images=total_images,
            vector_store_count=vector_store_count,
            bm25_stats=bm25_stats
        )
