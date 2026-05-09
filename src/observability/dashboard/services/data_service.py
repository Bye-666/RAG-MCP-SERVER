"""
数据服务层，封装 ChromaStore 和 ImageStorage 的读取操作。

为 Dashboard 提供统一的数据访问接口。
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.document_manager import DocumentManager, DocumentInfo, DocumentDetail


class DataService:
    """数据服务，封装存储层读取操作"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        image_storage: ImageStorage,
        document_manager: DocumentManager
    ):
        """
        初始化数据服务。

        Args:
            chroma_store: 向量存储实例
            image_storage: 图片存储实例
            document_manager: 文档管理器实例
        """
        self.chroma_store = chroma_store
        self.image_storage = image_storage
        self.document_manager = document_manager

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """
        列出所有文档。

        Args:
            collection: 可选的集合过滤器

        Returns:
            文档信息列表
        """
        return self.document_manager.list_documents(collection)

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        获取文档详细信息。

        Args:
            source_path: 文档源路径

        Returns:
            文档详情对象，如果未找到则返回 None
        """
        return self.document_manager.get_document_detail(source_path)

    def get_chunks_by_source(self, source_path: str) -> List[Dict[str, Any]]:
        """
        根据源路径获取所有 chunk。

        Args:
            source_path: 文档源路径

        Returns:
            Chunk 列表，每个 chunk 包含 id、text、metadata
        """
        return self.chroma_store.get_by_metadata({"source_path": source_path})

    def get_images_by_source(self, source_path: str) -> List[Dict[str, Any]]:
        """
        根据源路径获取所有图片。

        Args:
            source_path: 文档源路径

        Returns:
            图片信息列表
        """
        try:
            return self.image_storage.list_images(source_path)
        except Exception:
            return []

    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        根据 image_id 获取图片路径。

        Args:
            image_id: 图片 ID

        Returns:
            图片文件路径，如果未找到则返回 None
        """
        try:
            return self.image_storage.get_image_path(image_id)
        except Exception:
            return None

    def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Args:
            collection: 可选的集合名称

        Returns:
            统计信息字典
        """
        stats = self.document_manager.get_collection_stats(collection)
        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "total_images": stats.total_images,
            "vector_store_count": stats.vector_store_count,
            "bm25_stats": stats.bm25_stats
        }

    def search_documents(self, keyword: str) -> List[DocumentInfo]:
        """
        根据关键词搜索文档。

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的文档列表
        """
        all_docs = self.list_documents()
        if not keyword:
            return all_docs

        # 简单的文件名匹配
        keyword_lower = keyword.lower()
        return [
            doc for doc in all_docs
            if keyword_lower in doc.source_path.lower()
        ]
