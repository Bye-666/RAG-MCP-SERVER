import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from .base_vector_store import BaseVectorStore
from ...core.trace import TraceContext


class ChromaStore(BaseVectorStore):
    """ChromaDB 向量存储实现，支持本地持久化。"""

    def __init__(
        self,
        provider: str = "chroma",
        collection_name: str = "default",
        persist_directory: str = "data/db/chroma",
        **kwargs
    ):
        """初始化 ChromaDB 客户端并启用持久化。

        参数:
            provider: 提供商名称（用于工厂兼容性）
            collection_name: 要使用的集合名称
            persist_directory: 持久化存储目录
            **kwargs: 其他参数（为接口一致性而接受但被忽略）
        """
        self.provider = provider
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 如果持久化目录不存在则创建
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化 ChromaDB 客户端并启用持久化
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """在 ChromaDB 中插入或更新记录。

        参数:
            records: 记录列表，包含 'id'、'vector'、'text' 和可选的 'metadata'
            trace: 可选的跟踪上下文用于日志记录

        返回:
            已插入或更新的记录 ID 列表

        异常:
            ValueError: 如果记录无效或缺少必需字段
        """
        if not records:
            return []

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"记录 {i} 必须是字典，得到 {type(record).__name__}")

            if "id" not in record:
                raise ValueError(f"记录 {i} 缺少必需的 'id' 字段")
            if "vector" not in record:
                raise ValueError(f"记录 {i} 缺少必需的 'vector' 字段")
            if "text" not in record:
                raise ValueError(f"记录 {i} 缺少必需的 'text' 字段")

            ids.append(str(record["id"]))
            embeddings.append(record["vector"])
            documents.append(record["text"])

            # ChromaDB 不接受空元数据字典，使用 None 代替
            metadata = record.get("metadata", {})
            metadatas.append(metadata if metadata else None)

        # 插入或更新到 ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        if trace:
            trace.log("vector_store_upsert", {
                "provider": self.provider,
                "collection": self.collection_name,
                "record_count": len(ids)
            })

        return ids

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """使用向量查询 ChromaDB。

        参数:
            vector: 查询向量
            top_k: 返回的结果数量
            filters: 可选的元数据过滤器（ChromaDB where 子句）
            trace: 可选的跟踪上下文用于日志记录

        返回:
            包含 'id'、'score'、'text' 和 'metadata' 的结果列表
        """
        if not isinstance(vector, list):
            raise TypeError("vector 必须是列表")
        if not vector:
            raise ValueError("vector 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")

        # 查询 ChromaDB
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # 将距离转换为相似度
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        if trace:
            trace.log("vector_store_query", {
                "provider": self.provider,
                "collection": self.collection_name,
                "top_k": top_k,
                "result_count": len(formatted_results),
                "has_filters": filters is not None
            })

        return formatted_results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """通过 ID 检索记录。

        参数:
            ids: 要检索的记录 ID 列表

        返回:
            包含 'id'、'text' 和 'metadata' 的记录列表
        """
        if not ids:
            return []

        results = self.collection.get(
            ids=[str(id) for id in ids],
            include=["documents", "metadatas"]
        )

        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

        return formatted_results

    def delete_collection(self):
        """删除整个集合。用于测试清理。"""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # 集合可能不存在

    def count(self) -> int:
        """获取集合中的记录数量。"""
        return self.collection.count()

    def delete_by_metadata(self, filters: Dict[str, Any]) -> int:
        """删除匹配元数据过滤器的记录。

        参数:
            filters: 元数据过滤器（ChromaDB where 子句）

        返回:
            删除的记录数量
        """
        if not filters:
            return 0

        # 获取匹配过滤器的 ID
        results = self.collection.get(
            where=filters,
            include=[]
        )

        if not results["ids"]:
            return 0

        # 删除匹配的记录
        self.collection.delete(ids=results["ids"])

        return len(results["ids"])

    def get_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """通过元数据过滤器检索记录。

        参数:
            filters: 元数据过滤器（ChromaDB where 子句）

        返回:
            包含 'id'、'text' 和 'metadata' 的记录列表
        """
        if not filters:
            return []

        results = self.collection.get(
            where=filters,
            include=["documents", "metadatas"]
        )

        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

        return formatted_results
