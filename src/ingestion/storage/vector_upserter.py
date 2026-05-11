"""用于将块嵌入持久化到向量存储的向量上传器

该模块提供 VectorUpserter 类，生成稳定的块 ID
并将带有嵌入的块记录写入向量存储后端。
"""

import hashlib
from typing import List, Optional
from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.core.trace import TraceContext


class VectorUpserter:
    """将带有嵌入的块记录上传到向量存储

    该上传器基于内容和元数据生成确定性块 ID，
    然后将记录写入向量存储后端，具有幂等行为
    （相同内容产生相同 ID）。

    Attributes:
        vector_store: 向量存储后端实例
    """

    def __init__(self, vector_store: BaseVectorStore):
        """初始化向量上传器

        Args:
            vector_store: 用于存储的 BaseVectorStore 实例

        Raises:
            TypeError: 如果 vector_store 不是 BaseVectorStore 实例
        """
        if not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store 必须是 BaseVectorStore 的实例")
        self.vector_store = vector_store

    def _generate_chunk_id(self, record: ChunkRecord) -> str:
        """生成确定性块 ID

        ID 生成为: hash(source_path + chunk_index + content_hash[:8])
        这确保：
        - 相同位置的相同内容 = 相同 ID（幂等）
        - 内容更改 = 不同 ID
        - 不同位置 = 不同 ID

        Args:
            record: 要生成 ID 的 ChunkRecord

        Returns:
            确定性块 ID 字符串
        """
        # 从元数据中提取源路径
        source_path = record.metadata.get('source_path', '')

        # 如果可用，从现有 ID 中提取块索引
        # 假设 ID 格式: {doc_id}_{index:04d}_{hash}
        chunk_index = ''
        if record.id:
            parts = record.id.split('_')
            if len(parts) >= 2:
                chunk_index = parts[1]

        # 生成内容哈希
        content_hash = hashlib.sha256(record.text.encode('utf-8')).hexdigest()[:8]

        # 组合组件
        id_components = f"{source_path}_{chunk_index}_{content_hash}"

        # 生成最终 ID 哈希
        final_id = hashlib.sha256(id_components.encode('utf-8')).hexdigest()[:16]

        return final_id

    def upsert(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """将块记录上传到向量存储

        Args:
            records: 填充了 dense_vector 的 ChunkRecord 对象列表
            trace: 可选的跟踪上下文，用于可观测性

        Returns:
            已上传的块 ID 列表

        Raises:
            ValueError: 如果记录列表为空或缺少密集向量
        """
        if not records:
            raise ValueError("记录列表不能为空")

        # 验证所有记录都有密集向量
        for record in records:
            if record.dense_vector is None:
                raise ValueError(f"记录 {record.id} 缺少 dense_vector")

        # 生成稳定 ID 并为向量存储准备记录
        upsert_records = []
        generated_ids = []

        for record in records:
            # 生成确定性 ID
            chunk_id = self._generate_chunk_id(record)
            generated_ids.append(chunk_id)

            # 为向量存储准备记录
            # 向量存储期望具有特定字段的字典格式
            store_record = {
                'id': chunk_id,
                'vector': record.dense_vector,
                'text': record.text,
                'metadata': record.metadata.copy()
            }

            # 清理元数据: 删除空列表（某些向量存储不允许它们）
            cleaned_metadata = {}
            for key, value in store_record['metadata'].items():
                if isinstance(value, list) and len(value) == 0:
                    # 跳过空列表
                    continue
                if isinstance(value, dict):
                    # 跳过字典值（例如，sparse_vector 不应在元数据中）
                    continue
                cleaned_metadata[key] = value
            store_record['metadata'] = cleaned_metadata

            # 注意: sparse_vector 不存储在向量存储元数据中
            # 它由管道单独存储在 BM25 索引中

            upsert_records.append(store_record)

        # 调用向量存储上传
        self.vector_store.upsert(upsert_records, trace=trace)

        return generated_ids

    def upsert_single(
        self,
        record: ChunkRecord,
        trace: Optional[TraceContext] = None
    ) -> str:
        """上传单个块记录

        上传单个记录的便捷方法。

        Args:
            record: 要上传的 ChunkRecord
            trace: 可选的跟踪上下文

        Returns:
            生成的块 ID
        """
        ids = self.upsert([record], trace=trace)
        return ids[0]
