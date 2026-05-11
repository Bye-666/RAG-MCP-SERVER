"""RAG 系统的核心数据类型和契约

本模块定义了跨摄取、检索和 MCP 工具使用的核心数据结构，
以确保一致性并避免模块间的耦合。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class Document:
    """表示从文件加载的源文档

    属性:
        id: 唯一文档标识符（通常是文件哈希或基于路径的标识）
        text: 文档的完整文本内容
        metadata: 文档元数据，包括 source_path 和可选字段
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证必需的元数据字段"""
        if 'source_path' not in self.metadata:
            raise ValueError("文档元数据必须包含 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """从字典创建 Document"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """从 JSON 字符串创建 Document"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Chunk:
    """表示从文档中分割的文本块

    属性:
        id: 唯一块标识符（格式：{doc_id}_{index:04d}_{hash}）
        text: 块的文本内容
        metadata: 块元数据（继承自文档 + 块特定的元数据）
        start_offset: 块在原始文档中的起始字符偏移量
        end_offset: 块在原始文档中的结束字符偏移量
        source_ref: 可选的源文档 ID 引用，用于可追溯性
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def __post_init__(self):
        """验证块数据"""
        if not self.text or not self.text.strip():
            raise ValueError("块文本不能为空")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset 必须 >= start_offset")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """从字典创建 Chunk"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Chunk':
        """从 JSON 字符串创建 Chunk"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ChunkRecord:
    """表示带有 embedding 的块，用于存储和检索

    这是扩展了 Chunk 的存储/检索载体，包含向量数据。
    字段随着流水线的进展而演化（C8-C12）。

    属性:
        id: 唯一块标识符（与 Chunk.id 相同）
        text: 块的文本内容
        metadata: 块元数据
        dense_vector: 可选的稠密 embedding 向量（来自 embedding 模型）
        sparse_vector: 可选的稀疏 embedding 向量（来自 BM25/SPLADE）
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """验证块记录数据"""
        if not self.text or not self.text.strip():
            raise ValueError("ChunkRecord 文本不能为空")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkRecord':
        """从字典创建 ChunkRecord"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkRecord':
        """从 JSON 字符串创建 ChunkRecord"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> 'ChunkRecord':
        """从 Chunk 创建 ChunkRecord，可选向量

        参数:
            chunk: 源 Chunk 实例
            dense_vector: 可选的稠密 embedding 向量
            sparse_vector: 可选的稀疏 embedding 向量

        返回:
            ChunkRecord 实例
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# 多模态支持的元数据字段规范
class ImageMetadata:
    """图像元数据结构规范

    这不是一个 dataclass，而是对 metadata.images 字段预期结构的文档说明。

    结构：List[Dict]，其中每个 dict 包含：
        - id (str): 全局唯一图像标识符（格式：{doc_hash}_{page}_{seq}）
        - path (str): 图像文件存储路径（约定：data/images/{collection}/{image_id}.png）
        - page (int): 原始文档中的页码（可选，用于 PDF）
        - text_offset (int): 占位符在 Document.text 中的字符位置（从 0 开始）
        - text_length (int): 占位符字符串的长度（例如 len("[IMAGE: {image_id}]")）
        - position (dict): 原始文档中的物理位置信息（可选，例如 PDF 坐标）

    示例：
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


@dataclass
class RetrievalResult:
    """表示搜索返回的单个检索结果

    由 DenseRetriever、SparseRetriever 和 HybridSearch 使用，
    返回带有分数和元数据的排序结果。

    属性:
        chunk_id: 唯一块标识符
        score: 相关性分数（越高越相关）
        text: 块的文本内容
        metadata: 块元数据（source_path、collection 等）
    """
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建 RetrievalResult"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'RetrievalResult':
        """从 JSON 字符串创建 RetrievalResult"""
        return cls.from_dict(json.loads(json_str))

