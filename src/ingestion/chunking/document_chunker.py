"""
DocumentChunker: libs.splitter 和 core.types 之间的适配器层。

将 Document 对象转换为 List[Chunk]，包含业务逻辑：
- 块 ID 生成
- 元数据继承
- 图像引用分发
- 源引用跟踪
"""
import hashlib
import re
from typing import List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """
    使用 libs.splitter 将 Document 对象转换为 Chunk 对象的适配器。

    职责：
    1. 生成唯一且确定性的块 ID
    2. 从父文档继承元数据
    3. 添加 chunk_index 用于排序
    4. 建立到父文档的 source_ref
    5. 将图像引用分发到引用它们的块
    6. 将分割器的 List[str] 转换为 List[Chunk]
    """

    def __init__(self, settings):
        """
        使用设置初始化 DocumentChunker。

        Args:
            settings: 包含分割器配置的设置对象
        """
        self.settings = settings
        self.splitter = SplitterFactory.create(settings.splitter)

    def split_document(self, document: Document) -> List[Chunk]:
        """
        将文档分割为块列表。

        Args:
            document: 要分割的文档对象

        Returns:
            包含元数据和引用的块对象列表
        """
        # 使用分割器获取文本块
        text_chunks = self.splitter.split_text(document.text)

        # 使用业务逻辑转换为 Chunk 对象
        chunks = []
        for idx, text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(document.id, idx, text)
            metadata = self._inherit_metadata(document, idx, text)

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                source_ref=document.id
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        生成唯一且确定性的块 ID。

        格式: {doc_id}_{index:04d}_{hash_8chars}

        Args:
            doc_id: 父文档 ID
            index: 文档中的块索引
            text: 块文本内容

        Returns:
            块 ID 字符串
        """
        # 计算文本哈希以确保唯一性
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict:
        """
        从文档继承元数据并添加块特定字段。

        同时将图像引用分发到包含图像占位符的块。

        Args:
            document: 父文档
            chunk_index: 此块的索引
            chunk_text: 此块的文本内容

        Returns:
            块的元数据字典
        """
        # 复制所有文档元数据
        metadata = document.metadata.copy()

        # 添加 chunk_index
        metadata["chunk_index"] = chunk_index

        # 分发图像引用
        self._distribute_images(metadata, chunk_text)

        return metadata

    def _distribute_images(self, metadata: dict, chunk_text: str) -> None:
        """
        根据占位符将图像引用分发到块元数据。

        扫描 chunk_text 中的 [IMAGE: {id}] 占位符，并从文档级
        metadata["images"] 中提取匹配的图像。

        就地修改元数据：
        - 添加 metadata["images"]: 此块的 ImageRef 字典列表
        - 添加 metadata["image_refs"]: image_id 字符串列表
        - 如果未找到占位符，则删除文档级 "images"

        Args:
            metadata: 块元数据字典（将就地修改）
            chunk_text: 要扫描占位符的文本内容
        """
        # 从块文本中的占位符提取图像 ID
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, chunk_text)
        image_ids = [m.strip() for m in matches]

        if not image_ids:
            # 此块中没有图像占位符，删除 images 字段
            metadata.pop("images", None)
            return

        # 获取文档级图像
        doc_images = metadata.get("images", [])
        if not doc_images:
            # 文档没有图像元数据
            return

        # 过滤此块中引用的图像
        chunk_images = []
        for img in doc_images:
            if img.get("image_id") in image_ids:
                chunk_images.append(img)

        # 使用块特定图像更新元数据
        if chunk_images:
            metadata["images"] = chunk_images
            metadata["image_refs"] = image_ids
        else:
            # 未找到匹配的图像，删除字段
            metadata.pop("images", None)
