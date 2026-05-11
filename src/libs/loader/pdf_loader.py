"""
使用 markitdown 的 PDF 加载器实现。
"""
import hashlib
from pathlib import Path
from typing import Union

from markitdown import MarkItDown

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    """
    使用 markitdown 库的 PDF 文档加载器。

    从 PDF 文件中提取文本内容并转换为 Markdown 格式。
    """

    def __init__(self):
        """初始化 PDF 加载器。"""
        self._converter = MarkItDown()

    def load(self, path: Union[str, Path]) -> Document:
        """
        从给定的文件路径加载 PDF 文档。

        参数:
            path: PDF 文件路径

        返回:
            包含提取内容和元数据的 Document 对象

        异常:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件不是有效的 PDF
            Exception: 如果 PDF 无法解析
        """
        # 转换为 Path 对象
        file_path = Path(path)

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        # 验证 PDF 格式
        if not self._is_valid_pdf(file_path):
            raise ValueError(f"文件不是有效的 PDF: {file_path}")

        # 基于文件内容哈希计算文档 ID
        doc_id = self._compute_file_hash(file_path)

        # 使用 markitdown 提取文本
        try:
            result = self._converter.convert(str(file_path))
            text_content = result.text_content
        except Exception as e:
            raise Exception(f"解析 PDF 失败: {e}") from e

        # 构建元数据
        metadata = {
            "source_path": str(file_path),
            "doc_type": "pdf",
        }

        # 创建 Document
        doc = Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

        return doc

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        计算文件内容的 SHA256 哈希。

        参数:
            file_path: 文件路径

        返回:
            SHA256 哈希的十六进制字符串
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _is_valid_pdf(self, file_path: Path) -> bool:
        """
        通过检查文件头来检查文件是否为有效的 PDF。

        参数:
            file_path: 文件路径

        返回:
            如果文件以 PDF 头开始则为 True，否则为 False
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                return header == b"%PDF-"
        except Exception:
            return False
