"""
基础加载器抽象接口。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.core.types import Document


class BaseLoader(ABC):
    """
    文档加载器的抽象基类。

    加载器负责:
    1. 从磁盘读取文件
    2. 提取文本内容
    3. 提取图像（如果适用）
    4. 创建带有适当元数据的 Document 对象
    """

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Document:
        """
        从给定的文件路径加载文档。

        参数:
            path: 要加载的文件路径

        返回:
            包含提取内容和元数据的 Document 对象

        异常:
            FileNotFoundError: 如果文件不存在
            Exception: 如果文件无法解析
        """
        pass
