"""MCP 响应的多模态内容组装器。

处理从检索结果组装文本和图像内容。
"""

import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes


class MultimodalAssembler:
    """为 MCP 响应组装多模态内容（文本 + 图像）。"""

    def __init__(self, images_base_dir: str = "data/images"):
        """
        初始化多模态组装器。

        参数:
            images_base_dir: 图像存储的基础目录
        """
        self.images_base_dir = Path(images_base_dir)

    def assemble(self, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从检索结果组装多模态内容。

        参数:
            retrieval_results: 包含文本和元数据的检索结果列表

        返回:
            MCP 内容项列表（文本和图像）
        """
        content_items = []

        for result in retrieval_results:
            # 添加文本内容
            text = result.get("text", "")
            if text:
                content_items.append({
                    "type": "text",
                    "text": text
                })

            # 检查元数据中的图像
            metadata = result.get("metadata", {})
            images = metadata.get("images", [])

            # 添加图像内容
            for image_info in images:
                image_content = self._load_image(image_info)
                if image_content:
                    content_items.append(image_content)

        return content_items

    def _load_image(self, image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        从文件加载图像并转换为 base64。

        参数:
            image_info: 包含 path、id 等的图像元数据

        返回:
            MCP 图像内容字典，如果加载失败则返回 None
        """
        try:
            image_path = image_info.get("path")
            if not image_path:
                return None

            # 解析路径（可以是相对或绝对路径）
            if not Path(image_path).is_absolute():
                image_path = self.images_base_dir / image_path
            else:
                image_path = Path(image_path)

            # 检查文件是否存在
            if not image_path.exists():
                return None

            # 读取图像文件
            with open(image_path, "rb") as f:
                image_data = f.read()

            # 转换为 base64
            base64_data = base64.b64encode(image_data).decode("utf-8")

            # 确定 MIME 类型
            mime_type = self._get_mime_type(image_path)

            return {
                "type": "image",
                "data": base64_data,
                "mimeType": mime_type
            }

        except Exception:
            # 静默跳过加载失败的图像
            return None

    def _get_mime_type(self, file_path: Path) -> str:
        """
        获取图像文件的 MIME 类型。

        参数:
            file_path: 图像文件路径

        返回:
            MIME 类型字符串（例如 "image/png"）
        """
        # 尝试从扩展名猜测
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type and mime_type.startswith("image/"):
            return mime_type

        # 根据扩展名默认为常见图像类型
        ext = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml"
        }

        return mime_map.get(ext, "image/png")  # 默认为 PNG
