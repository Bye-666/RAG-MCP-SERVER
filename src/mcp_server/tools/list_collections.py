"""列出集合工具实现。"""

import os
from typing import Dict, Any, List
from pathlib import Path


def get_tool_schema() -> Dict[str, Any]:
    """获取 list_collections 的工具 schema。"""
    return {
        "name": "list_collections",
        "description": "列出知识库中所有可用的文档集合",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }


class ListCollections:
    """列出集合工具处理器。"""

    def __init__(self, documents_dir: str = "data/documents"):
        """
        初始化列出集合工具。

        Args:
            documents_dir: 文档目录路径
        """
        self.documents_dir = Path(documents_dir)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 list_collections 工具。

        Args:
            arguments: 工具参数（无需参数）

        Returns:
            包含集合列表的 MCP 工具结果
        """
        try:
            collections = self._get_collections()

            if not collections:
                return {
                    "content": [{
                        "type": "text",
                        "text": "未找到集合。请先导入文档。"
                    }],
                    "isError": False
                }

            # 构建响应文本
            lines = ["# 可用集合", ""]
            for collection in collections:
                lines.append(f"- **{collection['name']}**")
                if collection.get('document_count'):
                    lines.append(f"  - 文档数：{collection['document_count']}")
                lines.append("")

            return {
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(lines)
                    },
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "collections://list",
                            "mimeType": "application/json",
                            "text": self._format_collections_json(collections)
                        }
                    }
                ],
                "isError": False
            }

        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"列出集合时出错：{str(e)}"
                }],
                "isError": True
            }

    def _get_collections(self) -> List[Dict[str, Any]]:
        """
        从文档目录获取集合列表。

        Returns:
            集合信息字典列表
        """
        collections = []

        if not self.documents_dir.exists():
            return collections

        # 列出文档目录中的子目录
        for item in self.documents_dir.iterdir():
            if item.is_dir():
                collection_info = {
                    "name": item.name,
                    "path": str(item),
                    "document_count": self._count_documents(item)
                }
                collections.append(collection_info)

        return sorted(collections, key=lambda x: x['name'])

    def _count_documents(self, collection_path: Path) -> int:
        """
        统计集合目录中的文档数量。

        Args:
            collection_path: 集合目录路径

        Returns:
            文档数量
        """
        count = 0
        for item in collection_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.pdf', '.md', '.txt']:
                count += 1
        return count

    def _format_collections_json(self, collections: List[Dict[str, Any]]) -> str:
        """将集合格式化为 JSON 字符串。"""
        import json
        return json.dumps({"collections": collections}, indent=2)


# 全局实例
_tool_instance: ListCollections = None


def list_collections(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    列出集合工具入口点。

    Args:
        arguments: 来自 MCP 客户端的工具参数

    Returns:
        MCP 工具结果
    """
    global _tool_instance

    if _tool_instance is None:
        _tool_instance = ListCollections()

    return _tool_instance.execute(arguments)
