"""获取文档摘要工具实现。"""

from typing import Dict, Any, Optional
from pathlib import Path


def get_tool_schema() -> Dict[str, Any]:
    """获取 get_document_summary 的工具 schema。"""
    return {
        "name": "get_document_summary",
        "description": "通过文档 ID 获取文档摘要信息，包括标题、摘要和标签",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "文档 ID（文档内容的 SHA256 哈希值）"
                }
            },
            "required": ["doc_id"]
        }
    }


class GetDocumentSummary:
    """获取文档摘要工具处理器。"""

    def __init__(self, vector_store=None):
        """
        初始化获取文档摘要工具。

        Args:
            vector_store: 用于查询文档元数据的向量存储实例
        """
        self.vector_store = vector_store

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 get_document_summary 工具。

        Args:
            arguments: 包含 doc_id 的工具参数

        Returns:
            包含文档摘要的 MCP 工具结果
        """
        try:
            doc_id = arguments.get("doc_id")
            if not doc_id:
                return self._build_error_response("缺少必需参数：doc_id")

            # 获取文档元数据
            doc_info = self._get_document_info(doc_id)

            if not doc_info:
                return self._build_error_response(
                    f"文档未找到：{doc_id}",
                    suggestion="使用 list_collections 工具检查文档是否已导入"
                )

            # 构建响应
            return self._build_success_response(doc_info)

        except Exception as e:
            return self._build_error_response(f"检索文档摘要时出错：{str(e)}")

    def _get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        从向量存储获取文档信息。

        Args:
            doc_id: 文档 ID

        Returns:
            包含文档信息的字典，如果未找到则返回 None
        """
        if not self.vector_store:
            # 回退：返回测试用的模拟数据
            return None

        try:
            # 查询属于此文档的块
            # 块 ID 遵循格式：{doc_id}_{index:04d}_{hash}
            # 我们将获取第一个块，它通常包含文档级元数据
            chunk_id_prefix = f"{doc_id}_0000"

            # 尝试通过 ID 模式获取块
            # 注意：这是一个简化的实现
            # 在生产环境中，您可能希望为 doc_id 添加元数据过滤器
            results = self.vector_store.get_by_ids([chunk_id_prefix])

            if not results:
                return None

            # 从第一个块提取元数据
            first_chunk = results[0]
            metadata = first_chunk.get("metadata", {})

            return {
                "doc_id": doc_id,
                "title": metadata.get("title", "无标题"),
                "summary": metadata.get("summary", "无摘要"),
                "tags": metadata.get("tags", []),
                "source_path": metadata.get("source_path", "未知"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "collection": metadata.get("collection", "default")
            }

        except Exception:
            return None

    def _build_success_response(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """构建包含文档信息的成功响应。"""
        # 构建 markdown 文本
        lines = [
            f"# 文档摘要：{doc_info['title']}",
            "",
            f"**文档 ID：** `{doc_info['doc_id']}`",
            f"**来源：** {doc_info['source_path']}",
            f"**类型：** {doc_info['doc_type']}",
            f"**集合：** {doc_info['collection']}",
            "",
            "## 摘要",
            doc_info['summary'],
            "",
            "## 标签",
            ", ".join(f"`{tag}`" for tag in doc_info['tags']) if doc_info['tags'] else "无标签",
            ""
        ]

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(lines)
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": f"document://{doc_info['doc_id']}/summary",
                        "mimeType": "application/json",
                        "text": self._format_json(doc_info)
                    }
                }
            ],
            "isError": False
        }

    def _build_error_response(self, message: str, suggestion: str = "") -> Dict[str, Any]:
        """构建错误响应。"""
        text = f"错误：{message}"
        if suggestion:
            text += f"\n\n建议：{suggestion}"

        return {
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": True
        }

    def _format_json(self, data: Dict[str, Any]) -> str:
        """将数据格式化为 JSON 字符串。"""
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)


# 全局实例
_tool_instance: GetDocumentSummary = None


def get_document_summary(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取文档摘要工具入口点。

    Args:
        arguments: 来自 MCP 客户端的工具参数

    Returns:
        MCP 工具结果
    """
    global _tool_instance

    if _tool_instance is None:
        _tool_instance = GetDocumentSummary()

    return _tool_instance.execute(arguments)
