"""查询结果的 MCP 响应构建器。"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from .citation_generator import CitationGenerator, Citation


@dataclass
class MCPResponse:
    """MCP 工具响应格式。"""
    content: List[Dict[str, Any]]
    isError: bool = False


class ResponseBuilder:
    """从检索结果构建 MCP 格式的响应。"""

    def __init__(self, citation_generator: Optional[CitationGenerator] = None):
        """
        初始化响应构建器。

        参数:
            citation_generator: 引用生成器实例（如果为 None 则创建默认实例）
        """
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: List[Dict[str, Any]], query: str) -> MCPResponse:
        """
        从检索结果构建 MCP 响应。

        参数:
            retrieval_results: 检索结果列表
            query: 原始查询文本

        返回:
            包含 markdown 文本和结构化引用的 MCPResponse
        """
        if not retrieval_results:
            return self._build_empty_response(query)

        # 生成引用
        citations = self.citation_generator.generate(retrieval_results)

        # 构建带引用标记的 markdown 文本
        markdown_text = self._build_markdown(query, retrieval_results, citations)

        # 构建结构化引用
        structured_citations = self._build_structured_citations(citations)

        # 构造 MCP 响应
        content = [
            {
                "type": "text",
                "text": markdown_text
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "citations://query-results",
                    "mimeType": "application/json",
                    "text": structured_citations
                }
            }
        ]

        return MCPResponse(content=content, isError=False)

    def _build_empty_response(self, query: str) -> MCPResponse:
        """为空结果构建响应。"""
        message = (
            f"未找到与查询相关的文档: \"{query}\"\n\n"
            "建议:\n"
            "- 尝试不同的关键词或重新表述您的查询\n"
            "- 使用 list_collections 工具检查文档是否已被摄取\n"
            "- 如果指定了集合过滤器，请验证其正确性"
        )

        content = [{"type": "text", "text": message}]
        return MCPResponse(content=content, isError=False)

    def _build_markdown(
        self,
        query: str,
        results: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """构建带引用标记的 markdown 文本。"""
        lines = [
            f"# 查询结果: {query}",
            "",
            f"找到 {len(results)} 个相关文档:",
            ""
        ]

        for idx, citation in enumerate(citations, start=1):
            lines.append(f"## [{idx}] {citation.title}")
            lines.append(f"**来源:** {citation.source} (第 {citation.page} 页)")
            lines.append(f"**相关性分数:** {citation.score:.4f}")
            lines.append("")
            lines.append(citation.snippet)
            lines.append("")

        return "\n".join(lines)

    def _build_structured_citations(self, citations: List[Citation]) -> str:
        """将结构化引用构建为 JSON 字符串。"""
        import json

        citations_data = {
            "citations": [
                {
                    "index": idx,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page": c.page,
                    "title": c.title,
                    "score": c.score
                }
                for idx, c in enumerate(citations, start=1)
            ]
        }

        return json.dumps(citations_data, indent=2)
