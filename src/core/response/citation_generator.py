"""检索结果的引用生成器。"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Citation:
    """检索结果的引用信息。"""
    chunk_id: str
    source: str
    page: int
    title: str
    score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        """将引用转换为字典。"""
        return {
            'chunk_id': self.chunk_id,
            'source': self.source,
            'page': self.page,
            'title': self.title,
            'score': self.score,
            'snippet': self.snippet
        }


class CitationGenerator:
    """从检索结果生成引用信息。"""

    def __init__(self, snippet_length: int = 150):
        """
        初始化引用生成器。

        参数:
            snippet_length: 引用中文本片段的最大长度
        """
        self.snippet_length = snippet_length

    def generate(self, retrieval_results: List[Dict[str, Any]]) -> List[Citation]:
        """
        从检索结果生成引用。

        参数:
            retrieval_results: 包含 chunk_id、score、text、metadata 的检索结果列表

        返回:
            包含 source、page、title 等的 Citation 对象列表
        """
        citations = []

        for result in retrieval_results:
            # 提取元数据
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 0)
            title = metadata.get('title', 'Untitled')

            # 创建文本片段
            text = result.get('text', '')
            snippet = self._create_snippet(text)

            citation = Citation(
                chunk_id=result.get('chunk_id', ''),
                source=source,
                page=page,
                title=title,
                score=result.get('score', 0.0),
                snippet=snippet
            )
            citations.append(citation)

        return citations

    def _create_snippet(self, text: str) -> str:
        """如果需要，创建带省略号的文本片段。"""
        if len(text) <= self.snippet_length:
            return text
        return text[:self.snippet_length].rsplit(' ', 1)[0] + '...'
