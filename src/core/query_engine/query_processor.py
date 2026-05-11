"""
用于关键词提取和过滤器解析的查询处理器。

将原始用户查询处理为结构化格式以便检索:
- 使用分词和停用词过滤提取关键词
- 解析元数据过滤器（当前为占位符实现）
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ProcessedQuery:
    """已处理查询的结构化表示

    属性:
        raw_query: 原始用户查询字符串
        keywords: 提取的检索关键词
        filters: 元数据过滤器（字典格式，可以为空）
    """
    raw_query: str
    keywords: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)


class QueryProcessor:
    """
    将原始查询处理为结构化格式以便检索。

    当前实现:
    - 使用正则表达式的简单分词
    - 基本停用词过滤
    - 占位符过滤器解析（返回空字典）

    未来增强:
    - 查询扩展（同义词/别名）
    - 从自然语言进行高级过滤器解析
    - 特定语言的分词
    """

    # 基本英文停用词
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or'
    }

    def __init__(
        self,
        stopwords: Optional[set] = None,
        min_keyword_length: int = 2,
        max_keywords: Optional[int] = None
    ):
        """
        初始化 QueryProcessor。

        参数:
            stopwords: 自定义停用词集（如果为 None 则使用 DEFAULT_STOPWORDS）
            min_keyword_length: 关键词的最小长度（默认：2）
            max_keywords: 返回的最大关键词数量（None = 无限制）
        """
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_keyword_length = min_keyword_length
        self.max_keywords = max_keywords

    def process(self, query: str, filters: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """
        将原始查询处理为结构化格式。

        参数:
            query: 原始用户查询字符串
            filters: 可选的元数据过滤器（原样传递）

        返回:
            包含提取的关键词和过滤器的 ProcessedQuery

        异常:
            ValueError: 如果查询为空或仅包含空白字符
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        # 提取关键词
        keywords = self._extract_keywords(query)

        # 使用提供的过滤器或空字典
        processed_filters = filters if filters is not None else {}

        return ProcessedQuery(
            raw_query=query,
            keywords=keywords,
            filters=processed_filters
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        使用分词和过滤从查询中提取关键词。

        策略:
        1. 将查询转换为小写
        2. 使用正则表达式分词（字母数字序列）
        3. 按长度和停用词过滤
        4. 如果指定，限制为 max_keywords

        参数:
            query: 原始查询字符串

        返回:
            提取的关键词列表
        """
        # 小写并分词
        query_lower = query.lower()
        tokens = re.findall(r'\b\w+\b', query_lower)

        # 过滤词元
        keywords = [
            token for token in tokens
            if len(token) >= self.min_keyword_length
            and token not in self.stopwords
        ]

        # 如果指定，应用 max_keywords 限制
        if self.max_keywords is not None and len(keywords) > self.max_keywords:
            keywords = keywords[:self.max_keywords]

        return keywords
