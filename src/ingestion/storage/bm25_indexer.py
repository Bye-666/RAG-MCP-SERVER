"""用于构建和持久化倒排索引的 BM25 索引器

该模块提供 BM25Indexer 类，从稀疏编码的块构建倒排索引，
计算 IDF 分数，并将索引持久化到磁盘以供检索。
"""

import os
import pickle
import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from src.core.types import ChunkRecord


@dataclass
class PostingEntry:
    """倒排索引中的单个发布条目

    Attributes:
        chunk_id: 包含此词项的块的 ID
        tf: 词频（按文档长度标准化）
        doc_length: 文档中的词项总数
    """
    chunk_id: str
    tf: float
    doc_length: int


@dataclass
class TermIndex:
    """单个词项的索引条目

    Attributes:
        term: 词项字符串
        idf: 逆文档频率分数
        postings: 此词项的发布条目列表
    """
    term: str
    idf: float
    postings: List[PostingEntry] = field(default_factory=list)


class BM25Indexer:
    """构建和管理 BM25 倒排索引

    该索引器接收带有稀疏向量（词项权重）的 ChunkRecords，
    构建倒排索引，计算 IDF 分数，并将索引持久化到磁盘。

    Attributes:
        index_dir: 存储索引文件的目录
        index: 将词项映射到 TermIndex 对象的字典
        num_documents: 索引中的文档总数
    """

    def __init__(self, index_dir=None, settings=None):
        """初始化 BM25 索引器

        Args:
            index_dir: 存储索引文件的目录（字符串）或用于向后兼容的 Settings 对象
            settings: 可选的 Settings 对象（用于关键字参数兼容性）
        """
        # 根据参数确定 index_dir
        final_index_dir = "data/db/bm25"  # 默认

        # 处理不同的调用模式
        if settings is not None:
            # 使用 settings=settings 关键字参数调用
            # 使用默认 index_dir（将来可以从 settings 中提取）
            pass
        elif index_dir is not None:
            # 检查它是否是 Settings 对象（有 __dict__ 或不是字符串）
            if isinstance(index_dir, str):
                # 它是 index_dir 字符串
                final_index_dir = index_dir
            else:
                # 它是作为第一个位置参数传递的 Settings 对象
                # 使用默认 index_dir
                pass

        self.index_dir = final_index_dir
        self.index: Dict[str, TermIndex] = {}
        self.num_documents: int = 0

        # 如果索引目录不存在则创建
        os.makedirs(self.index_dir, exist_ok=True)

    def build(self, records: List[ChunkRecord]) -> None:
        """从块记录构建倒排索引

        Args:
            records: 填充了 sparse_vector 的 ChunkRecord 对象列表

        Raises:
            ValueError: 如果记录列表为空或缺少稀疏向量
        """
        if not records:
            raise ValueError("记录列表不能为空")

        # 验证所有记录都有稀疏向量
        for record in records:
            if record.sparse_vector is None:
                raise ValueError(f"记录 {record.id} 缺少 sparse_vector")

        # 清除现有索引
        self.index = {}
        self.num_documents = len(records)

        # 构建倒排索引结构
        term_doc_freq: Dict[str, int] = defaultdict(int)
        term_postings: Dict[str, List[PostingEntry]] = defaultdict(list)

        for record in records:
            sparse_vector = record.sparse_vector
            doc_length = len(sparse_vector)

            # 跟踪此文档中出现的词项
            doc_terms = set(sparse_vector.keys())

            # 更新每个唯一词项的文档频率
            for term in doc_terms:
                term_doc_freq[term] += 1

            # 创建发布条目
            for term, tf in sparse_vector.items():
                posting = PostingEntry(
                    chunk_id=record.id,
                    tf=tf,
                    doc_length=doc_length
                )
                term_postings[term].append(posting)

        # 计算 IDF 并构建最终索引
        for term, df in term_doc_freq.items():
            idf = self._calculate_idf(df, self.num_documents)
            self.index[term] = TermIndex(
                term=term,
                idf=idf,
                postings=term_postings[term]
            )

    def _calculate_idf(self, df: int, num_docs: int) -> float:
        """计算词项的 IDF 分数

        使用 BM25 IDF 公式: log((N - df + 0.5) / (df + 0.5))

        Args:
            df: 文档频率（包含该词项的文档数）
            num_docs: 文档总数

        Returns:
            IDF 分数
        """
        return math.log((num_docs - df + 0.5) / (df + 0.5))

    def save(self, filename: str = "index.pkl") -> None:
        """将索引保存到磁盘

        Args:
            filename: 索引文件名
        """
        filepath = os.path.join(self.index_dir, filename)

        # 准备序列化数据
        index_data = {
            'index': self.index,
            'num_documents': self.num_documents
        }

        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)

    def load(self, filename: str = "index.pkl") -> None:
        """从磁盘加载索引

        Args:
            filename: 索引文件名

        Raises:
            FileNotFoundError: 如果索引文件不存在
        """
        filepath = os.path.join(self.index_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"索引文件未找到: {filepath}")

        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        self.index = index_data['index']
        self.num_documents = index_data['num_documents']

    def query(self, terms: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """查询索引以获取相关块

        Args:
            terms: 查询词项列表
            top_k: 返回的顶部结果数

        Returns:
            按分数降序排序的 (chunk_id, score) 元组列表
        """
        if not terms:
            return []

        # 累积每个块的分数
        chunk_scores: Dict[str, float] = defaultdict(float)

        for term in terms:
            term_lower = term.lower()
            if term_lower not in self.index:
                continue

            term_index = self.index[term_lower]

            # 为每个发布添加 IDF 加权分数
            for posting in term_index.postings:
                # 简单评分: TF * IDF
                score = posting.tf * term_index.idf
                chunk_scores[posting.chunk_id] += score

        # 按分数降序排序并返回 top_k
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:top_k]

    def get_term_info(self, term: str) -> Optional[TermIndex]:
        """获取特定词项的索引信息

        Args:
            term: 要查找的词项

        Returns:
            如果词项存在则返回 TermIndex 对象，否则返回 None
        """
        return self.index.get(term.lower())

    def get_stats(self) -> Dict[str, any]:
        """获取索引统计信息

        Returns:
            包含索引统计信息的字典
        """
        return {
            'num_documents': self.num_documents,
            'num_terms': len(self.index),
            'total_postings': sum(len(term_idx.postings) for term_idx in self.index.values())
        }

    def remove_document(self, chunk_ids: List[str]) -> int:
        """通过块 ID 从索引中删除文档

        Args:
            chunk_ids: 要删除的块 ID 列表

        Returns:
            删除的发布数
        """
        if not chunk_ids:
            return 0

        chunk_id_set = set(chunk_ids)
        removed_count = 0
        terms_to_remove = []

        # 删除指定块 ID 的发布
        for term, term_index in self.index.items():
            original_count = len(term_index.postings)
            term_index.postings = [
                posting for posting in term_index.postings
                if posting.chunk_id not in chunk_id_set
            ]
            removed_count += original_count - len(term_index.postings)

            # 标记没有发布的词项以供删除
            if not term_index.postings:
                terms_to_remove.append(term)

        # 删除空词项
        for term in terms_to_remove:
            del self.index[term]

        # 更新文档计数（近似 - 假设每个块是一个文档）
        self.num_documents = max(0, self.num_documents - len(chunk_ids))

        return removed_count
