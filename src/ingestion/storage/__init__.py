"""数据摄取管道的存储模块

该模块处理索引数据的持久化存储，包括
BM25 索引、向量存储和图像文件。
"""

from .bm25_indexer import BM25Indexer

__all__ = ['BM25Indexer']
