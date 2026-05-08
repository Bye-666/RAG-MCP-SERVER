"""Storage module for ingestion pipeline

This module handles persistent storage of indexed data including
BM25 indices, vector stores, and image files.
"""

from .bm25_indexer import BM25Indexer

__all__ = ['BM25Indexer']
