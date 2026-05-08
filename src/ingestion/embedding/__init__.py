"""Embedding module for ingestion pipeline

This module handles dense and sparse encoding of text chunks.
"""

from .dense_encoder import DenseEncoder

__all__ = ['DenseEncoder']
