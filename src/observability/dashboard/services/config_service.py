"""
Configuration service for Dashboard.

Reads and formats system configuration for display.
"""

from typing import Dict, Any

from src.core.settings import Settings


class ConfigService:
    """Service for reading and formatting system configuration"""

    def __init__(self, settings: Settings):
        """
        Initialize config service.

        Args:
            settings: System settings instance
        """
        self.settings = settings

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        llm = self.settings.llm
        return {
            "provider": llm.get("provider", "N/A"),
            "model": llm.get("model", "N/A"),
            "api_base": llm.get("api_base", "default"),
            "temperature": llm.get("temperature", "N/A"),
            "max_tokens": llm.get("max_tokens", "N/A"),
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        emb = self.settings.embedding
        return {
            "provider": emb.get("provider", "N/A"),
            "model": emb.get("model", "N/A"),
            "api_base": emb.get("api_base", "default"),
            "dimensions": emb.get("dimensions", "N/A"),
        }

    def get_splitter_config(self) -> Dict[str, Any]:
        """Get splitter configuration"""
        splitter = self.settings.splitter.get("splitter", {}) if self.settings.splitter else {}
        return {
            "type": splitter.get("provider", "recursive"),
            "chunk_size": splitter.get("chunk_size", 500),
            "chunk_overlap": splitter.get("chunk_overlap", 50),
        }

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        vs = self.settings.vector_store
        return {
            "provider": vs.get("provider", "N/A"),
            "collection": vs.get("collection_name", vs.get("collection", "N/A")),
            "persist_directory": vs.get("persist_directory", vs.get("host", "N/A")),
        }

    def get_reranker_config(self) -> Dict[str, Any]:
        """Get reranker configuration"""
        # Reranker config might be in retrieval section or separate
        reranker = self.settings.retrieval.get("reranker", {})
        if not reranker or not reranker.get("enabled", False):
            return {"enabled": False}

        return {
            "enabled": True,
            "provider": reranker.get("provider", "N/A"),
            "model": reranker.get("model", "default"),
            "top_k": reranker.get("top_k", "N/A"),
        }

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        ret = self.settings.retrieval
        return {
            "dense_top_k": ret.get("dense_top_k", ret.get("top_k", 5)),
            "sparse_top_k": ret.get("sparse_top_k", ret.get("top_k", 5)),
            "final_top_k": ret.get("final_top_k", ret.get("top_k", 5)),
            "rrf_k": ret.get("rrf_k", 60),
        }

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all system configurations"""
        return {
            "llm": self.get_llm_config(),
            "embedding": self.get_embedding_config(),
            "splitter": self.get_splitter_config(),
            "vector_store": self.get_vector_store_config(),
            "reranker": self.get_reranker_config(),
            "retrieval": self.get_retrieval_config(),
        }
