import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from .base_vector_store import BaseVectorStore
from ...core.trace import TraceContext


class ChromaStore(BaseVectorStore):
    """ChromaDB vector store implementation with local persistence."""

    def __init__(
        self,
        provider: str = "chroma",
        collection_name: str = "default",
        persist_directory: str = "data/db/chroma",
        **kwargs
    ):
        """Initialize ChromaDB client with persistence.

        Args:
            provider: Provider name (for factory compatibility)
            collection_name: Name of the collection to use
            persist_directory: Directory for persistent storage
            **kwargs: Additional arguments (ignored but accepted for interface consistency)
        """
        self.provider = provider
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        """Insert or update records in ChromaDB.

        Args:
            records: List of records with 'id', 'vector', 'text', and optional 'metadata'
            trace: Optional trace context for logging

        Returns:
            List of record IDs that were upserted

        Raises:
            ValueError: If records are invalid or missing required fields
        """
        if not records:
            return []

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {i} must be a dict, got {type(record).__name__}")

            if "id" not in record:
                raise ValueError(f"Record {i} missing required 'id' field")
            if "vector" not in record:
                raise ValueError(f"Record {i} missing required 'vector' field")
            if "text" not in record:
                raise ValueError(f"Record {i} missing required 'text' field")

            ids.append(str(record["id"]))
            embeddings.append(record["vector"])
            documents.append(record["text"])

            # ChromaDB doesn't accept empty metadata dicts, use None instead
            metadata = record.get("metadata", {})
            metadatas.append(metadata if metadata else None)

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        if trace:
            trace.log("vector_store_upsert", {
                "provider": self.provider,
                "collection": self.collection_name,
                "record_count": len(ids)
            })

        return ids

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB with a vector.

        Args:
            vector: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters (ChromaDB where clause)
            trace: Optional trace context for logging

        Returns:
            List of results with 'id', 'score', 'text', and 'metadata'
        """
        if not isinstance(vector, list):
            raise TypeError("vector must be a list")
        if not vector:
            raise ValueError("vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        if trace:
            trace.log("vector_store_query", {
                "provider": self.provider,
                "collection": self.collection_name,
                "top_k": top_k,
                "result_count": len(formatted_results),
                "has_filters": filters is not None
            })

        return formatted_results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve records by their IDs.

        Args:
            ids: List of record IDs to retrieve

        Returns:
            List of records with 'id', 'text', and 'metadata'
        """
        if not ids:
            return []

        results = self.collection.get(
            ids=[str(id) for id in ids],
            include=["documents", "metadatas"]
        )

        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

        return formatted_results

    def delete_collection(self):
        """Delete the entire collection. Useful for testing cleanup."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection might not exist

    def count(self) -> int:
        """Get the number of records in the collection."""
        return self.collection.count()
