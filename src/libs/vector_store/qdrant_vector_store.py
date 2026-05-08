from .base_vector_store import BaseVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int, **kwargs):
        self.client = QdrantClient(host=host, port=port, **kwargs)
        self.collection_name = collection_name
        self._create_collection(vector_size)

    def _create_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        points = [
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {})
                }
            ) for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        return [
            {
                "text": hit.payload["text"],
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            } for hit in results
        ]

    def delete(self, ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )