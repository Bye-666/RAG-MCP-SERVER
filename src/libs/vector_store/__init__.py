from .vector_store_factory import VectorStoreFactory
from .qdrant_vector_store import QdrantVectorStore

VectorStoreFactory.register_provider("qdrant", QdrantVectorStore)