from .vector_store_factory import VectorStoreFactory

# Optional: Qdrant (requires qdrant-client)
try:
    from .qdrant_vector_store import QdrantVectorStore
    VectorStoreFactory.register_provider("qdrant", QdrantVectorStore)
except ImportError:
    pass

# Optional: Chroma (requires chromadb)
try:
    from .chroma_store import ChromaStore
    VectorStoreFactory.register_provider("chroma", ChromaStore)
except ImportError:
    pass