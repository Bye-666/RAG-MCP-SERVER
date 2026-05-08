import pytest
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    def __init__(self, **kwargs):
        pass

    def upsert(self, records, trace=None):
        return [r.get('id', 'fake_id') for r in records]

    def query(self, vector, top_k=5, filters=None, trace=None):
        return [{'id': 'fake_1', 'score': 0.9, 'text': 'test', 'metadata': {}}]

    def get_by_ids(self, ids):
        return [{'id': i, 'text': 'test', 'metadata': {}} for i in ids]


class TestVectorStoreContract:
    def setup_method(self):
        VectorStoreFactory._providers = {}

    def test_register_provider(self):
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        assert "fake" in VectorStoreFactory._providers

    def test_create_valid_provider(self):
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        settings = {"vector_store": {"provider": "fake"}}
        store = VectorStoreFactory.create(settings)
        assert isinstance(store, FakeVectorStore)

    def test_create_invalid_provider(self):
        settings = {"vector_store": {"provider": "invalid"}}
        with pytest.raises(ValueError):
            VectorStoreFactory.create(settings)

    def test_upsert_contract(self):
        store = FakeVectorStore()
        records = [{'id': '1', 'vector': [0.1, 0.2], 'text': 'test'}]
        result = store.upsert(records)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_query_contract(self):
        store = FakeVectorStore()
        result = store.query([0.1, 0.2], top_k=3)
        assert isinstance(result, list)
        assert 'id' in result[0]
        assert 'score' in result[0]

    def test_get_by_ids_contract(self):
        store = FakeVectorStore()
        result = store.get_by_ids(['1', '2'])
        assert isinstance(result, list)
        assert len(result) == 2