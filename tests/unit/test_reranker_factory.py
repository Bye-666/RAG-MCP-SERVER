import pytest
from src.libs.reranker.reranker_factory import RerankerFactory, NoneReranker


class TestRerankerFactory:
    def setup_method(self):
        # Reset providers but keep 'none'
        RerankerFactory._providers = {"none": NoneReranker}

    def test_create_none_reranker(self):
        settings = {"reranker": {"provider": "none"}}
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_none_reranker_preserves_order(self):
        reranker = NoneReranker()
        candidates = [{'id': '1', 'score': 0.5}, {'id': '2', 'score': 0.9}]
        result = reranker.rerank("query", candidates)
        assert result == candidates

    def test_create_invalid_provider(self):
        settings = {"reranker": {"provider": "invalid"}}
        with pytest.raises(ValueError):
            RerankerFactory.create(settings)

    def test_default_to_none(self):
        settings = {}
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)