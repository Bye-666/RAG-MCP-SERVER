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

    def test_rerank_empty_candidates(self):
        """边界测试：空候选列表"""
        reranker = NoneReranker()
        result = reranker.rerank("query", [])
        assert result == []
        assert isinstance(result, list)

    def test_rerank_empty_query(self):
        """边界测试：空查询字符串"""
        reranker = NoneReranker()
        candidates = [{'id': '1', 'score': 0.5}]
        result = reranker.rerank("", candidates)
        assert result == candidates

    def test_rerank_single_candidate(self):
        """边界测试：单个候选"""
        reranker = NoneReranker()
        candidates = [{'id': '1', 'score': 0.9}]
        result = reranker.rerank("query", candidates)
        assert len(result) == 1
        assert result[0]['id'] == '1'

    def test_rerank_preserves_structure(self):
        """边界测试：保持候选结构完整"""
        reranker = NoneReranker()
        candidates = [
            {'id': '1', 'score': 0.5, 'metadata': {'key': 'value'}},
            {'id': '2', 'score': 0.9, 'extra_field': 'data'}
        ]
        result = reranker.rerank("query", candidates)
        assert result[0]['metadata'] == {'key': 'value'}
        assert result[1]['extra_field'] == 'data'