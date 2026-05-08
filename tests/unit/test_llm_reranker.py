"""Unit tests for LLMReranker"""

import pytest
import json
from unittest.mock import Mock
from src.libs.reranker.llm_reranker import LLMReranker


class TestLLMReranker:
    """Test cases for LLMReranker class"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        llm = Mock()
        llm.chat = Mock()
        return llm

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidates for testing"""
        return [
            {"id": "doc1", "text": "Python programming tutorial", "score": 0.9},
            {"id": "doc2", "text": "Java programming guide", "score": 0.8},
            {"id": "doc3", "text": "JavaScript basics", "score": 0.7}
        ]

    def test_initialization_with_prompt_text(self, mock_llm):
        """Test initialization with custom prompt text"""
        prompt = "Custom prompt: {query} {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)
        assert reranker.prompt_template == prompt

    def test_initialization_with_default_prompt_path(self, mock_llm):
        """Test initialization with default prompt path"""
        reranker = LLMReranker(mock_llm)
        assert "ranked_ids" in reranker.prompt_template
        assert "{query}" in reranker.prompt_template
        assert "{candidates}" in reranker.prompt_template

    def test_initialization_with_custom_prompt_path(self, mock_llm, tmp_path):
        """Test initialization with custom prompt path"""
        prompt_file = tmp_path / "custom_rerank.txt"
        prompt_file.write_text("Custom: {query} {candidates}")

        reranker = LLMReranker(mock_llm, prompt_path=str(prompt_file))
        assert reranker.prompt_template == "Custom: {query} {candidates}"

    def test_initialization_missing_prompt_file(self, mock_llm):
        """Test initialization with non-existent prompt file"""
        with pytest.raises(FileNotFoundError, match="Rerank prompt file not found"):
            LLMReranker(mock_llm, prompt_path="nonexistent.txt")

    def test_rerank_basic(self, mock_llm, sample_candidates):
        """Test basic reranking functionality"""
        # Mock LLM response
        mock_llm.chat.return_value = json.dumps({
            "ranked_ids": ["doc3", "doc1", "doc2"]
        })

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        result = reranker.rerank("test query", sample_candidates)

        assert len(result) == 3
        assert result[0]["id"] == "doc3"
        assert result[1]["id"] == "doc1"
        assert result[2]["id"] == "doc2"

        # Check rerank scores (reciprocal rank)
        assert result[0]["rerank_score"] == 1.0
        assert result[1]["rerank_score"] == 0.5
        assert result[2]["rerank_score"] == pytest.approx(0.333, rel=0.01)

    def test_rerank_with_markdown_code_block(self, mock_llm, sample_candidates):
        """Test parsing LLM response with markdown code blocks"""
        mock_llm.chat.return_value = """```json
{"ranked_ids": ["doc2", "doc3", "doc1"]}
```"""

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        result = reranker.rerank("test query", sample_candidates)

        assert len(result) == 3
        assert result[0]["id"] == "doc2"

    def test_rerank_empty_candidates(self, mock_llm):
        """Test reranking with empty candidates list"""
        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        result = reranker.rerank("test query", [])

        assert result == []
        mock_llm.chat.assert_not_called()

    def test_rerank_missing_id_field(self, mock_llm):
        """Test that missing 'id' field raises ValueError"""
        candidates = [{"text": "some text"}]
        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="missing 'id' field"):
            reranker.rerank("test query", candidates)

    def test_rerank_missing_text_field(self, mock_llm):
        """Test that missing 'text' field raises ValueError"""
        candidates = [{"id": "doc1"}]
        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="missing 'text' field"):
            reranker.rerank("test query", candidates)

    def test_rerank_llm_failure(self, mock_llm, sample_candidates):
        """Test that LLM failure raises RuntimeError for fallback"""
        mock_llm.chat.side_effect = Exception("API error")

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(RuntimeError, match="LLM reranking failed"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_invalid_json(self, mock_llm, sample_candidates):
        """Test that invalid JSON raises ValueError"""
        mock_llm.chat.return_value = "not valid json"

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="not valid JSON"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_missing_ranked_ids_field(self, mock_llm, sample_candidates):
        """Test that missing 'ranked_ids' field raises ValueError"""
        mock_llm.chat.return_value = json.dumps({"wrong_field": []})

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="missing 'ranked_ids' field"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_ranked_ids_not_list(self, mock_llm, sample_candidates):
        """Test that non-list 'ranked_ids' raises ValueError"""
        mock_llm.chat.return_value = json.dumps({"ranked_ids": "not a list"})

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="must be a list"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_missing_ids(self, mock_llm, sample_candidates):
        """Test that missing IDs in ranked output raises ValueError"""
        mock_llm.chat.return_value = json.dumps({
            "ranked_ids": ["doc1", "doc2"]  # Missing doc3
        })

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="missing IDs"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_extra_ids(self, mock_llm, sample_candidates):
        """Test that extra IDs in ranked output raises ValueError"""
        mock_llm.chat.return_value = json.dumps({
            "ranked_ids": ["doc1", "doc2", "doc3", "doc4"]  # Extra doc4
        })

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        with pytest.raises(ValueError, match="extra IDs"):
            reranker.rerank("test query", sample_candidates)

    def test_rerank_with_trace(self, mock_llm, sample_candidates):
        """Test reranking with trace context"""
        mock_llm.chat.return_value = json.dumps({
            "ranked_ids": ["doc1", "doc2", "doc3"]
        })

        trace = Mock()
        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        result = reranker.rerank("test query", sample_candidates, trace=trace)

        assert len(result) == 3
        assert trace.log.call_count >= 2  # At least start and end logs

    def test_rerank_preserves_original_fields(self, mock_llm, sample_candidates):
        """Test that reranking preserves original candidate fields"""
        mock_llm.chat.return_value = json.dumps({
            "ranked_ids": ["doc2", "doc1", "doc3"]
        })

        prompt = "Query: {query}\nCandidates: {candidates}"
        reranker = LLMReranker(mock_llm, prompt_text=prompt)

        result = reranker.rerank("test query", sample_candidates)

        # Check original fields are preserved
        assert result[0]["text"] == "Java programming guide"
        assert result[0]["score"] == 0.8
        assert "rerank_score" in result[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
