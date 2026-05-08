import pytest
from unittest.mock import MagicMock, patch
import httpx
from src.libs.embedding.ollama_embedding import OllamaEmbedding

class TestOllamaEmbedding:
    """Test cases for OllamaEmbedding class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        embedding = OllamaEmbedding()
        assert embedding.base_url == "http://localhost:11434"
        assert embedding.model == "nomic-embed-text"
        assert embedding.timeout == 60

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        embedding = OllamaEmbedding(
            base_url="http://custom:11434",
            model="mistral",
            timeout=120
        )
        assert embedding.base_url == "http://custom:11434"
        assert embedding.model == "mistral"
        assert embedding.timeout == 120

    def test_init_from_env_base_url(self, monkeypatch):
        """Test that environment variable OLLAMA_BASE_URL is respected."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-server:11434")
        embedding = OllamaEmbedding()
        assert embedding.base_url == "http://env-server:11434"

    def test_init_env_overrides_none_explicit(self, monkeypatch):
        """Test that env var takes precedence when base_url is explicitly None."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        embedding = OllamaEmbedding(base_url=None)  # Explicit None should still use env
        assert embedding.base_url == "http://from-env:11434"

    def test_init_env_trailing_slash_stripped(self, monkeypatch):
        """Test that trailing slashes are stripped from base_url."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://server:11434/")
        embedding = OllamaEmbedding()
        assert embedding.base_url == "http://server:11434"

    def test_embed_normal_success(self):
        """Test normal embed completion success path."""
        embedding = OllamaEmbedding(model="test-model")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
        }

        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            result = embedding.embed(["text1", "text2"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.post.assert_called_once_with(
            "/api/embeddings",
            json={
                "model": "test-model",
                "input": ["text1", "text2"]
            },
            timeout=60
        )

    def test_embed_multiple_texts(self):
        """Test embed with multiple text inputs."""
        embedding = OllamaEmbedding(model="test-model")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
        }

        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            result = embedding.embed([
                "text1",
                "text2",
                "text3"
            ])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    def test_embed_empty_list(self):
        """Test that empty texts list raises ValueError."""
        embedding = OllamaEmbedding()
        with pytest.raises(ValueError, match="texts must be a non-empty list"):
            embedding.embed([])

    def test_embed_single_text(self):
        """Test embed with single text input."""
        embedding = OllamaEmbedding()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            result = embedding.embed(["single text"])

        assert result == [[0.1, 0.2, 0.3]]

    def test_embed_non_list(self):
        """Test that non-list input raises ValueError."""
        embedding = OllamaEmbedding()
        with pytest.raises(ValueError, match="texts must be a non-empty list"):
            embedding.embed("not a list")

    def test_embed_invalid_text_type(self):
        """Test that non-string input raises ValueError."""
        embedding = OllamaEmbedding()
        with pytest.raises(ValueError, match="text must be a string"):
            embedding.embed([123, "text2"])

    def test_embed_connection_error(self):
        """Test that connection errors produce readable messages."""
        embedding = OllamaEmbedding(base_url="http://localhost:11434")

        with patch('httpx.Client') as mock_client_class:
            mock_client_class.side_effect = httpx.ConnectError(
                "Connection refused",
                request=MagicMock()
            )

            with pytest.raises(RuntimeError) as exc_info:
                embedding.embed(["Hello"])

            error_msg = str(exc_info.value)
            assert "Cannot connect to Ollama server" in error_msg
            assert "localhost:11434" in error_msg
            assert "OLLAMA_BASE_URL" in error_msg

    def test_embed_timeout_error(self):
        """Test that timeout errors produce readable messages."""
        embedding = OllamaEmbedding(timeout=1)

        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.post.side_effect = httpx.TimeoutException(
                "Request timed out",
                request=MagicMock()
            )
            mock_client_class.return_value = mock_client_instance

            with pytest.raises(RuntimeError) as exc_info:
                embedding.embed(["Hello"])

            error_msg = str(exc_info.value)
            assert "timed out" in error_msg.lower()
            assert "1s" in error_msg

    def test_embed_http_status_error(self):
        """Test HTTP status error handling."""
        embedding = OllamaEmbedding(model="test")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=mock_response
        )
        mock_response.status_code = 404
        mock_response.text = "Model not found"

        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                embedding.embed(["Hello"])

            error_msg = str(exc_info.value)
            assert "404" in error_msg

    def test_embed_unexpected_response_format(self):
        """Test handling of unexpected response format."""
        embedding = OllamaEmbedding()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "something went wrong"}  # Missing expected structure

        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                embedding.embed(["Hello"])

            assert "Unexpected Ollama response format" in str(exc_info.value)

    def test_context_manager(self):
        """Test context manager protocol."""
        embedding = OllamaEmbedding()
        with patch.object(embedding, '_client') as mock_client:
            with embedding:
                pass
            mock_client.close.assert_called_once()

    def test_embed_invalid_model_name(self):
        """Test embed with invalid model name."""
        with pytest.raises(RuntimeError) as exc_info:
            OllamaEmbedding(model="invalid-model")
        assert "Invalid model name" in str(exc_info.value)

    def test_embed_custom_model_name(self):
        """Test embed with custom model name."""
        embedding = OllamaEmbedding(model="test-model")
        with patch.object(embedding, '_client') as mock_client:
            mock_client.post.return_value = MagicMock()
            embedding.embed(["text"])

        call_args = mock_client.post.call_args[1]['json']
        assert call_args["model"] == "test-model"

    def test_embed_no_input(self):
        """Test embed with no input."""
        with pytest.raises(ValueError, match="texts must be a non-empty list"):
            OllamaEmbedding().embed([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
