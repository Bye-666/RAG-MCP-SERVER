"""Unit tests for OllamaLLM implementation.

Uses HTTP mocking to test without requiring a real Ollama server.
Covers:
- Normal response handling
- Connection failure with readable error message
- Timeout scenario
- Invalid messages validation
- Environment variable fallback for base_url
"""

import pytest
from unittest.mock import Mock, patch
import httpx

from src.libs.llm.ollama_llm import OllamaLLM


class TestOllamaLLM:
    """Test cases for OllamaLLM class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        llm = OllamaLLM()
        assert llm.base_url == "http://localhost:11434"
        assert llm.model == "llama2"
        assert llm.timeout == 60
        assert llm.temperature == 0.7
        assert llm.max_tokens == 512

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        llm = OllamaLLM(
            base_url="http://custom:11434",
            model="mistral",
            timeout=120,
            temperature=0.9,
            max_tokens=1024
        )
        assert llm.base_url == "http://custom:11434"
        assert llm.model == "mistral"
        assert llm.timeout == 120
        assert llm.temperature == 0.9
        assert llm.max_tokens == 1024

    def test_init_from_env_base_url(self, monkeypatch):
        """Test that environment variable OLLAMA_BASE_URL is respected."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-server:11434")
        llm = OllamaLLM()
        assert llm.base_url == "http://env-server:11434"

    def test_init_env_overrides_none_explicit(self, monkeypatch):
        """Test that env var takes precedence when base_url is explicitly None."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        llm = OllamaLLM(base_url=None)  # Explicit None should still use env
        assert llm.base_url == "http://from-env:11434"

    def test_init_env_trailing_slash_stripped(self, monkeypatch):
        """Test that trailing slashes are stripped from base_url."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://server:11434/")
        llm = OllamaLLM()
        assert llm.base_url == "http://server:11434"

    def test_chat_normal_success(self):
        """Test normal chat completion success path."""
        llm = OllamaLLM(model="test-model")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "This is the model response."
            }
        }

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            result = llm.chat([{"role": "user", "content": "Hello"}])

        assert result == "This is the model response."
        mock_client.post.assert_called_once_with(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 512}
            }
        )

    def test_chat_multiple_messages(self):
        """Test chat with multiple conversation turns."""
        llm = OllamaLLM(model="test-model")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"}
        }

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            result = llm.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG is..."},
                {"role": "user", "content": "Tell me more."}
            ])

        assert result == "Response"

    def test_chat_invalid_empty_list(self):
        """Test that empty messages list raises ValueError."""
        llm = OllamaLLM()
        with pytest.raises(ValueError, match="messages must be a non-empty list"):
            llm.chat([])

    def test_chat_invalid_not_a_list(self):
        """Test that non-list input raises ValueError."""
        llm = OllamaLLM()
        with pytest.raises(ValueError, match="messages must be a non-empty list"):
            llm.chat("not a list")

    def test_chat_message_missing_role(self):
        """Test that message missing role field raises ValueError."""
        llm = OllamaLLM()
        with pytest.raises(ValueError, match="missing required 'role' field"):
            llm.chat([{"content": "no role here"}])

    def test_chat_message_missing_content(self):
        """Test that message missing content field raises ValueError."""
        llm = OllamaLLM()
        with pytest.raises(ValueError, match="missing required 'content' field"):
            llm.chat([{"role": "user"}])

    def test_chat_message_not_dict(self):
        """Test that non-dict message raises ValueError."""
        llm = OllamaLLM()
        with pytest.raises(ValueError, match="must be a dict"):
            llm.chat(["string message"])

    def test_connection_error_readable(self):
        """Test that connection errors produce readable messages."""
        llm = OllamaLLM(base_url="http://localhost:11434")

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.side_effect = httpx.ConnectError(
                "Connection refused",
                request=Mock()
            )

            with pytest.raises(RuntimeError) as exc_info:
                llm.chat([{"role": "user", "content": "Hello"}])

            error_msg = str(exc_info.value)
            assert "Cannot connect to Ollama server" in error_msg
            assert "localhost:11434" in error_msg
            assert "OLLAMA_BASE_URL" in error_msg

    def test_timeout_error_readable(self):
        """Test that timeout errors produce readable messages."""
        llm = OllamaLLM(timeout=1)

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.side_effect = httpx.TimeoutException(
                "Request timed out",
                request=Mock()
            )

            with pytest.raises(RuntimeError) as exc_info:
                llm.chat([{"role": "user", "content": "Hello"}])

            error_msg = str(exc_info.value)
            assert "timed out" in error_msg.lower()
            assert "1s" in error_msg

    def test_http_status_error(self):
        """Test HTTP status error handling."""
        llm = OllamaLLM(model="test")

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=Mock(),
            response=mock_response
        )
        mock_response.status_code = 404
        mock_response.text = "Model not found"

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                llm.chat([{"role": "user", "content": "Hello"}])

            error_msg = str(exc_info.value)
            assert "404" in error_msg

    def test_unexpected_response_format(self):
        """Test handling of unexpected response format."""
        llm = OllamaLLM()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"error": "something went wrong"}  # Missing expected structure

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                llm.chat([{"role": "user", "content": "Hello"}])

            assert "Unexpected Ollama response format" in str(exc_info.value)

    def test_context_manager(self):
        """Test context manager protocol."""
        llm = OllamaLLM()
        with patch.object(llm, '_client') as mock_client:
            with llm:
                pass
            mock_client.close.assert_called_once()

    def test_temperature_and_max_tokens_in_payload(self):
        """Test that custom temperature and max_tokens are passed in payload."""
        llm = OllamaLLM(model="custom", temperature=0.5, max_tokens=2048)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"message": {"content": "OK"}}

        with patch.object(llm, '_client') as mock_client:
            mock_client.post.return_value = mock_response
            llm.chat([{"role": "user", "content": "Test"}])

            call_args = mock_client.post.call_args
            json_payload = call_args[1]['json']

            assert json_payload['options']['temperature'] == 0.5
            assert json_payload['options']['num_predict'] == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
