from typing import List, Dict, Any, Optional
import os
import httpx
from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama Local LLM Implementation

    Supports local HTTP endpoint (default base_url + model).
    Handles connection failure/timeout scenarios with readable errors.

    Attributes:
        base_url: Ollama server address (default: http://localhost:11434)
        model: Model name to use
        timeout: Request timeout in seconds
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # seconds

    def __init__(self, base_url: Optional[str] = None, model: str = "llama2",
                 timeout: Optional[int] = None, temperature: float = 0.7,
                 max_tokens: int = 512, **kwargs):
        """Initialize Ollama LLM client.

        Args:
            base_url: Ollama server URL (default from env OLLAMA_BASE_URL or default localhost)
            model: Model name (default: llama2)
            timeout: Request timeout in seconds (default: 60)
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Maximum tokens to generate (default: 512)
            **kwargs: Additional arguments (ignored but accepted for interface consistency)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use chat API instead of generate API for better compatibility
        self._endpoint = "/api/chat"
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def chat(self, messages: list) -> str:
        """Send chat messages to Ollama and return the response.

        Args:
            messages: List of chat messages in OpenAI format [{"role": "user", "content": "..."}]

        Returns:
            Model's text response

        Raises:
            RuntimeError: If API request fails with detailed error information
            ValueError: If messages input is invalid
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list")

        # Validate message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"message[{i}] must be a dict, got {type(msg).__name__}")
            if "role" not in msg:
                raise ValueError(f"message[{i}] missing required 'role' field")
            if "content" not in msg:
                raise ValueError(f"message[{i}] missing required 'content' field")

        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }

            response = self._client.post(
                self._endpoint,
                json=payload
            )

            # Raise exception for HTTP errors with readable message
            response.raise_for_status()

            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                raise RuntimeError(f"Unexpected Ollama response format: {result}")

            return result["message"]["content"]

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running and accessible. "
                f"You can set OLLAMA_BASE_URL environment variable to customize the address. "
                f"Details: {str(e)}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Request to Ollama server timed out after {self.timeout}s. "
                f"The model may be loading or the query is complex. "
                f"Details: {str(e)}"
            ) from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_body = e.response.text[:500]  # Limit error body size
            raise RuntimeError(
                f"Ollama API returned status {status_code}: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed: {str(e)}") from e

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
