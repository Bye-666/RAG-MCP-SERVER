import os
from typing import List
from .base_embedding import BaseEmbedding
import httpx

class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding Implementation

    Uses Ollama's /api/embeddings endpoint to generate embeddings.
    Uses the same configuration style as the LLM implementation.

    Attributes:
        base_url: Ollama server address
        model: Model name to use
        timeout: Request timeout in seconds
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # seconds

    def __init__(self, base_url: str = None, model: str = "nomic-embed-text", timeout: int = None):
        """Initialize Ollama embedding client.

        Args:
            base_url: Ollama server URL (default from env OLLAMA_BASE_URL or default localhost)
            model: Model name to use (default: nomic-embed-text)
            timeout: Request timeout in seconds (default: 60)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text inputs.

        Args:
            texts: List of input text strings

        Returns:
            List of embeddings (each embedding is a list of floats)

        Raises:
            RuntimeError: If API request fails with detailed error information
            ValueError: If input is invalid
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("texts must be a non-empty list")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"text[{i}] must be a string, got {type(text).__name__}")

        try:
            # Build request payload
            payload = {
                "model": self.model,
                "input": texts
            }

            response = self._client.post(
                "/api/embeddings",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            if "embeddings" not in result:
                raise RuntimeError(f"Unexpected Ollama response format: {result}")

            # Validate shape
            if len(result["embeddings"]) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(result['embeddings'])} embeddings for {len(texts)} texts"
                )

            return result["embeddings"]

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
                f"The request may be too large. "
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