from typing import Optional
from openai import OpenAI
from .base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM Implementation

    Supports DeepSeek API (OpenAI-compatible) with proper error handling and input validation.

    Attributes:
        api_key: DeepSeek API key
        base_url: DeepSeek API base URL
        model: Model name to use (default: deepseek-chat)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7,
                 max_tokens: Optional[int] = None, **kwargs):
        """Initialize DeepSeek LLM client.

        Args:
            api_key: DeepSeek API key
            base_url: DeepSeek API base URL (default: https://api.deepseek.com)
            model: Model name (default: deepseek-chat)
            temperature: Sampling temperature (0.0-2.0, default: 0.7)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments (ignored but accepted for interface consistency)
        """
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list) -> str:
        """Send chat messages to DeepSeek and return the response.

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
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"DeepSeek API request failed: {str(e)}") from e