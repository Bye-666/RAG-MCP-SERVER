from typing import Optional
from openai import AzureOpenAI
from .base_llm import BaseLLM


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM Implementation

    Supports Azure OpenAI API with proper error handling and input validation.

    Attributes:
        api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        deployment_name: Azure deployment name
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """

    def __init__(self, api_key: str, azure_endpoint: str, api_version: str,
                 deployment_name: str, temperature: float = 0.7,
                 max_tokens: Optional[int] = None, **kwargs):
        """Initialize Azure OpenAI LLM client.

        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version (e.g., "2023-07-01-preview")
            deployment_name: Azure deployment name
            temperature: Sampling temperature (0.0-2.0, default: 0.7)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments (ignored but accepted for interface consistency)
        """
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        if not api_version:
            raise ValueError("Azure OpenAI API version is required")
        if not deployment_name:
            raise ValueError("Azure deployment name is required")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list) -> str:
        """Send chat messages to Azure OpenAI and return the response.

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
                "model": self.deployment_name,
                "messages": messages,
                "temperature": self.temperature
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API request failed: {str(e)}") from e