from abc import ABC, abstractmethod
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Response from Vision LLM chat"""
    content: str
    model: Optional[str] = None
    usage: Optional[dict] = None


class BaseVisionLLM(ABC):
    """Abstract interface for Vision LLM backends

    Vision LLMs support multimodal input (text + image) for tasks like
    image captioning, visual question answering, and document understanding.
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        trace: Optional[object] = None
    ) -> ChatResponse:
        """Send text prompt with image to Vision LLM and return response.

        Args:
            text: Text prompt/question about the image
            image: Image input, either:
                - str: File path to image (e.g., "/path/to/image.jpg")
                - bytes: Raw image bytes (will be base64 encoded internally)
            trace: Optional trace context for logging (TraceContext instance)

        Returns:
            ChatResponse containing the model's text response

        Raises:
            ValueError: If input validation fails (invalid image format, empty text, etc.)
            RuntimeError: If API request fails (network error, model error, etc.)

        Notes:
            - Implementations should handle image preprocessing (resize, format conversion)
            - Implementations should validate image size/format before API call
            - Implementations should provide clear error messages including provider name
        """
        pass
