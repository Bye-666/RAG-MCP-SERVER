"""Azure Vision LLM Implementation"""

import base64
import os
from pathlib import Path
from typing import Union, Optional
from openai import AzureOpenAI
from PIL import Image
import io

from .base_vision_llm import BaseVisionLLM, ChatResponse


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM Implementation

    Supports Azure OpenAI Vision API (GPT-4o, GPT-4-Vision-Preview) with
    proper error handling, input validation, and automatic image preprocessing.

    Attributes:
        api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        deployment_name: Azure deployment name (e.g., "gpt-4o")
        max_image_size: Maximum image dimension in pixels (default: 2048)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """

    DEFAULT_MAX_IMAGE_SIZE = 2048
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Initialize Azure Vision LLM client.

        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version (e.g., "2024-02-15-preview")
            deployment_name: Azure deployment name
            max_image_size: Maximum image dimension in pixels (default: 2048)
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
        self.max_image_size = max_image_size
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        trace: Optional[object] = None
    ) -> ChatResponse:
        """Send text prompt with image to Azure Vision LLM.

        Args:
            text: Text prompt/question about the image
            image: Image input, either:
                - str: File path to image
                - bytes: Raw image bytes
            trace: Optional trace context for logging

        Returns:
            ChatResponse containing the model's text response

        Raises:
            ValueError: If input validation fails
            RuntimeError: If API request fails
        """
        if not text or not text.strip():
            raise ValueError("Text prompt cannot be empty")

        if trace:
            trace.log("azure_vision_llm", f"Processing image with prompt: {text[:50]}...")

        # Process image to base64
        try:
            image_base64 = self._process_image(image, trace)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}") from e

        # Build messages with vision content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Call Azure OpenAI API
        try:
            kwargs = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": self.temperature
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            if trace:
                trace.log("azure_vision_llm", f"Calling Azure OpenAI with deployment: {self.deployment_name}")

            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None

            if trace:
                trace.log("azure_vision_llm", f"Response received: {len(content)} chars")

            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage
            )

        except Exception as e:
            error_msg = f"Azure Vision LLM API request failed: {str(e)}"
            if trace:
                trace.log("azure_vision_llm", f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e

    def _process_image(self, image: Union[str, bytes], trace: Optional[object] = None) -> str:
        """Process image to base64 string with optional resizing.

        Args:
            image: Image file path or bytes
            trace: Optional trace context

        Returns:
            Base64 encoded image string

        Raises:
            ValueError: If image format is invalid or file not found
            RuntimeError: If image processing fails
        """
        try:
            # Load image
            if isinstance(image, str):
                # File path
                image_path = Path(image)
                if not image_path.exists():
                    raise ValueError(f"Image file not found: {image}")

                # Check file extension
                if image_path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                    raise ValueError(
                        f"Unsupported image format: {image_path.suffix}. "
                        f"Supported formats: {', '.join(self.SUPPORTED_IMAGE_FORMATS)}"
                    )

                img = Image.open(image_path)
                if trace:
                    trace.log("azure_vision_llm", f"Loaded image from path: {image_path.name}")

            elif isinstance(image, bytes):
                # Raw bytes
                img = Image.open(io.BytesIO(image))
                if trace:
                    trace.log("azure_vision_llm", f"Loaded image from bytes: {len(image)} bytes")

            else:
                raise ValueError(f"Image must be str (path) or bytes, got {type(image).__name__}")

            # Resize if needed
            original_size = img.size
            if max(img.size) > self.max_image_size:
                # Calculate new size maintaining aspect ratio
                ratio = self.max_image_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                if trace:
                    trace.log(
                        "azure_vision_llm",
                        f"Resized image from {original_size} to {img.size}"
                    )

            # Convert to RGB if needed (handle RGBA, grayscale, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            if trace:
                trace.log("azure_vision_llm", f"Encoded image to base64: {len(image_base64)} chars")

            return image_base64

        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}") from e
