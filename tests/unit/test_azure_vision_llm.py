"""Unit tests for Azure Vision LLM"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import io
from PIL import Image

from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.base_vision_llm import ChatResponse


class TestAzureVisionLLMInitialization:
    """Test cases for Azure Vision LLM initialization"""

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_initialization_success(self, mock_azure):
        """Test successful initialization"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )
        assert llm.deployment_name == "gpt-4o"
        assert llm.max_image_size == 2048
        assert llm.temperature == 0.7
        mock_azure.assert_called_once()

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_initialization_with_custom_params(self, mock_azure):
        """Test initialization with custom parameters"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o",
            max_image_size=1024,
            temperature=0.5,
            max_tokens=500
        )
        assert llm.max_image_size == 1024
        assert llm.temperature == 0.5
        assert llm.max_tokens == 500

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="API key is required"):
            AzureVisionLLM(
                api_key="",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                deployment_name="gpt-4o"
            )

    def test_initialization_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureVisionLLM(
                api_key="test-key",
                azure_endpoint="",
                api_version="2024-02-15-preview",
                deployment_name="gpt-4o"
            )

    def test_initialization_missing_api_version(self):
        """Test that missing API version raises ValueError"""
        with pytest.raises(ValueError, match="API version is required"):
            AzureVisionLLM(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                api_version="",
                deployment_name="gpt-4o"
            )

    def test_initialization_missing_deployment_name(self):
        """Test that missing deployment name raises ValueError"""
        with pytest.raises(ValueError, match="deployment name is required"):
            AzureVisionLLM(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                deployment_name=""
            )


class TestAzureVisionLLMChatWithImage:
    """Test cases for chat_with_image method"""

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_chat_with_image_success(self, mock_image_class, mock_azure):
        """Test successful chat with image"""
        # Mock image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.mode = 'RGB'
        mock_image_class.open.return_value = mock_img

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test image"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        # Create a fake image bytes
        response = llm.chat_with_image("What's in this image?", b"fake_image_bytes")

        assert isinstance(response, ChatResponse)
        assert response.content == "This is a test image"
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 150

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_chat_with_image_empty_text(self, mock_azure):
        """Test that empty text raises ValueError"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            llm.chat_with_image("", b"fake_image")

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_chat_with_image_api_failure(self, mock_image_class, mock_azure):
        """Test that API failure raises RuntimeError"""
        # Mock image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.mode = 'RGB'
        mock_image_class.open.return_value = mock_img

        # Mock API error
        mock_azure.return_value.chat.completions.create.side_effect = Exception("API Error")

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        with pytest.raises(RuntimeError, match="Azure Vision LLM API request failed"):
            llm.chat_with_image("Test", b"fake_image")

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_chat_with_image_with_trace(self, mock_image_class, mock_azure):
        """Test chat with image with trace context"""
        # Mock image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.mode = 'RGB'
        mock_image_class.open.return_value = mock_img

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        trace = Mock()
        response = llm.chat_with_image("Test", b"fake_image", trace=trace)

        assert isinstance(response, ChatResponse)
        assert trace.log.call_count >= 3  # Multiple trace logs


class TestAzureVisionLLMImageProcessing:
    """Test cases for image processing"""

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_image_resize_when_too_large(self, mock_image_class, mock_azure):
        """Test that large images are resized"""
        # Mock large image
        mock_img = MagicMock()
        mock_img.size = (4000, 3000)  # Larger than default 2048
        mock_img.mode = 'RGB'
        mock_image_class.open.return_value = mock_img

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        response = llm.chat_with_image("Test", b"fake_image")

        # Verify resize was called
        mock_img.resize.assert_called_once()
        assert isinstance(response, ChatResponse)

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_image_no_resize_when_small(self, mock_image_class, mock_azure):
        """Test that small images are not resized"""
        # Mock small image
        mock_img = MagicMock()
        mock_img.size = (800, 600)  # Smaller than 2048
        mock_img.mode = 'RGB'
        mock_image_class.open.return_value = mock_img

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        response = llm.chat_with_image("Test", b"fake_image")

        # Verify resize was NOT called
        mock_img.resize.assert_not_called()
        assert isinstance(response, ChatResponse)

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    @patch('src.libs.llm.azure_vision_llm.Image')
    def test_image_convert_rgba_to_rgb(self, mock_image_class, mock_azure):
        """Test that RGBA images are converted to RGB"""
        # Mock RGBA image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.mode = 'RGBA'
        mock_image_class.open.return_value = mock_img

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        response = llm.chat_with_image("Test", b"fake_image")

        # Verify convert was called
        mock_img.convert.assert_called_once_with('RGB')
        assert isinstance(response, ChatResponse)

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_image_file_not_found(self, mock_azure):
        """Test that non-existent file raises ValueError"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        with pytest.raises(ValueError, match="Image file not found"):
            llm.chat_with_image("Test", "/nonexistent/image.jpg")

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_image_unsupported_format(self, mock_azure):
        """Test that unsupported format raises ValueError"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        # Create a temporary file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                llm.chat_with_image("Test", temp_path)
        finally:
            Path(temp_path).unlink()

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_image_invalid_type(self, mock_azure):
        """Test that invalid image type raises ValueError"""
        llm = AzureVisionLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4o"
        )

        with pytest.raises(ValueError, match="Image must be str .* or bytes"):
            llm.chat_with_image("Test", 12345)  # Invalid type


class TestAzureVisionLLMFactoryIntegration:
    """Test integration with LLMFactory"""

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_factory_creates_azure_vision_llm(self, mock_azure):
        """Test that factory creates Azure Vision LLM correctly"""
        from src.libs.llm.llm_factory import LLMFactory

        settings = {
            "vision_llm": {
                "provider": "azure",
                "api_key": "test-key",
                "azure_endpoint": "https://test.openai.azure.com",
                "api_version": "2024-02-15-preview",
                "deployment_name": "gpt-4o"
            }
        }

        vision_llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(vision_llm, AzureVisionLLM)
        assert vision_llm.deployment_name == "gpt-4o"

    @patch('src.libs.llm.azure_vision_llm.AzureOpenAI')
    def test_factory_passes_custom_params(self, mock_azure):
        """Test that factory passes custom parameters"""
        from src.libs.llm.llm_factory import LLMFactory

        settings = {
            "vision_llm": {
                "provider": "azure",
                "api_key": "test-key",
                "azure_endpoint": "https://test.openai.azure.com",
                "api_version": "2024-02-15-preview",
                "deployment_name": "gpt-4o",
                "max_image_size": 1024,
                "temperature": 0.5
            }
        }

        vision_llm = LLMFactory.create_vision_llm(settings)

        assert vision_llm.max_image_size == 1024
        assert vision_llm.temperature == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
