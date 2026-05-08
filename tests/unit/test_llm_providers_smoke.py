"""Smoke tests for OpenAI-compatible LLM providers"""

import pytest
from unittest.mock import patch, MagicMock
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM


class TestOpenAILLM:
    """Test cases for OpenAI LLM provider"""

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_initialization_success(self, mock_openai):
        """Test successful initialization"""
        llm = OpenAILLM(api_key="test-key", model="gpt-4o")
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.7
        mock_openai.assert_called_once_with(api_key="test-key")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="API key is required"):
            OpenAILLM(api_key="")

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_success(self, mock_openai):
        """Test successful chat request"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello from OpenAI"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key")
        response = llm.chat([{"role": "user", "content": "Hello"}])

        assert response == "Hello from OpenAI"
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_empty_messages(self, mock_openai):
        """Test that empty messages list raises ValueError"""
        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(ValueError, match="must be a non-empty list"):
            llm.chat([])

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_invalid_message_type(self, mock_openai):
        """Test that non-dict message raises ValueError"""
        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(ValueError, match="must be a dict"):
            llm.chat(["not a dict"])

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_missing_role_field(self, mock_openai):
        """Test that missing 'role' field raises ValueError"""
        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(ValueError, match="missing required 'role' field"):
            llm.chat([{"content": "Hello"}])

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_missing_content_field(self, mock_openai):
        """Test that missing 'content' field raises ValueError"""
        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(ValueError, match="missing required 'content' field"):
            llm.chat([{"role": "user"}])

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_chat_api_failure(self, mock_openai):
        """Test that API failure raises RuntimeError"""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(RuntimeError, match="OpenAI API request failed"):
            llm.chat([{"role": "user", "content": "Hello"}])


class TestAzureLLM:
    """Test cases for Azure OpenAI LLM provider"""

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_initialization_success(self, mock_azure):
        """Test successful initialization"""
        llm = AzureLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-07-01-preview",
            deployment_name="test-deployment"
        )
        assert llm.deployment_name == "test-deployment"
        assert llm.temperature == 0.7
        mock_azure.assert_called_once()

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="API key is required"):
            AzureLLM(
                api_key="",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2023-07-01-preview",
                deployment_name="test-deployment"
            )

    def test_initialization_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureLLM(
                api_key="test-key",
                azure_endpoint="",
                api_version="2023-07-01-preview",
                deployment_name="test-deployment"
            )

    def test_initialization_missing_api_version(self):
        """Test that missing API version raises ValueError"""
        with pytest.raises(ValueError, match="API version is required"):
            AzureLLM(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                api_version="",
                deployment_name="test-deployment"
            )

    def test_initialization_missing_deployment_name(self):
        """Test that missing deployment name raises ValueError"""
        with pytest.raises(ValueError, match="deployment name is required"):
            AzureLLM(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2023-07-01-preview",
                deployment_name=""
            )

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_chat_success(self, mock_azure):
        """Test successful chat request"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello from Azure"
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        llm = AzureLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-07-01-preview",
            deployment_name="test-deployment"
        )
        response = llm.chat([{"role": "user", "content": "Hello"}])

        assert response == "Hello from Azure"
        mock_azure.return_value.chat.completions.create.assert_called_once()

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_chat_empty_messages(self, mock_azure):
        """Test that empty messages list raises ValueError"""
        llm = AzureLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-07-01-preview",
            deployment_name="test-deployment"
        )
        with pytest.raises(ValueError, match="must be a non-empty list"):
            llm.chat([])

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_chat_api_failure(self, mock_azure):
        """Test that API failure raises RuntimeError"""
        mock_azure.return_value.chat.completions.create.side_effect = Exception("API Error")

        llm = AzureLLM(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-07-01-preview",
            deployment_name="test-deployment"
        )
        with pytest.raises(RuntimeError, match="Azure OpenAI API request failed"):
            llm.chat([{"role": "user", "content": "Hello"}])


class TestDeepSeekLLM:
    """Test cases for DeepSeek LLM provider"""

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_initialization_success(self, mock_openai):
        """Test successful initialization"""
        llm = DeepSeekLLM(api_key="test-key", model="deepseek-chat")
        assert llm.model == "deepseek-chat"
        assert llm.temperature == 0.7
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com"
        )

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="API key is required"):
            DeepSeekLLM(api_key="")

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_chat_success(self, mock_openai):
        """Test successful chat request"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello from DeepSeek"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = DeepSeekLLM(api_key="test-key")
        response = llm.chat([{"role": "user", "content": "Hello"}])

        assert response == "Hello from DeepSeek"
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_chat_empty_messages(self, mock_openai):
        """Test that empty messages list raises ValueError"""
        llm = DeepSeekLLM(api_key="test-key")
        with pytest.raises(ValueError, match="must be a non-empty list"):
            llm.chat([])

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_chat_invalid_message_structure(self, mock_openai):
        """Test that invalid message structure raises ValueError"""
        llm = DeepSeekLLM(api_key="test-key")
        with pytest.raises(ValueError, match="missing required 'role' field"):
            llm.chat([{"content": "Hello"}])

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_chat_api_failure(self, mock_openai):
        """Test that API failure raises RuntimeError"""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        llm = DeepSeekLLM(api_key="test-key")
        with pytest.raises(RuntimeError, match="DeepSeek API request failed"):
            llm.chat([{"role": "user", "content": "Hello"}])


class TestLLMFactoryIntegration:
    """Test LLM Factory integration with all providers"""

    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_factory_creates_openai(self, mock_openai):
        """Test factory creates OpenAI LLM correctly"""
        from src.libs.llm.llm_factory import LLMFactory

        settings = {
            "llm": {
                "provider": "openai",
                "api_key": "test-key",
                "model": "gpt-4o"
            }
        }
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_factory_creates_azure(self, mock_azure):
        """Test factory creates Azure LLM correctly"""
        from src.libs.llm.llm_factory import LLMFactory

        settings = {
            "llm": {
                "provider": "azure",
                "api_key": "test-key",
                "azure_endpoint": "https://test.openai.azure.com",
                "api_version": "2023-07-01-preview",
                "deployment_name": "test-deployment"
            }
        }
        llm = LLMFactory.create(settings)
        assert isinstance(llm, AzureLLM)
        assert llm.deployment_name == "test-deployment"

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_factory_creates_deepseek(self, mock_openai):
        """Test factory creates DeepSeek LLM correctly"""
        from src.libs.llm.llm_factory import LLMFactory

        settings = {
            "llm": {
                "provider": "deepseek",
                "api_key": "test-key",
                "model": "deepseek-chat"
            }
        }
        llm = LLMFactory.create(settings)
        assert isinstance(llm, DeepSeekLLM)
        assert llm.model == "deepseek-chat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
