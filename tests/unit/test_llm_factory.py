import pytest
from unittest.mock import patch, MagicMock
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    def __init__(self, **kwargs):
        pass

    def chat(self, messages):
        return "Fake response"

class TestLLMFactory:
    def setup_method(self):
        # Clear registered providers before each test
        LLMFactory._providers = {}

    def test_register_provider(self):
        LLMFactory.register_provider("fake", FakeLLM)
        assert "fake" in LLMFactory._providers
        assert LLMFactory._providers["fake"] == FakeLLM

    def test_create_valid_provider(self):
        LLMFactory.register_provider("fake", FakeLLM)
        settings = {"llm": {"provider": "fake"}}
        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)

    def test_create_invalid_provider(self):
        settings = {"llm": {"provider": "invalid"}}
        with pytest.raises(ValueError):
            LLMFactory.create(settings)

class TestLLMProviders:
    @patch('src.libs.llm.openai_llm.OpenAI')
    def test_openai_smoke(self, mock_openai):
        from src.libs.llm.openai_llm import OpenAILLM

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create instance
        llm = OpenAILLM(api_key="test")
        response = llm.chat([{"role": "user", "content": "Hello"}])

        # Assertions
        assert response == "OpenAI response"
        mock_openai.assert_called_once_with(api_key="test")
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

    @patch('src.libs.llm.azure_llm.AzureOpenAI')
    def test_azure_smoke(self, mock_azure):
        from src.libs.llm.azure_llm import AzureLLM

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Azure response"
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        # Create instance
        llm = AzureLLM(
            api_key="test",
            azure_endpoint="test-endpoint",
            api_version="2023-07-01-preview",
            deployment_name="test-deployment"
        )
        response = llm.chat([{"role": "user", "content": "Hello"}])

        # Assertions
        assert response == "Azure response"
        mock_azure.assert_called_once_with(
            api_key="test",
            api_version="2023-07-01-preview",
            azure_endpoint="test-endpoint"
        )
        mock_azure.return_value.chat.completions.create.assert_called_once_with(
            model="test-deployment",
            messages=[{"role": "user", "content": "Hello"}]
        )

    @patch('src.libs.llm.deepseek_llm.OpenAI')
    def test_deepseek_smoke(self, mock_openai):
        from src.libs.llm.deepseek_llm import DeepSeekLLM

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "DeepSeek response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create instance
        llm = DeepSeekLLM(api_key="test")
        response = llm.chat([{"role": "user", "content": "Hello"}])

        # Assertions
        assert response == "DeepSeek response"
        mock_openai.assert_called_once_with(
            api_key="test",
            base_url="https://api.deepseek.com"
        )
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )