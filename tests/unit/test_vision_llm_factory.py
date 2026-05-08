"""Unit tests for Vision LLM Factory"""

import pytest
from unittest.mock import Mock
from src.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from src.libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM for testing"""

    def __init__(self, provider: str = "fake", model: str = "fake-vision", **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs

    def chat_with_image(self, text: str, image, trace=None) -> ChatResponse:
        return ChatResponse(
            content=f"Fake response for: {text}",
            model=self.model
        )


class TestBaseVisionLLM:
    """Test cases for BaseVisionLLM interface"""

    def test_chat_response_dataclass(self):
        """Test ChatResponse dataclass"""
        response = ChatResponse(content="Test response", model="gpt-4o")
        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.usage is None

    def test_chat_response_with_usage(self):
        """Test ChatResponse with usage information"""
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        response = ChatResponse(content="Test", model="gpt-4o", usage=usage)
        assert response.usage == usage

    def test_fake_vision_llm_implementation(self):
        """Test that FakeVisionLLM implements BaseVisionLLM correctly"""
        vision_llm = FakeVisionLLM(provider="fake", model="test-model")
        assert isinstance(vision_llm, BaseVisionLLM)
        assert vision_llm.provider == "fake"
        assert vision_llm.model == "test-model"

    def test_fake_vision_llm_chat_with_image(self):
        """Test FakeVisionLLM chat_with_image method"""
        vision_llm = FakeVisionLLM()
        response = vision_llm.chat_with_image("What's in this image?", "/path/to/image.jpg")

        assert isinstance(response, ChatResponse)
        assert "What's in this image?" in response.content
        assert response.model == "fake-vision"


class TestVisionLLMFactory:
    """Test cases for Vision LLM Factory"""

    def setup_method(self):
        """Clear registered vision providers before each test"""
        LLMFactory._vision_providers = {}

    def test_register_vision_provider(self):
        """Test registering a vision provider"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        assert "fake" in LLMFactory._vision_providers
        assert LLMFactory._vision_providers["fake"] == FakeVisionLLM

    def test_create_vision_llm_success(self):
        """Test creating a vision LLM instance"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        settings = {
            "vision_llm": {
                "provider": "fake",
                "model": "test-vision-model"
            }
        }

        vision_llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(vision_llm, FakeVisionLLM)
        assert vision_llm.model == "test-vision-model"

    def test_create_vision_llm_missing_settings_key(self):
        """Test that missing 'vision_llm' key raises KeyError"""
        settings = {"llm": {"provider": "openai"}}

        with pytest.raises(KeyError, match="must contain 'vision_llm' key"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_missing_provider(self):
        """Test that missing provider raises ValueError"""
        settings = {"vision_llm": {"model": "gpt-4o"}}

        with pytest.raises(ValueError, match="provider not specified"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_unsupported_provider(self):
        """Test that unsupported provider raises ValueError"""
        settings = {
            "vision_llm": {
                "provider": "unsupported",
                "model": "test"
            }
        }

        with pytest.raises(ValueError, match="Unsupported Vision LLM provider: unsupported"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_error_message_includes_available_providers(self):
        """Test that error message includes available providers"""
        LLMFactory.register_vision_provider("azure", FakeVisionLLM)
        LLMFactory.register_vision_provider("openai", FakeVisionLLM)

        settings = {
            "vision_llm": {
                "provider": "invalid"
            }
        }

        with pytest.raises(ValueError, match="Available providers:"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_with_multiple_providers(self):
        """Test creating vision LLMs with multiple registered providers"""
        LLMFactory.register_vision_provider("provider1", FakeVisionLLM)
        LLMFactory.register_vision_provider("provider2", FakeVisionLLM)

        settings1 = {"vision_llm": {"provider": "provider1", "model": "model1"}}
        settings2 = {"vision_llm": {"provider": "provider2", "model": "model2"}}

        vision_llm1 = LLMFactory.create_vision_llm(settings1)
        vision_llm2 = LLMFactory.create_vision_llm(settings2)

        assert vision_llm1.model == "model1"
        assert vision_llm2.model == "model2"

    def test_create_vision_llm_passes_all_kwargs(self):
        """Test that factory passes all kwargs to vision LLM constructor"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        settings = {
            "vision_llm": {
                "provider": "fake",
                "model": "test-model",
                "api_key": "test-key",
                "temperature": 0.5,
                "max_tokens": 1000
            }
        }

        vision_llm = LLMFactory.create_vision_llm(settings)

        assert vision_llm.model == "test-model"
        assert vision_llm.kwargs["api_key"] == "test-key"
        assert vision_llm.kwargs["temperature"] == 0.5
        assert vision_llm.kwargs["max_tokens"] == 1000

    def test_vision_llm_factory_independent_from_llm_factory(self):
        """Test that vision LLM providers are independent from regular LLM providers"""
        # Regular LLM providers should still work
        assert "openai" in LLMFactory._providers
        assert "azure" in LLMFactory._providers

        # Vision LLM providers are separate
        LLMFactory.register_vision_provider("azure_vision", FakeVisionLLM)
        assert "azure_vision" in LLMFactory._vision_providers
        assert "azure_vision" not in LLMFactory._providers


class TestVisionLLMIntegration:
    """Integration tests for Vision LLM workflow"""

    def setup_method(self):
        """Setup for integration tests"""
        LLMFactory._vision_providers = {}

    def test_end_to_end_vision_llm_workflow(self):
        """Test complete workflow: register -> create -> use"""
        # Register provider
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        # Create instance
        settings = {
            "vision_llm": {
                "provider": "fake",
                "model": "vision-model-v1"
            }
        }
        vision_llm = LLMFactory.create_vision_llm(settings)

        # Use instance
        response = vision_llm.chat_with_image(
            text="Describe this image",
            image="/path/to/test.jpg"
        )

        assert isinstance(response, ChatResponse)
        assert "Describe this image" in response.content
        assert response.model == "vision-model-v1"

    def test_vision_llm_with_trace_context(self):
        """Test vision LLM with trace context"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        settings = {"vision_llm": {"provider": "fake"}}
        vision_llm = LLMFactory.create_vision_llm(settings)

        # Mock trace context
        trace = Mock()
        response = vision_llm.chat_with_image(
            text="What's in this image?",
            image=b"fake_image_bytes",
            trace=trace
        )

        assert isinstance(response, ChatResponse)

    def test_vision_llm_with_different_image_inputs(self):
        """Test vision LLM accepts both file path and bytes"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        settings = {"vision_llm": {"provider": "fake"}}
        vision_llm = LLMFactory.create_vision_llm(settings)

        # Test with file path
        response1 = vision_llm.chat_with_image("Test", "/path/to/image.jpg")
        assert isinstance(response1, ChatResponse)

        # Test with bytes
        response2 = vision_llm.chat_with_image("Test", b"image_bytes")
        assert isinstance(response2, ChatResponse)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
