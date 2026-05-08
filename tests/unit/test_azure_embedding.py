"""Unit tests for AzureEmbedding implementation.

Uses mocking to test without requiring real Azure OpenAI API calls.
Covers:
- Normal embedding generation
- Batch processing
- Empty text handling
- API error handling
- Azure-specific configuration (endpoint, deployment)
"""

import pytest
from unittest.mock import Mock, patch
from src.libs.embedding.azure_embedding import AzureEmbedding


class TestAzureEmbedding:
    """Test cases for AzureEmbedding class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI'):
            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            assert embedding.deployment_name == "test-deployment"
            assert embedding.batch_size == 100

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI'):
            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment",
                api_version="2024-03-01",
                batch_size=50
            )
            assert embedding.deployment_name == "test-deployment"
            assert embedding.batch_size == 50

    def test_azure_client_initialization(self):
        """Test that AzureOpenAI client is initialized with correct parameters."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment",
                api_version="2024-02-01"
            )

            mock_azure.assert_called_once_with(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2024-02-01",
                azure_deployment="test-deployment"
            )

    def test_embed_single_text(self):
        """Test embedding a single text."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            result = embedding.embed(["Hello world"])

            assert len(result) == 1
            assert result[0] == [0.1, 0.2, 0.3]
            mock_client.embeddings.create.assert_called_once()

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            # Mock response with multiple embeddings
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6]),
                Mock(embedding=[0.7, 0.8, 0.9])
            ]
            mock_client.embeddings.create.return_value = mock_response

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            result = embedding.embed(["Text 1", "Text 2", "Text 3"])

            assert len(result) == 3
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            assert result[2] == [0.7, 0.8, 0.9]

    def test_embed_empty_text_handling(self):
        """Test that empty texts are handled with zero vectors."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            # Mock response for two non-empty texts
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.embeddings.create.return_value = mock_response

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            result = embedding.embed(["Hello", "", "World"])

            assert len(result) == 3
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.0] * 1536  # Zero vector for empty text
            assert result[2] == [0.4, 0.5, 0.6]
            assert len(result[1]) == 1536

    def test_embed_all_empty_texts(self):
        """Test embedding when all texts are empty."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            result = embedding.embed(["", "  ", "\t"])

            assert len(result) == 3
            assert all(vec == [0.0] * 1536 for vec in result)
            # API should not be called for all-empty batch
            mock_client.embeddings.create.assert_not_called()

    def test_embed_batch_processing(self):
        """Test that large inputs are processed in batches."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            # Mock response
            def create_mock_response(input_texts):
                return Mock(data=[Mock(embedding=[0.1] * 3) for _ in input_texts])

            mock_client.embeddings.create.side_effect = lambda **kwargs: create_mock_response(kwargs['input'])

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment",
                batch_size=10
            )
            texts = [f"Text {i}" for i in range(25)]
            result = embedding.embed(texts)

            assert len(result) == 25
            # Should be called 3 times: 10 + 10 + 5
            assert mock_client.embeddings.create.call_count == 3

    def test_embed_api_error_handling(self):
        """Test that API errors are handled gracefully with zero vectors."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            # Mock API error
            mock_client.embeddings.create.side_effect = Exception("API Error")

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            result = embedding.embed(["Test text"])

            assert len(result) == 1
            assert result[0] == [0.0] * 1536

    def test_embed_uses_deployment_name_as_model(self):
        """Test that deployment_name is used as the model parameter."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="my-embedding-deployment"
            )
            embedding.embed(["Test text"])

            # In Azure, the model parameter should be the deployment name
            mock_client.embeddings.create.assert_called_once_with(
                input=["Test text"],
                model="my-embedding-deployment"
            )

    def test_embed_correct_api_parameters(self):
        """Test that correct parameters are passed to Azure OpenAI API."""
        with patch('src.libs.embedding.azure_embedding.AzureOpenAI') as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response

            embedding = AzureEmbedding(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
                deployment_name="test-deployment"
            )
            embedding.embed(["Test text"])

            mock_client.embeddings.create.assert_called_once_with(
                input=["Test text"],
                model="test-deployment"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
