from openai import AzureOpenAI
from .base_llm import BaseLLM


class AzureLLM(BaseLLM):
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, deployment_name: str, **kwargs):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

    def chat(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API request failed: {str(e)}") from e