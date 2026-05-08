from openai import OpenAI
from .base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {str(e)}") from e