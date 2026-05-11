from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list) -> str:
        """向 LLM 发送聊天消息并返回响应"""
        pass