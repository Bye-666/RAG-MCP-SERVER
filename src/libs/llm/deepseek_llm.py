from typing import Optional
from openai import OpenAI
from .base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 实现

    支持 DeepSeek API（兼容 OpenAI），具有适当的错误处理和输入验证。

    属性:
        api_key: DeepSeek API 密钥
        base_url: DeepSeek API 基础 URL
        model: 使用的模型名称（默认: deepseek-chat）
        temperature: 采样温度（0.0-2.0）
        max_tokens: 生成的最大 token 数
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7,
                 max_tokens: Optional[int] = None, **kwargs):
        """初始化 DeepSeek LLM 客户端。

        参数:
            api_key: DeepSeek API 密钥
            base_url: DeepSeek API 基础 URL（默认: https://api.deepseek.com）
            model: 模型名称（默认: deepseek-chat）
            temperature: 采样温度（0.0-2.0，默认: 0.7）
            max_tokens: 生成的最大 token 数（可选）
            **kwargs: 其他参数（为接口一致性而接受但被忽略）
        """
        if not api_key:
            raise ValueError("需要 DeepSeek API 密钥")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list) -> str:
        """向 DeepSeek 发送聊天消息并返回响应。

        参数:
            messages: OpenAI 格式的聊天消息列表 [{"role": "user", "content": "..."}]

        返回:
            模型的文本响应

        异常:
            RuntimeError: 如果 API 请求失败，包含详细错误信息
            ValueError: 如果消息输入无效
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages 必须是非空列表")

        # 验证消息结构
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"message[{i}] 必须是字典，得到 {type(msg).__name__}")
            if "role" not in msg:
                raise ValueError(f"message[{i}] 缺少必需的 'role' 字段")
            if "content" not in msg:
                raise ValueError(f"message[{i}] 缺少必需的 'content' 字段")

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 请求失败: {str(e)}") from e