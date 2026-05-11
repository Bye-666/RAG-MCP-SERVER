from typing import List, Dict, Any, Optional
import os
import httpx
from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama 本地 LLM 实现

    支持本地 HTTP 端点（默认 base_url + model）。
    处理连接失败/超时场景，提供可读的错误信息。

    属性:
        base_url: Ollama 服务器地址（默认: http://localhost:11434）
        model: 使用的模型名称
        timeout: 请求超时时间（秒）
        temperature: 采样温度（0.0-1.0）
        max_tokens: 生成的最大 token 数
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # 秒

    def __init__(self, base_url: Optional[str] = None, model: str = "llama2",
                 timeout: Optional[int] = None, temperature: float = 0.7,
                 max_tokens: int = 512, **kwargs):
        """初始化 Ollama LLM 客户端。

        参数:
            base_url: Ollama 服务器 URL（默认从环境变量 OLLAMA_BASE_URL 或默认 localhost）
            model: 模型名称（默认: llama2）
            timeout: 请求超时时间（秒）（默认: 60）
            temperature: 采样温度（0.0-1.0，默认: 0.7）
            max_tokens: 生成的最大 token 数（默认: 512）
            **kwargs: 其他参数（为接口一致性而接受但被忽略）
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 使用 chat API 而不是 generate API 以获得更好的兼容性
        self._endpoint = "/api/chat"
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def chat(self, messages: list) -> str:
        """向 Ollama 发送聊天消息并返回响应。

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
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }

            response = self._client.post(
                self._endpoint,
                json=payload
            )

            # 对 HTTP 错误抛出异常并提供可读消息
            response.raise_for_status()

            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                raise RuntimeError(f"意外的 Ollama 响应格式: {result}")

            return result["message"]["content"]

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 Ollama 服务器 {self.base_url}。"
                f"请确保 Ollama 正在运行且可访问。"
                f"您可以设置 OLLAMA_BASE_URL 环境变量来自定义地址。"
                f"详情: {str(e)}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"请求 Ollama 服务器超时，超过 {self.timeout} 秒。"
                f"模型可能正在加载或查询较复杂。"
                f"详情: {str(e)}"
            ) from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_body = e.response.text[:500]  # 限制错误体大小
            raise RuntimeError(
                f"Ollama API 返回状态 {status_code}: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama API 请求失败: {str(e)}") from e

    def close(self):
        """关闭 HTTP 客户端。"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
