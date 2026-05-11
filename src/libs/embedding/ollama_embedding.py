import os
from typing import List
from .base_embedding import BaseEmbedding
import httpx

class OllamaEmbedding(BaseEmbedding):
    """Ollama 嵌入实现

    使用 Ollama 的 /api/embeddings 端点生成嵌入向量。
    使用与 LLM 实现相同的配置风格。

    属性:
        base_url: Ollama 服务器地址
        model: 使用的模型名称
        timeout: 请求超时时间（秒）
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # 秒

    def __init__(self, base_url: str = None, model: str = "nomic-embed-text", timeout: int = None):
        """初始化 Ollama 嵌入客户端。

        参数:
            base_url: Ollama 服务器 URL（默认从环境变量 OLLAMA_BASE_URL 或默认 localhost）
            model: 使用的模型名称（默认: nomic-embed-text）
            timeout: 请求超时时间（秒）（默认: 60）
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """为文本列表生成嵌入向量。

        参数:
            texts: 输入文本字符串列表

        返回:
            嵌入向量列表（每个嵌入是浮点数列表）

        异常:
            RuntimeError: 如果 API 请求失败，包含详细错误信息
            ValueError: 如果输入无效
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("texts 必须是非空列表")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"text[{i}] 必须是字符串，得到 {type(text).__name__}")

        try:
            # 构建请求载荷
            payload = {
                "model": self.model,
                "input": texts
            }

            response = self._client.post(
                "/api/embeddings",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            if "embeddings" not in result:
                raise RuntimeError(f"意外的 Ollama 响应格式: {result}")

            # 验证形状
            if len(result["embeddings"]) != len(texts):
                raise RuntimeError(
                    f"嵌入数量不匹配: {len(result['embeddings'])} 个嵌入对应 {len(texts)} 个文本"
                )

            return result["embeddings"]

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
                f"请求可能过大。"
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