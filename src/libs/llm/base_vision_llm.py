from abc import ABC, abstractmethod
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Vision LLM 聊天响应"""
    content: str
    model: Optional[str] = None
    usage: Optional[dict] = None


class BaseVisionLLM(ABC):
    """Vision LLM 后端的抽象接口

    Vision LLM 支持多模态输入（文本 + 图像），用于图像描述、
    视觉问答和文档理解等任务。
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        trace: Optional[object] = None
    ) -> ChatResponse:
        """向 Vision LLM 发送文本提示和图像并返回响应。

        参数:
            text: 关于图像的文本提示/问题
            image: 图像输入，可以是:
                - str: 图像文件路径（例如 "/path/to/image.jpg"）
                - bytes: 原始图像字节（将在内部进行 base64 编码）
            trace: 可选的跟踪上下文用于日志记录（TraceContext 实例）

        返回:
            包含模型文本响应的 ChatResponse

        异常:
            ValueError: 如果输入验证失败（无效的图像格式、空文本等）
            RuntimeError: 如果 API 请求失败（网络错误、模型错误等）

        注意:
            - 实现应处理图像预处理（调整大小、格式转换）
            - 实现应在 API 调用前验证图像大小/格式
            - 实现应提供包含提供商名称的清晰错误消息
        """
        pass
