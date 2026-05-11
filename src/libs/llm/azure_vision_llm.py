"""Azure Vision LLM 实现"""

import base64
import os
from pathlib import Path
from typing import Union, Optional
from openai import AzureOpenAI
from PIL import Image
import io

from .base_vision_llm import BaseVisionLLM, ChatResponse


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM 实现

    支持 Azure OpenAI Vision API（GPT-4o、GPT-4-Vision-Preview），
    具有适当的错误处理、输入验证和自动图像预处理。

    属性:
        api_key: Azure OpenAI API 密钥
        azure_endpoint: Azure OpenAI 端点 URL
        api_version: Azure OpenAI API 版本
        deployment_name: Azure 部署名称（例如 "gpt-4o"）
        max_image_size: 最大图像尺寸（像素）（默认: 2048）
        temperature: 采样温度（0.0-2.0）
        max_tokens: 生成的最大 token 数
    """

    DEFAULT_MAX_IMAGE_SIZE = 2048
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """初始化 Azure Vision LLM 客户端。

        参数:
            api_key: Azure OpenAI API 密钥
            azure_endpoint: Azure OpenAI 端点 URL
            api_version: Azure OpenAI API 版本（例如 "2024-02-15-preview"）
            deployment_name: Azure 部署名称
            max_image_size: 最大图像尺寸（像素）（默认: 2048）
            temperature: 采样温度（0.0-2.0，默认: 0.7）
            max_tokens: 生成的最大 token 数（可选）
            **kwargs: 其他参数（为接口一致性而接受但被忽略）
        """
        if not api_key:
            raise ValueError("需要 Azure OpenAI API 密钥")
        if not azure_endpoint:
            raise ValueError("需要 Azure OpenAI 端点")
        if not api_version:
            raise ValueError("需要 Azure OpenAI API 版本")
        if not deployment_name:
            raise ValueError("需要 Azure 部署名称")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name
        self.max_image_size = max_image_size
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        trace: Optional[object] = None
    ) -> ChatResponse:
        """向 Azure Vision LLM 发送文本提示和图像。

        参数:
            text: 关于图像的文本提示/问题
            image: 图像输入，可以是:
                - str: 图像文件路径
                - bytes: 原始图像字节
            trace: 可选的跟踪上下文用于日志记录

        返回:
            包含模型文本响应的 ChatResponse

        异常:
            ValueError: 如果输入验证失败
            RuntimeError: 如果 API 请求失败
        """
        if not text or not text.strip():
            raise ValueError("文本提示不能为空")

        if trace:
            trace.log("azure_vision_llm", f"处理图像，提示: {text[:50]}...")

        # 将图像处理为 base64
        try:
            image_base64 = self._process_image(image, trace)
        except Exception as e:
            raise ValueError(f"图像处理失败: {str(e)}") from e

        # 构建包含视觉内容的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # 调用 Azure OpenAI API
        try:
            kwargs = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": self.temperature
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            if trace:
                trace.log("azure_vision_llm", f"调用 Azure OpenAI，部署: {self.deployment_name}")

            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None

            if trace:
                trace.log("azure_vision_llm", f"收到响应: {len(content)} 个字符")

            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage
            )

        except Exception as e:
            error_msg = f"Azure Vision LLM API 请求失败: {str(e)}"
            if trace:
                trace.log("azure_vision_llm", f"错误: {error_msg}")
            raise RuntimeError(error_msg) from e

    def _process_image(self, image: Union[str, bytes], trace: Optional[object] = None) -> str:
        """将图像处理为 base64 字符串，可选调整大小。

        参数:
            image: 图像文件路径或字节
            trace: 可选的跟踪上下文

        返回:
            Base64 编码的图像字符串

        异常:
            ValueError: 如果图像格式无效或文件未找到
            RuntimeError: 如果图像处理失败
        """
        try:
            # 加载图像
            if isinstance(image, str):
                # 文件路径
                image_path = Path(image)
                if not image_path.exists():
                    raise ValueError(f"图像文件未找到: {image}")

                # 检查文件扩展名
                if image_path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                    raise ValueError(
                        f"不支持的图像格式: {image_path.suffix}。"
                        f"支持的格式: {', '.join(self.SUPPORTED_IMAGE_FORMATS)}"
                    )

                img = Image.open(image_path)
                if trace:
                    trace.log("azure_vision_llm", f"从路径加载图像: {image_path.name}")

            elif isinstance(image, bytes):
                # 原始字节
                img = Image.open(io.BytesIO(image))
                if trace:
                    trace.log("azure_vision_llm", f"从字节加载图像: {len(image)} 字节")

            else:
                raise ValueError(f"图像必须是 str（路径）或 bytes，得到 {type(image).__name__}")

            # 如果需要则调整大小
            original_size = img.size
            if max(img.size) > self.max_image_size:
                # 计算保持宽高比的新尺寸
                ratio = self.max_image_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                if trace:
                    trace.log(
                        "azure_vision_llm",
                        f"调整图像大小从 {original_size} 到 {img.size}"
                    )

            # 如果需要则转换为 RGB（处理 RGBA、灰度等）
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # 编码为 base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            if trace:
                trace.log("azure_vision_llm", f"将图像编码为 base64: {len(image_base64)} 个字符")

            return image_base64

        except ValueError:
            raise  # 重新抛出验证错误
        except Exception as e:
            raise RuntimeError(f"图像处理失败: {str(e)}") from e
