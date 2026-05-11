"""
ImageCaptioner: 为分块中的图像生成标题的转换器。

提供两种模式：
1. 启用模式：使用 Vision LLM 为图像生成标题
2. 禁用/回退模式：标记包含未处理图像的分块

当 Vision LLM 不可用或失败时优雅地回退。
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ImageCaptioner(BaseTransform):
    """
    为分块中引用的图像生成标题。

    两阶段处理：
    1. Vision LLM 模式（如果启用且可用）
    2. 回退模式（标记未处理的图像）

    优雅降级：Vision LLM 失败不会阻塞摄取。
    """

    def __init__(
        self,
        settings,
        vision_llm=None,
        prompt_path: Optional[str] = None
    ):
        """
        初始化 ImageCaptioner。

        Args:
            settings: 包含 vision_llm 配置的设置对象
            vision_llm: 可选的 Vision LLM 实例（如果为 None，将从设置创建）
            prompt_path: 可选的提示模板文件路径
        """
        self.settings = settings
        # 安全检查 ingestion.image_captioner.use_vision 配置
        self.use_vision = False
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'image_captioner'):
            self.use_vision = getattr(settings.ingestion.image_captioner, 'use_vision', False)

        # 如果启用，初始化 Vision LLM
        self.vision_llm = None
        if self.use_vision:
            if vision_llm is not None:
                self.vision_llm = vision_llm
            else:
                try:
                    self.vision_llm = LLMFactory.create_vision_llm(settings)
                except Exception as e:
                    logger.warning(f"初始化图像标题生成的 Vision LLM 失败: {e}")
                    self.use_vision = False

        # 加载提示模板
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        从文件加载提示模板。

        Args:
            prompt_path: 可选的自定义提示路径

        Returns:
            提示模板字符串
        """
        if prompt_path is None:
            prompt_path = "config/prompts/image_captioning.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"从 {prompt_path} 加载提示失败: {e}")

        # 回退提示
        return "详细描述这张图像。重点关注主要内容、对象、文本和任何重要的视觉元素。"

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        通过为图像生成标题来转换分块。

        Args:
            chunks: 要处理的分块列表
            trace: 可选的追踪上下文

        Returns:
            带有图像标题的分块列表（如果适用）
        """
        stage = None
        if trace:
            stage = trace.record_stage("image_captioner", {"use_vision": self.use_vision})

        processed_chunks = []
        images_processed = 0
        images_failed = 0

        for chunk in chunks:
            try:
                processed_chunk = self._process_chunk(chunk, trace)
                processed_chunks.append(processed_chunk)

                # 跟踪统计信息
                if "image_captions" in processed_chunk.metadata:
                    images_processed += len(processed_chunk.metadata["image_captions"])
                if processed_chunk.metadata.get("has_unprocessed_images"):
                    images_failed += 1

            except Exception as e:
                logger.error(f"处理分块 {chunk.id} 失败: {e}")
                # 出错时，保留原始分块但标记为未处理
                error_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={**chunk.metadata, "image_processing_error": str(e)},
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref
                )
                if "image_refs" in chunk.metadata:
                    error_chunk.metadata["has_unprocessed_images"] = True
                processed_chunks.append(error_chunk)

        if trace and stage:
            trace.finish_stage(stage, {
                "chunks_processed": len(processed_chunks),
                "images_captioned": images_processed,
                "images_failed": images_failed
            })

        return processed_chunks

    def _process_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """
        处理单个分块以生成图像标题。

        Args:
            chunk: 要处理的分块
            trace: 可选的追踪上下文

        Returns:
            带有标题或回退标记的处理后分块
        """
        # 检查分块是否有图像引用
        image_refs = chunk.metadata.get("image_refs", [])
        if not image_refs:
            # 没有图像，原样返回
            return chunk

        # 如果 Vision LLM 可用，尝试生成标题
        captions = {}
        unprocessed = []

        if self.use_vision and self.vision_llm:
            for image_ref in image_refs:
                caption = self._generate_caption(image_ref, chunk, trace)
                if caption:
                    captions[image_ref] = caption
                else:
                    unprocessed.append(image_ref)
        else:
            # Vision 禁用，标记所有为未处理
            unprocessed = image_refs

        # 构建更新的元数据
        updated_metadata = {**chunk.metadata}

        if captions:
            updated_metadata["image_captions"] = captions
            updated_metadata["captioned_by"] = "vision_llm"

        if unprocessed:
            updated_metadata["has_unprocessed_images"] = True
            updated_metadata["unprocessed_image_refs"] = unprocessed

        # 创建更新的分块
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=updated_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )

    def _generate_caption(
        self,
        image_ref: str,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """
        为单个图像生成标题。

        Args:
            image_ref: 图像引用 ID
            chunk: 父分块（用于上下文）
            trace: 可选的追踪上下文

        Returns:
            标题字符串，失败时返回 None
        """
        if not self.vision_llm:
            return None

        try:
            # 从元数据获取图像路径
            image_path = self._resolve_image_path(image_ref, chunk)
            if not image_path:
                logger.warning(f"无法解析 {image_ref} 的图像路径")
                return None

            # 检查图像文件是否存在
            if not Path(image_path).exists():
                logger.warning(f"图像文件未找到: {image_path}")
                return None

            # 使用 Vision LLM 生成标题
            response = self.vision_llm.chat_with_image(
                text=self.prompt_template,
                image=image_path,
                trace=trace
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"为 {image_ref} 生成标题失败: {e}")
            return None

    def _resolve_image_path(self, image_ref: str, chunk: Chunk) -> Optional[str]:
        """
        将图像引用解析为文件路径。

        Args:
            image_ref: 图像引用 ID
            chunk: 父分块

        Returns:
            图像文件路径，未找到时返回 None
        """
        # 检查分块元数据是否有包含路径信息的图像列表
        images = chunk.metadata.get("images", [])

        for img in images:
            if isinstance(img, dict) and img.get("image_id") == image_ref:
                return img.get("path")

        # 回退：从 image_ref 构造路径
        # 假设格式：data/images/{doc_hash}/{image_id}.{ext}
        # 这是一个启发式方法，可能需要根据实际存储进行调整
        logger.debug(f"在元数据中找不到 {image_ref} 的图像路径，使用回退方法")
        return None
