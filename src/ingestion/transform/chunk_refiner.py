"""
ChunkRefiner: 清理和精炼块文本的转换器。

提供两种模式：
1. 基于规则的精炼：使用正则表达式模式的快速、确定性清理
2. LLM 增强精炼：使用 LLM 的可选智能精炼

在错误时从 LLM 优雅降级到基于规则。
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ChunkRefiner(BaseTransform):
    """
    通过去除噪音和提高可读性来精炼块文本。

    两阶段精炼：
    1. 基于规则的清理（始终应用）
    2. 可选的 LLM 增强（如果启用且可用）

    优雅降级：LLM 失败时回退到基于规则的结果。
    """

    def __init__(
        self,
        settings,
        llm=None,
        prompt_path: Optional[str] = None
    ):
        """
        初始化 ChunkRefiner。

        Args:
            settings: 包含 chunk_refiner 配置的设置对象
            llm: 可选的 LLM 实例（如果为 None，将从设置创建）
            prompt_path: 可选的提示模板文件路径
        """
        self.settings = settings
        # 安全检查 ingestion.chunk_refiner.use_llm 配置
        self.use_llm = False
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'chunk_refiner'):
            self.use_llm = getattr(settings.ingestion.chunk_refiner, 'use_llm', False)

        # 如果启用则初始化 LLM
        self.llm = None
        if self.use_llm:
            if llm is not None:
                self.llm = llm
            else:
                try:
                    self.llm = LLMFactory.create(settings.llm)
                except Exception as e:
                    logger.warning(f"无法为块精炼初始化 LLM: {e}")
                    self.use_llm = False

        # 加载提示模板
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        从文件加载提示模板。

        Args:
            prompt_path: 可选的自定义提示路径

        Returns:
            带有 {text} 占位符的提示模板字符串
        """
        if prompt_path is None:
            prompt_path = "config/prompts/chunk_refinement.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"无法从 {prompt_path} 加载提示: {e}")

        # 回退提示
        return "精炼以下文本块，使其干净且可读。删除页眉/页脚、过多的空白和格式标记，同时保留代码块和内容结构。\n\n{text}"

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        通过精炼文本来转换块。

        Args:
            chunks: 要精炼的块列表
            trace: 可选的跟踪上下文

        Returns:
            精炼后的块列表
        """
        stage = None
        if trace:
            stage = trace.record_stage("chunk_refiner", {"use_llm": self.use_llm})

        refined_chunks = []
        for chunk in chunks:
            try:
                refined_chunk = self._refine_chunk(chunk, trace)
                refined_chunks.append(refined_chunk)
            except Exception as e:
                logger.error(f"无法精炼块 {chunk.id}: {e}")
                # 出错时，保留原始块
                refined_chunks.append(chunk)

        if trace and stage:
            trace.finish_stage(stage, {"chunks_processed": len(refined_chunks)})

        return refined_chunks

    def _refine_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """
        精炼单个块。

        Args:
            chunk: 要精炼的块
            trace: 可选的跟踪上下文

        Returns:
            带有更新文本和元数据的精炼块
        """
        # 始终首先应用基于规则的精炼
        rule_refined = self._rule_based_refine(chunk.text)

        # 如果精炼导致文本为空，返回原始块
        if not rule_refined or not rule_refined.strip():
            logger.warning(f"块 {chunk.id} 在精炼后变为空，保留原始")
            return chunk

        # 如果启用则尝试 LLM 精炼
        final_text = rule_refined
        refined_by = "rule"
        fallback_reason = None

        if self.use_llm and self.llm:
            llm_result = self._llm_refine(rule_refined, trace)
            if llm_result is not None:
                final_text = llm_result
                refined_by = "llm"
            else:
                fallback_reason = "llm_failed"

        # 创建带有更新元数据的精炼块
        refined_chunk = Chunk(
            id=chunk.id,
            text=final_text,
            metadata={**chunk.metadata, "refined_by": refined_by},
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )

        if fallback_reason:
            refined_chunk.metadata["fallback_reason"] = fallback_reason

        return refined_chunk

    def _rule_based_refine(self, text: str) -> str:
        """
        应用基于规则的文本清理。

        删除：
        - 过多的空白
        - 页眉/页脚模式
        - HTML 注释
        - 常见格式伪影

        保留：
        - 代码块
        - Markdown 结构
        - 有意的格式

        Args:
            text: 要清理的文本

        Returns:
            清理后的文本
        """
        if not text or not text.strip():
            return text

        # 删除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 删除常见的页眉/页脚模式
        # 模式: "Page X of Y" 或 "Page X"
        text = re.sub(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # 模式: "Header: ..." 或 "Footer: ..." 或独立的 "Footer Text"
        text = re.sub(r'^\s*(Header|Footer):\s*.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*Footer\s+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # 折叠多个空格（但不在代码块中）
        # 简单启发式：保留缩进行中的间距
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 如果行以空白开头，可能是代码 - 保留内部间距
            if line and line[0].isspace():
                cleaned_lines.append(line)
            else:
                # 将多个空格折叠为单个空格
                cleaned_line = re.sub(r' {2,}', ' ', line)
                cleaned_lines.append(cleaned_line)

        text = '\n'.join(cleaned_lines)

        # 折叠过多的换行符（超过 2 个连续）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除前导/尾随空白
        text = text.strip()

        return text

    def _llm_refine(self, text: str, trace: Optional[TraceContext] = None) -> Optional[str]:
        """
        应用基于 LLM 的精炼。

        Args:
            text: 要精炼的文本（已经过规则清理）
            trace: 可选的跟踪上下文

        Returns:
            精炼后的文本，如果 LLM 调用失败则返回 None
        """
        if not self.llm:
            return None

        try:
            # 使用文本格式化提示
            prompt = self.prompt_template.format(text=text)

            # 调用 LLM
            response = self.llm.generate(prompt)

            # 从响应中提取精炼文本
            if response and response.strip():
                return response.strip()
            else:
                logger.warning("LLM 返回空响应")
                return None

        except Exception as e:
            logger.warning(f"LLM 精炼失败: {e}")
            return None
