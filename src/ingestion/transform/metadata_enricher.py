"""
MetadataEnricher: 使用标题、摘要和标签丰富分块元数据的转换器。

提供两种模式：
1. 基于规则的丰富：使用启发式方法快速、确定性地生成元数据
2. LLM 增强丰富：可选的使用 LLM 进行智能元数据生成

在 LLM 出错时优雅地回退到基于规则的方法。
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.types import Chunk
from src.core.trace import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """
    使用标题、摘要和标签丰富分块元数据。

    两阶段丰富：
    1. 基于规则的生成（始终作为回退应用）
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
        初始化 MetadataEnricher。

        Args:
            settings: 包含 metadata_enricher 配置的设置对象
            llm: 可选的 LLM 实例（如果为 None，将从设置创建）
            prompt_path: 可选的提示模板文件路径
        """
        self.settings = settings
        # 安全检查 ingestion.metadata_enricher.use_llm 配置
        self.use_llm = False
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'metadata_enricher'):
            self.use_llm = getattr(settings.ingestion.metadata_enricher, 'use_llm', False)

        # 如果启用，初始化 LLM
        self.llm = None
        if self.use_llm:
            if llm is not None:
                self.llm = llm
            else:
                try:
                    self.llm = LLMFactory.create(settings.llm)
                except Exception as e:
                    logger.warning(f"初始化元数据丰富的 LLM 失败: {e}")
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
            prompt_path = "config/prompts/metadata_enrichment.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"从 {prompt_path} 加载提示失败: {e}")

        # 回退提示
        return """Analyze the following text chunk and generate metadata in JSON format.

Text:
{text}

Generate a JSON object with:
- "title": A concise, descriptive title (max 100 chars)
- "summary": A brief summary of the main content (max 200 chars)
- "tags": A list of 3-5 relevant keywords/topics

Output only valid JSON, no additional text."""

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        通过丰富元数据来转换分块。

        Args:
            chunks: 要丰富的分块列表
            trace: 可选的追踪上下文

        Returns:
            丰富后的分块列表
        """
        stage = None
        if trace:
            stage = trace.record_stage("metadata_enricher", {"use_llm": self.use_llm})

        enriched_chunks = []
        for chunk in chunks:
            try:
                enriched_chunk = self._enrich_chunk(chunk, trace)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"丰富分块 {chunk.id} 失败: {e}")
                # 出错时，向原始分块添加基于规则的元数据
                rule_metadata = self._rule_based_enrich(chunk.text)
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={**chunk.metadata, **rule_metadata, "enriched_by": "rule", "enrichment_error": str(e)},
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)

        if trace and stage:
            trace.finish_stage(stage, {"chunks_processed": len(enriched_chunks)})

        return enriched_chunks

    def _enrich_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """
        使用元数据丰富单个分块。

        Args:
            chunk: 要丰富的分块
            trace: 可选的追踪上下文

        Returns:
            带有更新元数据的丰富分块
        """
        # 始终生成基于规则的元数据作为回退
        rule_metadata = self._rule_based_enrich(chunk.text)

        # 如果启用，尝试 LLM 丰富
        final_metadata = rule_metadata
        enriched_by = "rule"
        fallback_reason = None

        if self.use_llm and self.llm:
            llm_result = self._llm_enrich(chunk.text, trace)
            if llm_result is not None:
                final_metadata = llm_result
                enriched_by = "llm"
            else:
                fallback_reason = "llm_failed"

        # 创建带有更新元数据的丰富分块
        enriched_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata={**chunk.metadata, **final_metadata, "enriched_by": enriched_by},
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )

        if fallback_reason:
            enriched_chunk.metadata["fallback_reason"] = fallback_reason

        return enriched_chunk

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """
        使用基于规则的启发式方法生成元数据。

        Args:
            text: 要分析的文本

        Returns:
            包含标题、摘要和标签的字典
        """
        if not text or not text.strip():
            return {
                "title": "空分块",
                "summary": "无可用内容",
                "tags": []
            }

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # 提取标题：第一个标题或第一行
        title = self._extract_title(lines, text)

        # 生成摘要：前 200 个字符或第一段
        summary = self._extract_summary(text)

        # 提取标签：简单的关键词提取
        tags = self._extract_tags(text)

        return {
            "title": title,
            "summary": summary,
            "tags": tags
        }

    def _extract_title(self, lines: List[str], text: str) -> str:
        """使用启发式方法从文本中提取标题。"""
        # 尝试查找 markdown 标题
        for line in lines[:5]:  # 检查前 5 行
            if line.startswith('#'):
                # 移除 markdown 标题标记
                title = re.sub(r'^#+\s*', '', line)
                return title[:100]  # 最多 100 个字符

        # 使用第一个非空行
        if lines:
            return lines[0][:100]

        # 回退：前 50 个字符
        return text.strip()[:50]

    def _extract_summary(self, text: str) -> str:
        """从文本中提取摘要。"""
        # 移除 markdown 标题以生成摘要
        text_no_headings = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text_clean = text_no_headings.strip()

        if not text_clean:
            text_clean = text.strip()

        # 取前 200 个字符
        summary = text_clean[:200]

        # 尝试在句子边界结束
        last_period = summary.rfind('.')
        last_question = summary.rfind('?')
        last_exclaim = summary.rfind('!')

        last_sentence_end = max(last_period, last_question, last_exclaim)

        if last_sentence_end > 50:  # 只有在有合理句子时才截断
            summary = summary[:last_sentence_end + 1]

        return summary

    def _extract_tags(self, text: str) -> List[str]:
        """使用简单的关键词提取来提取标签。"""
        # 转换为小写以进行分析
        text_lower = text.lower()

        # 要查找的常见技术关键词
        keyword_patterns = [
            r'\b(api|rest|graphql|sdk)\b',
            r'\b(database|sql|nosql|mongodb|postgres)\b',
            r'\b(authentication|authorization|oauth|jwt)\b',
            r'\b(docker|kubernetes|container)\b',
            r'\b(python|javascript|java|typescript|go|rust)\b',
            r'\b(test|testing|unit test|integration)\b',
            r'\b(security|encryption|ssl|tls)\b',
            r'\b(performance|optimization|cache)\b',
            r'\b(frontend|backend|fullstack)\b',
            r'\b(machine learning|ai|neural network)\b',
        ]

        tags = set()
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text_lower)
            tags.update(matches)

        # 提取大写单词（可能是重要术语）
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        tags.update([word.lower() for word in capitalized[:3]])

        # 限制为 5 个标签
        return sorted(list(tags))[:5]

    def _llm_enrich(self, text: str, trace: Optional[TraceContext] = None) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 生成元数据。

        Args:
            text: 要分析的文本
            trace: 可选的追踪上下文

        Returns:
            包含标题、摘要和标签的字典，失败时返回 None
        """
        if not self.llm:
            return None

        try:
            # 如果文本太长则截断（避免超出 token 限制）
            max_chars = 2000
            text_truncated = text[:max_chars]
            if len(text) > max_chars:
                text_truncated += "..."

            # 格式化提示
            prompt = self.prompt_template.format(text=text_truncated)

            # 调用 LLM
            response = self.llm.generate(prompt)

            # 解析 JSON 响应
            metadata = self._parse_llm_response(response)

            if metadata:
                return metadata
            else:
                logger.warning("解析元数据丰富的 LLM 响应失败")
                return None

        except Exception as e:
            logger.warning(f"LLM 丰富失败: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 响应以提取元数据。

        Args:
            response: LLM 响应文本

        Returns:
            包含标题、摘要和标签的字典，解析失败时返回 None
        """
        try:
            # 尝试从响应中提取 JSON
            # LLM 可能会将 JSON 包装在 markdown 代码块中
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接查找 JSON 对象
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            # 解析 JSON
            metadata = json.loads(json_str)

            # 验证必需字段
            if not all(key in metadata for key in ['title', 'summary', 'tags']):
                logger.warning("LLM 响应缺少必需字段")
                return None

            # 验证类型
            if not isinstance(metadata['title'], str):
                return None
            if not isinstance(metadata['summary'], str):
                return None
            if not isinstance(metadata['tags'], list):
                return None

            # 如果需要则截断
            metadata['title'] = metadata['title'][:100]
            metadata['summary'] = metadata['summary'][:200]
            metadata['tags'] = metadata['tags'][:5]

            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"从 LLM 响应解析 JSON 失败: {e}")
            return None
        except Exception as e:
            logger.warning(f"解析 LLM 响应时出错: {e}")
            return None
