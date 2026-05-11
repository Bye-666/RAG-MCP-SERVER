"""
ChunkRefiner: Transform that cleans and refines chunk text.

Provides two modes:
1. Rule-based refinement: Fast, deterministic cleaning using regex patterns
2. LLM-enhanced refinement: Optional intelligent refinement using LLM

Falls back gracefully from LLM to rule-based on errors.
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
    Refines chunk text by removing noise and improving readability.

    Two-stage refinement:
    1. Rule-based cleaning (always applied)
    2. Optional LLM enhancement (if enabled and available)

    Graceful degradation: LLM failures fall back to rule-based results.
    """

    def __init__(
        self,
        settings,
        llm=None,
        prompt_path: Optional[str] = None
    ):
        """
        Initialize ChunkRefiner.

        Args:
            settings: Settings object with chunk_refiner configuration
            llm: Optional LLM instance (if None, will create from settings)
            prompt_path: Optional path to prompt template file
        """
        self.settings = settings
        # Safely check for ingestion.chunk_refiner.use_llm configuration
        self.use_llm = False
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'chunk_refiner'):
            self.use_llm = getattr(settings.ingestion.chunk_refiner, 'use_llm', False)

        # Initialize LLM if enabled
        self.llm = None
        if self.use_llm:
            if llm is not None:
                self.llm = llm
            else:
                try:
                    self.llm = LLMFactory.create(settings.llm)
                except Exception as e:
                    logger.warning(f"Failed to initialize LLM for chunk refinement: {e}")
                    self.use_llm = False

        # Load prompt template
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        Load prompt template from file.

        Args:
            prompt_path: Optional custom prompt path

        Returns:
            Prompt template string with {text} placeholder
        """
        if prompt_path is None:
            prompt_path = "config/prompts/chunk_refinement.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_path}: {e}")

        # Fallback prompt
        return "Refine the following text chunk to be clean and readable. Remove page headers/footers, excessive whitespace, and formatting marks while preserving code blocks and content structure.\n\n{text}"

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        Transform chunks by refining their text.

        Args:
            chunks: List of chunks to refine
            trace: Optional trace context

        Returns:
            List of refined chunks
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
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                # On error, keep original chunk
                refined_chunks.append(chunk)

        if trace and stage:
            trace.finish_stage(stage, {"chunks_processed": len(refined_chunks)})

        return refined_chunks

    def _refine_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """
        Refine a single chunk.

        Args:
            chunk: Chunk to refine
            trace: Optional trace context

        Returns:
            Refined chunk with updated text and metadata
        """
        # Always apply rule-based refinement first
        rule_refined = self._rule_based_refine(chunk.text)

        # If refinement results in empty text, return original chunk
        if not rule_refined or not rule_refined.strip():
            logger.warning(f"Chunk {chunk.id} became empty after refinement, keeping original")
            return chunk

        # Try LLM refinement if enabled
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

        # Create refined chunk with updated metadata
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
        Apply rule-based text cleaning.

        Removes:
        - Excessive whitespace
        - Page headers/footers patterns
        - HTML comments
        - Common formatting artifacts

        Preserves:
        - Code blocks
        - Markdown structure
        - Intentional formatting

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return text

        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove common page header/footer patterns
        # Pattern: "Page X of Y" or "Page X"
        text = re.sub(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Pattern: "Header: ..." or "Footer: ..." or standalone "Footer Text"
        text = re.sub(r'^\s*(Header|Footer):\s*.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*Footer\s+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Collapse multiple spaces (but not in code blocks)
        # Simple heuristic: preserve spacing in lines with indentation
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # If line starts with whitespace, it might be code - preserve internal spacing
            if line and line[0].isspace():
                cleaned_lines.append(line)
            else:
                # Collapse multiple spaces to single space
                cleaned_line = re.sub(r' {2,}', ' ', line)
                cleaned_lines.append(cleaned_line)

        text = '\n'.join(cleaned_lines)

        # Collapse excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _llm_refine(self, text: str, trace: Optional[TraceContext] = None) -> Optional[str]:
        """
        Apply LLM-based refinement.

        Args:
            text: Text to refine (already rule-cleaned)
            trace: Optional trace context

        Returns:
            Refined text, or None if LLM call fails
        """
        if not self.llm:
            return None

        try:
            # Format prompt with text
            prompt = self.prompt_template.format(text=text)

            # Call LLM
            response = self.llm.generate(prompt)

            # Extract refined text from response
            if response and response.strip():
                return response.strip()
            else:
                logger.warning("LLM returned empty response")
                return None

        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
