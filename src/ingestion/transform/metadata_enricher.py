"""
MetadataEnricher: Transform that enriches chunk metadata with title, summary, and tags.

Provides two modes:
1. Rule-based enrichment: Fast, deterministic metadata generation using heuristics
2. LLM-enhanced enrichment: Optional intelligent metadata generation using LLM

Falls back gracefully from LLM to rule-based on errors.
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
    Enriches chunk metadata with title, summary, and tags.

    Two-stage enrichment:
    1. Rule-based generation (always applied as fallback)
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
        Initialize MetadataEnricher.

        Args:
            settings: Settings object with metadata_enricher configuration
            llm: Optional LLM instance (if None, will create from settings)
            prompt_path: Optional path to prompt template file
        """
        self.settings = settings
        # Safely check for ingestion.metadata_enricher.use_llm configuration
        self.use_llm = False
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'metadata_enricher'):
            self.use_llm = getattr(settings.ingestion.metadata_enricher, 'use_llm', False)

        # Initialize LLM if enabled
        self.llm = None
        if self.use_llm:
            if llm is not None:
                self.llm = llm
            else:
                try:
                    self.llm = LLMFactory.create(settings.llm)
                except Exception as e:
                    logger.warning(f"Failed to initialize LLM for metadata enrichment: {e}")
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
            prompt_path = "config/prompts/metadata_enrichment.txt"

        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_path}: {e}")

        # Fallback prompt
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
        Transform chunks by enriching their metadata.

        Args:
            chunks: List of chunks to enrich
            trace: Optional trace context

        Returns:
            List of enriched chunks
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
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # On error, add rule-based metadata to original chunk
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
        Enrich a single chunk with metadata.

        Args:
            chunk: Chunk to enrich
            trace: Optional trace context

        Returns:
            Enriched chunk with updated metadata
        """
        # Always generate rule-based metadata as fallback
        rule_metadata = self._rule_based_enrich(chunk.text)

        # Try LLM enrichment if enabled
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

        # Create enriched chunk with updated metadata
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
        Generate metadata using rule-based heuristics.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with title, summary, and tags
        """
        if not text or not text.strip():
            return {
                "title": "Empty Chunk",
                "summary": "No content available",
                "tags": []
            }

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Extract title: first heading or first line
        title = self._extract_title(lines, text)

        # Generate summary: first 200 chars or first paragraph
        summary = self._extract_summary(text)

        # Extract tags: simple keyword extraction
        tags = self._extract_tags(text)

        return {
            "title": title,
            "summary": summary,
            "tags": tags
        }

    def _extract_title(self, lines: List[str], text: str) -> str:
        """Extract title from text using heuristics."""
        # Try to find markdown heading
        for line in lines[:5]:  # Check first 5 lines
            if line.startswith('#'):
                # Remove markdown heading markers
                title = re.sub(r'^#+\s*', '', line)
                return title[:100]  # Max 100 chars

        # Use first non-empty line
        if lines:
            return lines[0][:100]

        # Fallback: first 50 chars
        return text.strip()[:50]

    def _extract_summary(self, text: str) -> str:
        """Extract summary from text."""
        # Remove markdown headings for summary
        text_no_headings = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text_clean = text_no_headings.strip()

        if not text_clean:
            text_clean = text.strip()

        # Take first 200 chars
        summary = text_clean[:200]

        # Try to end at sentence boundary
        last_period = summary.rfind('.')
        last_question = summary.rfind('?')
        last_exclaim = summary.rfind('!')

        last_sentence_end = max(last_period, last_question, last_exclaim)

        if last_sentence_end > 50:  # Only truncate if we have a reasonable sentence
            summary = summary[:last_sentence_end + 1]

        return summary

    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags using simple keyword extraction."""
        # Convert to lowercase for analysis
        text_lower = text.lower()

        # Common technical keywords to look for
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

        # Extract capitalized words (likely important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        tags.update([word.lower() for word in capitalized[:3]])

        # Limit to 5 tags
        return sorted(list(tags))[:5]

    def _llm_enrich(self, text: str, trace: Optional[TraceContext] = None) -> Optional[Dict[str, Any]]:
        """
        Generate metadata using LLM.

        Args:
            text: Text to analyze
            trace: Optional trace context

        Returns:
            Dictionary with title, summary, and tags, or None on failure
        """
        if not self.llm:
            return None

        try:
            # Truncate text if too long (to avoid token limits)
            max_chars = 2000
            text_truncated = text[:max_chars]
            if len(text) > max_chars:
                text_truncated += "..."

            # Format prompt
            prompt = self.prompt_template.format(text=text_truncated)

            # Call LLM
            response = self.llm.generate(prompt)

            # Parse JSON response
            metadata = self._parse_llm_response(response)

            if metadata:
                return metadata
            else:
                logger.warning("Failed to parse LLM response for metadata enrichment")
                return None

        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract metadata.

        Args:
            response: LLM response text

        Returns:
            Dictionary with title, summary, and tags, or None on parse failure
        """
        try:
            # Try to extract JSON from response
            # LLM might wrap JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            # Parse JSON
            metadata = json.loads(json_str)

            # Validate required fields
            if not all(key in metadata for key in ['title', 'summary', 'tags']):
                logger.warning("LLM response missing required fields")
                return None

            # Validate types
            if not isinstance(metadata['title'], str):
                return None
            if not isinstance(metadata['summary'], str):
                return None
            if not isinstance(metadata['tags'], list):
                return None

            # Truncate if needed
            metadata['title'] = metadata['title'][:100]
            metadata['summary'] = metadata['summary'][:200]
            metadata['tags'] = metadata['tags'][:5]

            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return None
