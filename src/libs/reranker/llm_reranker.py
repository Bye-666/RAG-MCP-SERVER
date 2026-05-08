import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base_reranker import BaseReranker
from ...core.trace import TraceContext


class LLMReranker(BaseReranker):
    """LLM-based reranker using structured output"""

    def __init__(self, llm, prompt_path: Optional[str] = None, prompt_text: Optional[str] = None):
        """
        Initialize LLM reranker

        Args:
            llm: LLM instance with chat() method
            prompt_path: Path to prompt template file (default: config/prompts/rerank.txt)
            prompt_text: Optional prompt text override (for testing)
        """
        self.llm = llm

        if prompt_text is not None:
            self.prompt_template = prompt_text
        else:
            if prompt_path is None:
                prompt_path = "config/prompts/rerank.txt"

            prompt_file = Path(prompt_path)
            if not prompt_file.exists():
                raise FileNotFoundError(f"Rerank prompt file not found: {prompt_path}")

            self.prompt_template = prompt_file.read_text(encoding='utf-8')

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using LLM

        Args:
            query: User query
            candidates: List of candidate dicts (must have 'id' and 'text' fields)
            trace: Optional trace context

        Returns:
            Reranked list of candidates with updated scores

        Raises:
            ValueError: If candidates format is invalid or LLM output doesn't match schema
            RuntimeError: If LLM call fails (allows fallback to original ranking)
        """
        if not candidates:
            return []

        if trace:
            trace.log("llm_reranker", f"Reranking {len(candidates)} candidates")

        # Validate candidates have required fields
        for i, cand in enumerate(candidates):
            if 'id' not in cand:
                raise ValueError(f"Candidate {i} missing 'id' field")
            if 'text' not in cand:
                raise ValueError(f"Candidate {i} missing 'text' field")

        # Build candidate list for prompt
        candidate_lines = []
        for cand in candidates:
            candidate_lines.append(f"ID: {cand['id']}\nText: {cand['text']}\n")

        candidates_text = "\n".join(candidate_lines)

        # Format prompt
        prompt = self.prompt_template.format(
            query=query,
            candidates=candidates_text
        )

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages)

            if trace:
                trace.log("llm_reranker", f"LLM response: {response[:200]}...")

        except Exception as e:
            raise RuntimeError(f"LLM reranking failed: {str(e)}") from e

        # Parse structured output
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Remove markdown code block markers
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

            result = json.loads(response_clean)

            if "ranked_ids" not in result:
                raise ValueError("LLM output missing 'ranked_ids' field")

            ranked_ids = result["ranked_ids"]

            if not isinstance(ranked_ids, list):
                raise ValueError(f"'ranked_ids' must be a list, got {type(ranked_ids).__name__}")

            # Validate all IDs are present
            candidate_ids = {cand['id'] for cand in candidates}
            ranked_id_set = set(ranked_ids)

            if ranked_id_set != candidate_ids:
                missing = candidate_ids - ranked_id_set
                extra = ranked_id_set - candidate_ids
                error_parts = []
                if missing:
                    error_parts.append(f"missing IDs: {missing}")
                if extra:
                    error_parts.append(f"extra IDs: {extra}")
                raise ValueError(f"Ranked IDs don't match candidates ({', '.join(error_parts)})")

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output is not valid JSON: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"LLM output doesn't match expected schema: {str(e)}") from e

        # Build reranked result
        id_to_candidate = {cand['id']: cand for cand in candidates}
        reranked = []

        for rank, cand_id in enumerate(ranked_ids):
            cand = id_to_candidate[cand_id].copy()
            cand['rerank_score'] = 1.0 / (rank + 1)  # Reciprocal rank as score
            reranked.append(cand)

        if trace:
            trace.log("llm_reranker", f"Reranked to: {[c['id'] for c in reranked]}")

        return reranked
