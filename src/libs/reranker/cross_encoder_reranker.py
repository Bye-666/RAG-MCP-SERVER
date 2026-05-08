"""Cross-Encoder based reranker"""

from typing import Any, Dict, List, Optional
from .base_reranker import BaseReranker
from ...core.trace import TraceContext


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker using sentence-transformers"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", scorer=None, timeout: Optional[float] = None):
        """
        Initialize Cross-Encoder reranker

        Args:
            model_name: Name of the cross-encoder model
            scorer: Optional scorer instance (for testing/mocking)
            timeout: Optional timeout in seconds for scoring
        """
        self.model_name = model_name
        self.timeout = timeout

        if scorer is not None:
            self.scorer = scorer
        else:
            try:
                from sentence_transformers import CrossEncoder
                self.scorer = CrossEncoder(model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                ) from e

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Cross-Encoder model

        Args:
            query: User query
            candidates: List of candidate dicts (must have 'text' field)
            trace: Optional trace context

        Returns:
            Reranked list of candidates with updated scores

        Raises:
            ValueError: If candidates format is invalid
            RuntimeError: If scoring fails (allows fallback to original ranking)
            TimeoutError: If scoring exceeds timeout (allows fallback)
        """
        if not candidates:
            return []

        if trace:
            trace.log("cross_encoder_reranker", f"Reranking {len(candidates)} candidates with {self.model_name}")

        # Validate candidates have required fields
        for i, cand in enumerate(candidates):
            if 'text' not in cand:
                raise ValueError(f"Candidate {i} missing 'text' field")

        # Build query-candidate pairs
        pairs = [[query, cand['text']] for cand in candidates]

        # Score pairs
        try:
            if self.timeout is not None:
                # For timeout support, we'd need to wrap the predict call
                # For now, we'll just pass through and let the caller handle timeout
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Cross-encoder scoring exceeded timeout of {self.timeout}s")

                # Note: signal.alarm only works on Unix systems
                # For Windows compatibility, we'll skip the actual timeout implementation
                # and just document the interface
                try:
                    if hasattr(signal, 'alarm'):
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(self.timeout))

                    scores = self.scorer.predict(pairs)

                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)  # Cancel alarm
                except TimeoutError:
                    raise
            else:
                scores = self.scorer.predict(pairs)

            if trace:
                trace.log("cross_encoder_reranker", f"Scored {len(scores)} pairs")

        except TimeoutError:
            raise  # Re-raise timeout for fallback handling
        except Exception as e:
            raise RuntimeError(f"Cross-encoder scoring failed: {str(e)}") from e

        # Build reranked result with scores
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = cand.copy()
            cand_copy['rerank_score'] = float(score)
            scored_candidates.append(cand_copy)

        # Sort by score (descending)
        reranked = sorted(scored_candidates, key=lambda x: x['rerank_score'], reverse=True)

        if trace:
            top_scores = [f"{c.get('id', 'N/A')}:{c['rerank_score']:.3f}" for c in reranked[:3]]
            trace.log("cross_encoder_reranker", f"Top 3 scores: {top_scores}")

        return reranked
