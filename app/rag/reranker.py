import logging
import os
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import _PROJECT_ROOT

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self) -> None:
        """Initialize the RerankerService.
        
        We check for the fine-tuned model at initialization rather than hardcoding
        the path. This makes the service work correctly in both local development
        where the model exists, and production deployment where it may not have
        been copied yet or is otherwise unavailable.
        """
        finetuned_path = _PROJECT_ROOT / "models" / "finetuned-reranker"
        
        self.model: CrossEncoder | None = None
        self._model_source: str | None = None
        
        # Check if local fine-tuned model path exists and contains a valid model config
        if os.path.exists(finetuned_path) and os.path.exists(finetuned_path / "config.json"):
            try:
                self.model = CrossEncoder(str(finetuned_path), device="cpu", local_files_only=True)
                self._model_source = "finetuned"
                logger.info("Loaded fine-tuned reranker from models/finetuned-reranker/")
            except Exception as e:
                logger.warning("Failed to load fine-tuned reranker: %s", e)
                self.model = None
                
        # Fallback to the base model if necessary
        if self.model is None:
            base_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            try:
                self.model = CrossEncoder(base_model, device="cpu")
                self._model_source = "base"
                logger.info("Fine-tuned model not found, using base reranker model")
            except Exception as _exc:
                logger.warning(
                    "Cross-encoder reranker could not be loaded (%s). "
                    "Retrieval will use hybrid-search ranking only.",
                    _exc,
                )
    
    @property
    def model_source(self) -> str | None:
        """Returns 'finetuned' or 'base' depending on which model is active."""
        return self._model_source
        
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score and re-sort candidates with the cross-encoder (synchronous).

        If the reranker model is unavailable (failed to load), falls back to
        returning the first ``top_k`` candidates ordered by hybrid-search RRF score.

        Args:
            query:      The raw user query string.
            candidates: List of dicts with at least a ``"text"`` key.
            top_k:      Number of results to return after reranking.

        Returns:
            Top ``top_k`` dicts from ``candidates``, re-sorted by cross-encoder score
            (or by original RRF order if the reranker is unavailable).
        """
        if not candidates:
            return []
            
        if self.model is None:
            logger.debug("Reranker unavailable — returning top-%d by RRF order.", top_k)
            # Use a list comprehension to avoid Pyre slice typing complaints
            limit = min(len(candidates), top_k)
            return [candidates[i] for i in range(limit)]

        pairs = [(query, hit["text"]) for hit in candidates]
        
        # predict() returns a numpy array of float32 logits — higher = more relevant
        scores: np.ndarray = self.model.predict(pairs, show_progress_bar=False)
        
        ranked = sorted(
            zip(scores.tolist(), candidates),
            key=lambda t: t[0],
            reverse=True,
        )
        
        # Use list comprehension for top_k to avoid Pyre slice typing complaints
        limit = min(len(ranked), top_k)
        top_ranked = [ranked[i] for i in range(limit)]
        
        logger.debug(
            "Reranker top scores: %s",
            [round(float(s), 3) for s, _ in top_ranked],
        )
        
        return [hit for _, hit in top_ranked]

# Module-level singleton
reranker_service = RerankerService()
