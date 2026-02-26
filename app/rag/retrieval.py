"""
Retrieval service for the Enterprise RAG Platform.

Two-stage retrieval pipeline:
  1. **Hybrid search** (dense cosine + sparse BM25, fused with RRF)
     Casts a wide net of ``top_k * RERANK_FACTOR`` candidate chunks cheaply.
  2. **Cross-encoder reranking** (ms-marco-MiniLM-L-6-v2)
     Scores each (query, chunk) pair jointly so token-level interactions can
     be modelled — then returns only the best ``top_k`` chunks.

The public interface (``retrieve(query, top_k)``) is unchanged, so
``pipeline.py`` needs no modifications.

Threading notes
---------------
Both ``sentence_transformers.CrossEncoder.predict`` and
``QdrantService.hybrid_search`` are synchronous.  All blocking calls are
dispatched to a thread pool via ``asyncio.to_thread`` to keep the FastAPI
event loop free.

Model loading
-------------
``CrossEncoder`` is loaded once at module import time as a singleton
(``_reranker``).  Loading takes ~1 s the first time; subsequent requests pay
zero overhead.  Download is ~24 MB and is cached by HuggingFace Hub.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.rag.embeddings import embedding_service
from app.vectorstore.qdrant_client import QdrantService

logger = logging.getLogger(__name__)

# ─── Cross-encoder reranker singleton ────────────────────────────────────────
# ms-marco-MiniLM-L-6-v2: ~24 MB, trained on MS-MARCO passage ranking.
# Loaded lazily and fault-tolerantly: if HuggingFace is unreachable (firewall,
# no internet, 403) the server still starts and serves requests — reranking is
# simply skipped and the pipeline falls back to hybrid-search ranking.
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    logger.info("Loading cross-encoder reranker: %s", _RERANKER_MODEL)
    _reranker: CrossEncoder | None = CrossEncoder(_RERANKER_MODEL, device="cpu")
    logger.info("Cross-encoder reranker ready.")
except Exception as _exc:
    logger.warning(
        "Cross-encoder reranker could not be loaded (%s). "
        "Retrieval will use hybrid-search ranking only. "
        "To fix: ensure huggingface.co is reachable, or set HF_TOKEN, or "
        "pre-download the model with: "
        "python -c \"from sentence_transformers import CrossEncoder; "
        "CrossEncoder('%s')\"",
        _exc,
        _RERANKER_MODEL,
    )
    _reranker = None

# How many candidates to fetch from hybrid search before reranking.
# E.g., top_k=5 with factor=4 → fetch 20 candidates, rerank, return best 5.
RERANK_FACTOR: int = 4


def _rerank(
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

    # ── Graceful fallback if the reranker model failed to load ────────────────
    if _reranker is None:
        logger.debug("Reranker unavailable — returning top-%d by RRF order.", top_k)
        return candidates[:top_k]

    pairs = [(query, hit["text"]) for hit in candidates]

    # predict() returns a numpy array of float32 logits — higher = more relevant
    scores: np.ndarray = _reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(scores.tolist(), candidates),
        key=lambda t: t[0],
        reverse=True,
    )

    logger.debug(
        "Reranker top scores: %s",
        [round(s, 3) for s, _ in ranked[:top_k]],
    )

    return [hit for _, hit in ranked[:top_k]]


class RetrievalService:
    """Async two-stage retrieval: hybrid search → cross-encoder reranking.

    Usage:
        service = RetrievalService()
        results = await service.retrieve("search text")
    """

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        embed_service=None,
    ) -> None:
        self._qdrant = qdrant_service or QdrantService()
        self._embed = embed_service or embedding_service

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k reranked results for the given query.

        Stage 1 — Hybrid search fetches ``top_k * RERANK_FACTOR`` candidates.
        Stage 2 — Cross-encoder reranks candidates and returns the best top_k.

        Args:
            query: The user query string.
            top_k: Number of final results to return (default 5).

        Returns:
            List of dicts with keys ``text`` and ``metadata``, ordered by
            cross-encoder relevance score (most relevant first).
        """
        candidate_k = top_k * RERANK_FACTOR   # e.g. 5 × 4 = 20

        # ── Stage 1: embed + hybrid search ──────────────────────────────────
        query_embedding: np.ndarray = await asyncio.to_thread(
            self._embed.embed_query, query
        )

        raw_candidates: list[dict[str, Any]] = await asyncio.to_thread(
            self._qdrant.hybrid_search,
            query,            # raw text → BM25 sparse leg
            query_embedding,  # dense vector → cosine leg
            candidate_k,      # fetch more than we need for reranking
        )

        logger.debug(
            "Hybrid search returned %d candidates for reranking.",
            len(raw_candidates),
        )

        # ── Stage 2: cross-encoder reranking ────────────────────────────────
        # Dispatched to a thread: CrossEncoder.predict is a blocking CPU call.
        reranked: list[dict[str, Any]] = await asyncio.to_thread(
            _rerank, query, raw_candidates, top_k
        )

        # hybrid_search already returns {"text": ..., "metadata": ..., "rrf_score": ...}
        # Strip the internal rrf_score key so the pipeline contract is unchanged.
        return [
            {
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
            }
            for hit in reranked
        ]

    async def retrieve_staged(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[dict], list[dict]]:
        """Two-stage retrieval exposing intermediate results for tracing.

        Identical to :meth:`retrieve` but returns both the raw hybrid-search
        candidates **and** the final reranked results as a tuple so that
        callers (e.g. ``pipeline.py``) can create separate Langfuse spans
        for the hybrid-search and reranking stages.

        Args:
            query:  The user query string.
            top_k:  Number of final results to return (default 5).

        Returns:
            ``(candidates, reranked)`` where:
            - ``candidates``: list of up to ``top_k * RERANK_FACTOR`` dicts
              from hybrid search, each with ``text``, ``metadata``, and
              ``rrf_score`` keys.
            - ``reranked``: top ``top_k`` dicts after cross-encoder reranking,
              with only ``text`` and ``metadata`` keys.
        """
        candidate_k = top_k * RERANK_FACTOR

        query_embedding: np.ndarray = await asyncio.to_thread(
            self._embed.embed_query, query
        )

        # Stage 1 — keep raw candidates (including rrf_score) for the trace
        candidates: list[dict] = await asyncio.to_thread(
            self._qdrant.hybrid_search,
            query,
            query_embedding,
            candidate_k,
        )

        # Stage 2 — rerank
        reranked_raw: list[dict] = await asyncio.to_thread(
            _rerank, query, candidates, top_k
        )

        # Strip rrf_score from the final result set (keeps pipeline contract)
        reranked = [
            {"text": h.get("text", ""), "metadata": h.get("metadata", {})}
            for h in reranked_raw
        ]

        return candidates, reranked


# Module-level singleton
_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """Return the singleton RetrievalService instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


retrieval_service = get_retrieval_service()
