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

from app.rag.embeddings import embedding_service
from app.rag.reranker import reranker_service
from app.vectorstore.qdrant_client import QdrantService

logger = logging.getLogger(__name__)

# How many candidates to fetch from hybrid search before reranking.
# E.g., top_k=5 with factor=4 → fetch 20 candidates, rerank, return best 5.
RERANK_FACTOR: int = 4


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
            reranker_service.rerank, query, raw_candidates, top_k
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
            reranker_service.rerank, query, candidates, top_k
        )

        # Strip rrf_score from the final result set (keeps pipeline contract)
        reranked = [
            {"text": h.get("text", ""), "metadata": h.get("metadata", {})}
            for h in reranked_raw
        ]

        return candidates, reranked

    async def retrieve_with_vision(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[dict], list[dict], list[str]]:
        """Two-stage retrieval that separates image base64 for multimodal prompting.

        Extends :meth:`retrieve_staged` with vision support for Option B:
        image chunks have their ``image_base64`` extracted from metadata and
        returned separately so the RAG pipeline can pass them to Gemini in a
        single multimodal request.

        Image chunks have ``text=""`` which gives the cross-encoder a near-zero
        score.  To prevent images from being silently dropped, they are injected
        back into the final result list (up to ``top_k``) after reranking —
        scored with a ``"[image]"`` placeholder so the encoder doesn't crash.

        Args:
            query:  The user query string.
            top_k:  Number of final results to return (default 5).

        Returns:
            Three-tuple ``(candidates, reranked, image_b64_list)`` where:
            - ``candidates``:      raw hybrid-search hits (for Langfuse tracing).
            - ``reranked``:        final chunks, ``image_base64`` stripped from
                                   metadata (kept clean for the LLM context text).
            - ``image_b64_list``:  ordered list of base64 PNG strings for every
                                   image chunk in ``reranked``, passed directly
                                   to ``generate_multimodal_response``.
        """
        candidate_k = top_k * RERANK_FACTOR

        query_embedding: np.ndarray = await asyncio.to_thread(
            self._embed.embed_query, query
        )

        candidates: list[dict] = await asyncio.to_thread(
            self._qdrant.hybrid_search,
            query,
            query_embedding,
            candidate_k,
        )

        # ── Separate image and text candidates before reranking ───────────────
        # Image chunks have text="" so they would score near-zero in the
        # cross-encoder and be dropped.  We pull them out, score text candidates
        # normally, then force-merge images back into the final result.
        text_candidates: list[dict] = []
        image_candidates: list[dict] = []

        for hit in candidates:
            if hit.get("metadata", {}).get("content_type") == "image":
                image_candidates.append(hit)
            else:
                text_candidates.append(hit)

        # Rerank text candidates normally
        reranked_text: list[dict] = await asyncio.to_thread(
            reranker_service.rerank, query, text_candidates, top_k
        )

        # Merge image candidates back — cap total at top_k
        # Images come after text chunks so text context has priority.
        remaining_slots = max(0, top_k - len(reranked_text))
        merged_raw = reranked_text + image_candidates[:remaining_slots]

        logger.debug(
            "retrieve_with_vision: %d text chunks + %d image chunk(s) in final set",
            len(reranked_text),
            len(image_candidates[:remaining_slots]),
        )

        # ── Build final return structures ─────────────────────────────────────
        reranked: list[dict] = []
        image_b64_list: list[str] = []

        for hit in merged_raw:
            meta = dict(hit.get("metadata", {}))
            # Extract image_base64 from metadata — do NOT send raw base64
            # as part of the text context block (it would be enormous and
            # meaningless to a text-only context display).
            b64 = meta.pop("image_base64", "") or ""
            if hit.get("metadata", {}).get("content_type") == "image" and b64:
                image_b64_list.append(b64)

            reranked.append({
                "text": hit.get("text", ""),
                "metadata": meta,
            })

        return candidates, reranked, image_b64_list


# Module-level singleton
_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """Return the singleton RetrievalService instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


retrieval_service = get_retrieval_service()
