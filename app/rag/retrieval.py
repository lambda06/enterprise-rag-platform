"""
Retrieval service for the Enterprise RAG Platform.

Two-stage retrieval pipeline:
  1. **Hybrid search** (dense cosine + sparse BM25, fused with RRF)
     Casts a wide net of ``top_k * RERANK_FACTOR`` candidate chunks cheaply.
  2. **Cross-encoder reranking** (Jina Rerank API)
     Scores each (query, chunk) pair jointly so token-level interactions can
     be modelled — then returns only the best ``top_k`` chunks.

The public interface (``retrieve(query, top_k)``) is unchanged, so
``pipeline.py`` needs no modifications.

Threading notes
---------------
``QdrantService.hybrid_search`` is synchronous.  The Jina Rerank HTTP call in
``RerankerService.rerank()`` is also synchronous (httpx blocking mode).
All blocking calls are dispatched to a thread pool via ``asyncio.to_thread``
to keep the FastAPI event loop free.

Reranker
--------
Reranking is performed by the Jina Rerank API (jina-reranker-v2-base-multilingual).
Set ``JINA_API_KEY`` in the environment to enable it.  If the key is absent or
the API is unreachable, ``rerank()`` degrades gracefully to RRF score order.
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

        # ── Stage 3: relational context expansion from PostgreSQL ───────────
        # Expand winning child chunks (`~350 chars`) to their full parent chunks
        # (`~2,000 chars`) by querying the `parent_chunks` table (`id = parent_id`).
        unexpanded = [
            {
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
            }
            for hit in reranked
        ]
        return await self._expand_parents(unexpanded)

    async def _expand_parents(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stage 3: Expand child chunks to their full ~2,000 char ParentChunk context from PostgreSQL.

        For any retrieved chunk whose metadata contains a valid `parent_id`, this method
        queries the `parent_chunks` table (`SELECT content ... WHERE id = parent_id`).
        If found, the chunk's text is expanded to the complete parent document context,
        providing the LLM with unfragmented surrounding paragraphs and headings while
        preserving the child chunk's specific trigger text in metadata.
        """
        if not chunks:
            return chunks

        parent_ids = {
            c.get("metadata", {}).get("parent_id")
            for c in chunks
            if c.get("metadata", {}).get("parent_id")
        }
        if not parent_ids:
            return chunks

        parent_map: dict[str, str] = {}
        try:
            from app.db.session import async_session
            from app.models.document import ParentChunk
            from sqlalchemy import select

            async with async_session() as db:
                stmt = select(ParentChunk).where(ParentChunk.id.in_(parent_ids))
                result = await db.execute(stmt)
                for parent_row in result.scalars():
                    parent_map[parent_row.id] = parent_row.content
        except Exception as exc:
            logger.warning("Failed to expand parent chunks from PostgreSQL (`parent_chunks`): %s", exc)
            return chunks

        expanded: list[dict[str, Any]] = []
        for hit in chunks:
            meta = dict(hit.get("metadata", {}))
            parent_id = meta.get("parent_id")
            text = hit.get("text", "")

            if parent_id and parent_id in parent_map:
                meta["expanded_from_child"] = True
                meta["child_text"] = text
                text = parent_map[parent_id]

            expanded.append({
                "text": text,
                "metadata": meta,
            })

        return expanded

    async def retrieve_staged(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[dict], list[dict]]:
        """Two-stage retrieval exposing intermediate results for tracing.

        Identical to :meth:`retrieve` but returns both the raw hybrid-search
        candidates **and** the final reranked/expanded results as a tuple so that
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
            - ``reranked``: top ``top_k`` dicts after cross-encoder reranking and
              relational parent expansion (`_expand_parents`), with `text` and `metadata`.
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

        # Stage 3 — relational parent expansion
        unexpanded = [
            {"text": h.get("text", ""), "metadata": h.get("metadata", {})}
            for h in reranked_raw
        ]
        reranked = await self._expand_parents(unexpanded)

        return candidates, reranked

    async def retrieve_with_vision(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[dict], list[dict], list[str]]:
        """Three-stage retrieval that separates image base64 for multimodal prompting.

        Extends :meth:`retrieve_staged` with vision support for Option B:
        image chunks have their ``image_base64`` extracted from metadata and
        returned separately so the RAG pipeline can pass them to Gemini in a
        single multimodal request.

        Image chunks have descriptive Vision-to-Text captions (`text`) so they
        are scored by the cross-encoder alongside text candidates. After reranking,
        winning child chunks are expanded to their full parent context (`ParentChunk`)
        via PostgreSQL, while image base64 strings are extracted for multimodal synthesis.

        Args:
            query:  The user query string.
            top_k:  Number of final results to return (default 5).

        Returns:
            Three-tuple ``(candidates, reranked, image_b64_list)`` where:
            - ``candidates``:      raw hybrid-search hits (for Langfuse tracing).
            - ``reranked``:        final expanded chunks, ``image_base64`` stripped from
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

        # Stage 2 — rerank all candidates jointly across modalities
        # Image (`content_type: 'image'`) and Table (`content_type: 'table'`) chunks now carry
        # descriptive Gemini 2.5 Flash Lite natural language captions/summaries (`text`),
        # allowing the cross-encoder (`jina-reranker-v2-base-multilingual`) to accurately score
        # visual charts and tabular data alongside narrative prose text.
        reranked_raw: list[dict] = await asyncio.to_thread(
            reranker_service.rerank, query, candidates, top_k
        )

        logger.debug(
            "retrieve_with_vision: %d total chunk(s) selected by cross-encoder before parent expansion",
            len(reranked_raw),
        )

        # Stage 3 — relational parent expansion (`_expand_parents`)
        expanded_raw = await self._expand_parents(reranked_raw)

        # ── Build final return structures ─────────────────────────────────────
        reranked: list[dict] = []
        image_b64_list: list[str] = []

        for hit in expanded_raw:
            meta = dict(hit.get("metadata", {}))
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
