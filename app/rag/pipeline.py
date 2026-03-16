"""
RAG pipeline that combines retrieval and LLM generation.

Provides `RAGPipeline` which uses the async `RetrievalService` to fetch
relevant context chunks and the configured LLM service (Gemini 2.0 Flash by
default, Groq as fallback) to generate a grounded answer.

LLM selection
-------------
The active provider is read from ``LLM_PROVIDER`` env var via
``app.llm.get_llm_service()``.  Default is ``"gemini"`` (Gemini 2.0 Flash).
Set ``LLM_PROVIDER=groq`` to fall back to Groq (text-only).

Vision at query time (Option B)
--------------------------------
When the query hits image chunks in Qdrant, ``RetrievalService.retrieve_with_vision``
separates the base64 images from the text chunks.  The pipeline then calls
``llm.generate_multimodal_response(question, contexts, images)`` — a single
Gemini request with full cross-attention over all text AND images simultaneously.
If no image chunks are retrieved, the cheaper text-only path is used.

The public ``query`` method is async and returns a dict with:

- ``answer``:        LLM response string
- ``source_chunks``: list of retrieved chunks (each with ``text`` and ``metadata``)
- ``chunk_count``:   number of returned chunks
- ``cache_hit``:     True if the response was served from Redis cache
- ``evaluation``:    RAGAS metric scores (only present when ``evaluate=True``)

Langfuse tracing
----------------
Every query creates a Langfuse trace with four child spans:

1. ``hybrid-search``   — dense + BM25 hybrid search; logs candidate count
                         and top RRF scores before reranking.
2. ``reranking``       — cross-encoder reranking; logs before/after chunk
                         ordering (source filenames + chunk indices).
3. ``llm-generation``  — Gemini/Groq completion; logs the prompt type
                         (text-only vs multimodal) and the generated answer.
4. (optional) ``ragas-evaluation`` — logged as trace metadata, not a span,
                         so evaluation scores appear on the trace overview.

Tracing is a no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are unset.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.cache.redis_client import cache_service, make_cache_key
from app.core.config import get_settings
from app.llm import get_llm_service
from app.observability.langfuse_tracer import tracer
from app.rag.retrieval import retrieval_service

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Combine retrieval and LLM to answer queries using retrieved context.

    The pipeline keeps the LLM grounded by providing only the retrieved
    chunks as context. This helps reduce hallucination risk and provides
    traceability for sourced answers.

    When image chunks are retrieved, a single multimodal Gemini request is
    built with all text contexts AND inline images so the model can reason
    over them jointly.

    Optionally, pass ``evaluate=True`` to `query()` to append inline RAGAS
    evaluation scores to the response. Evaluation is opt-in because it
    incurs additional LLM calls (roughly doubling latency per request).
    """

    def __init__(
        self,
        retrieval=None,
        llm=None,
    ) -> None:
        self._retrieval = retrieval or retrieval_service
        # Use injected LLM (useful for tests) or the configured factory default
        self._llm = llm or get_llm_service()

    async def query(
        self,
        user_question: str,
        top_k: int = 5,
        evaluate: bool = False,
    ) -> dict[str, Any]:
        """Answer a user question using retrieval + LLM with full Langfuse tracing.

        Steps:
        1. Cache lookup — return immediately on hit (no trace spans created).
        2. Hybrid search span — dense cosine + BM25, logs candidate pool.
        3. Reranking span — cross-encoder, logs before/after chunk ordering.
        4. LLM generation span — Gemini multimodal or text-only completion.
        5. Cache write — store result for 1 hour.
        6. (Optional) RAGAS evaluation — results logged as trace metadata.

        Args:
            user_question: The user's query string.
            top_k:         Number of final chunks to return (default 5).
            evaluate:      If True, run RAGAS evaluation and include scores
                           under the ``"evaluation"`` key. Default False.

        Returns:
            dict with keys: ``answer``, ``source_chunks``, ``chunk_count``,
            ``cache_hit``, and optionally ``evaluation``.
        """
        settings = get_settings()

        # ── Cache lookup ──────────────────────────────────────────────────────
        collection = settings.qdrant.collection_name
        cache_key = make_cache_key(user_question, collection)

        if not evaluate:
            cached = await asyncio.to_thread(
                cache_service.get_cached_response, cache_key
            )
            if cached is not None:
                cached["cache_hit"] = True
                return cached

        # ── Start Langfuse trace ──────────────────────────────────────────────
        llm_model = settings.gemini.generation_model
        trace = tracer.start_trace(
            "rag-query",
            input={"question": user_question},
            metadata={
                "collection": collection,
                "top_k": top_k,
                "evaluate": evaluate,
                "model": llm_model,
            },
        )

        # ── Span 1: Hybrid search ─────────────────────────────────────────────
        search_span = tracer.start_span(
            trace,
            "hybrid-search",
            input={"query": user_question, "candidate_k": top_k * 4},
        )

        # retrieve_with_vision returns (candidates, reranked, image_b64_list)
        candidates, source_chunks, image_b64_list = (
            await self._retrieval.retrieve_with_vision(user_question, top_k)
        )

        tracer.end_span(
            search_span,
            output={
                "candidate_count": len(candidates),
                "top_rrf_scores": [
                    round(float(c.get("rrf_score", 0)), 4)
                    for c in candidates[:5]
                ],
            },
        )

        # ── Span 2: Reranking ─────────────────────────────────────────────────
        rerank_span = tracer.start_span(
            trace,
            "reranking",
            input={
                "candidate_count": len(candidates),
                "before_order": [
                    {
                        "source": c.get("metadata", {}).get("source_filename", "?"),
                        "chunk": c.get("metadata", {}).get("chunk_index", "?"),
                        "rrf_score": round(float(c.get("rrf_score", 0)), 4),
                    }
                    for c in candidates[:top_k]
                ],
            },
        )

        tracer.end_span(
            rerank_span,
            output={
                "returned_count": len(source_chunks),
                "image_count": len(image_b64_list),
                "after_order": [
                    {
                        "source": c.get("metadata", {}).get("source_filename", "?"),
                        "chunk": c.get("metadata", {}).get("chunk_index", "?"),
                        "content_type": c.get("metadata", {}).get("content_type", "text"),
                    }
                    for c in source_chunks
                ],
            },
        )

        # Text context — for image chunks text is "" but metadata is still useful
        contexts: list[str] = [c.get("text", "") for c in source_chunks]

        # ── Span 3: LLM generation ────────────────────────────────────────────
        is_multimodal = len(image_b64_list) > 0
        gen_span = tracer.start_generation(
            trace,
            "llm-generation",
            model=llm_model,
            model_parameters={"temperature": 0.0, "max_output_tokens": 1024},
            input={
                "question": user_question,
                "context_count": len(contexts),
                "image_count": len(image_b64_list),
                "mode": "multimodal" if is_multimodal else "text-only",
            },
        )

        if is_multimodal:
            logger.info(
                "Multimodal generation: %d text chunks + %d image(s)",
                len(contexts),
                len(image_b64_list),
            )
            answer = await self._llm.generate_multimodal_response(
                user_question, contexts, image_b64_list
            )
        else:
            answer = await self._llm.generate_text_response(user_question, contexts)

        tracer.end_span(gen_span, output=answer)

        result: dict[str, Any] = {
            "answer": answer,
            "source_chunks": source_chunks,
            "chunk_count": len(source_chunks),
            "cache_hit": False,
        }

        # ── Cache write ───────────────────────────────────────────────────────
        if not evaluate:
            await asyncio.to_thread(
                cache_service.cache_response, cache_key, result, 3600
            )

        # ── Span 4 (optional): RAGAS evaluation ──────────────────────────────
        evaluation_metadata: dict[str, Any] = {}

        if evaluate:
            from app.evaluation.ragas_evaluator import evaluate_response

            logger.info(
                "Running RAGAS evaluation for question: %.80s...", user_question
            )
            scores = await evaluate_response(user_question, answer, contexts)
            result["evaluation"] = scores
            evaluation_metadata = scores

            if "error" in scores:
                logger.warning(
                    "RAGAS evaluation returned error: %s", scores["error"]
                )
            else:
                score_str = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in scores.items()
                )
                logger.info("RAGAS scores — %s", score_str)

        # ── Finalise trace ────────────────────────────────────────────────────
        tracer.end_trace(
            trace,
            output={"answer": answer, "chunk_count": len(source_chunks)},
            metadata={
                "cache_hit": False,
                "multimodal": is_multimodal,
                "evaluation": evaluation_metadata,
            },
        )
        tracer.flush()

        return result


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


rag_pipeline = get_rag_pipeline()
