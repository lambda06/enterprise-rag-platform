"""
RAG pipeline that combines retrieval and LLM generation.

Provides `RAGPipeline` which uses the async `RetrievalService` to fetch
relevant context chunks and the `GroqLLMService` to generate a grounded
answer. The public `query` method is async and returns a dict with:

- `answer`:        LLM response string
- `source_chunks`: list of retrieved chunks (each with `text` and `metadata`)
- `chunk_count`:   number of returned chunks
- `cache_hit`:     True if the response was served from Redis cache
- `evaluation`:    RAGAS metric scores (only present when ``evaluate=True``)

Langfuse tracing
----------------
Every query creates a Langfuse trace with four child spans:

1. ``hybrid-search``   — dense + BM25 hybrid search; logs candidate count
                         and top RRF scores before reranking.
2. ``reranking``       — cross-encoder reranking; logs before/after chunk
                         ordering (source filenames + chunk indices).
3. ``llm-generation``  — Groq completion; logs the full messages array as
                         input and the generated answer as output.
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
from app.llm.groq_client import groq_client
from app.observability.langfuse_tracer import tracer
from app.rag.retrieval import retrieval_service

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Combine retrieval and LLM to answer queries using retrieved context.

    The pipeline keeps the LLM grounded by providing only the retrieved
    chunks as context. This helps reduce hallucination risk and provides
    traceability for sourced answers.

    Optionally, pass ``evaluate=True`` to `query()` to append inline RAGAS
    evaluation scores to the response. Evaluation is opt-in because it
    incurs additional LLM calls (roughly doubling latency per request).
    In production, prefer the offline batch evaluation script instead.
    """

    def __init__(
        self,
        retrieval=None,
        llm=None,
    ) -> None:
        self._retrieval = retrieval or retrieval_service
        self._llm = llm or groq_client

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
        4. LLM generation span — Groq completion, logs full prompt + answer.
        5. Cache write — store result for 1 hour.
        6. (Optional) RAGAS evaluation — results logged as trace metadata.

        Args:
            user_question: The user's query string.
            top_k:         Number of final chunks to return (default 5).
            evaluate:      If True, run RAGAS evaluation and include scores
                           under the ``"evaluation"`` key. Evaluation results
                           are never cached. Default False.

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
        trace = tracer.start_trace(
            "rag-query",
            input={"question": user_question},
            metadata={
                "collection": collection,
                "top_k": top_k,
                "evaluate": evaluate,
                "model": settings.groq.model,
            },
        )

        # ── Span 1: Hybrid search ─────────────────────────────────────────────
        search_span = tracer.start_span(
            trace,
            "hybrid-search",
            input={"query": user_question, "candidate_k": top_k * 4},
        )

        candidates, source_chunks = await self._retrieval.retrieve_staged(
            user_question, top_k
        )

        tracer.end_span(
            search_span,
            output={
                "candidate_count": len(candidates),
                # Top 5 RRF scores give a quick signal of retrieval quality
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
                # Ordered list of (source_file, chunk_index) before reranking
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
                # Ordered list of (source_file, chunk_index) after reranking
                "after_order": [
                    {
                        "source": c.get("metadata", {}).get("source_filename", "?"),
                        "chunk": c.get("metadata", {}).get("chunk_index", "?"),
                    }
                    for c in source_chunks
                ],
            },
        )

        # Materialise contexts as a concrete list so it can be consumed twice
        # (once for the LLM prompt, once optionally for RAGAS).
        contexts: list[str] = [c.get("text", "") for c in source_chunks]

        # ── Span 3: LLM generation ────────────────────────────────────────────
        messages = self._llm._build_messages(user_question, iter(contexts))

        gen_span = tracer.start_generation(
            trace,
            "llm-generation",
            model=settings.groq.model,
            model_parameters={"temperature": 0.0, "max_tokens": 512},
            input=messages,
        )

        answer = await asyncio.to_thread(self._llm._call_sync, messages)

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
                "evaluation": evaluation_metadata,
            },
        )
        # Flush immediately so the trace is visible in Langfuse right away.
        # Without an explicit flush the SDK batches events and only sends them
        # every few seconds — which gets cut short by uvicorn --reload cycles.
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
