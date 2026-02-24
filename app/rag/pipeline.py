"""
RAG pipeline that combines retrieval and LLM generation.

Provides `RAGPipeline` which uses the async `RetrievalService` to fetch
relevant context chunks and the `GroqLLMService` to generate a grounded
answer. The public `query` method is async and returns a dict with:

- `answer`:        LLM response string
- `source_chunks`: list of retrieved chunks (each with `text` and `metadata`)
- `chunk_count`:   number of returned chunks
- `evaluation`:    RAGAS metric scores (only present when ``evaluate=True``)
"""

from __future__ import annotations

import logging
from typing import Any

from app.llm.groq_client import groq_client
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
        """Answer a user question using retrieval + LLM.

        Steps:
        1. Retrieve top_k relevant chunks for the question.
        2. Pass the chunks to the LLM (grounded prompt) with the question.
        3. Optionally run RAGAS evaluation on the result.
        4. Return the answer, source chunks, count, and (if requested) scores.

        Args:
            user_question: The user's query string.
            top_k:         Number of chunks to retrieve (default 5).
            evaluate:      If True, run RAGAS evaluation and include scores
                           in the response under the ``"evaluation"`` key.
                           Adds latency due to extra LLM calls. Default False.

        Returns:
            dict with keys: ``answer``, ``source_chunks``, ``chunk_count``,
            and optionally ``evaluation``.
        """
        # Retrieve relevant chunks (async)
        source_chunks = await self._retrieval.retrieve(user_question, top_k)

        # Extract plain-text contexts as a concrete list so we can reuse it
        # both for the LLM prompt and (if requested) for RAGAS evaluation.
        # A generator would be exhausted after the first pass.
        contexts: list[str] = [c.get("text", "") for c in source_chunks]

        # Generate answer (async wrapper over sync Groq API call)
        answer = await self._llm.generate(user_question, iter(contexts))

        result: dict[str, Any] = {
            "answer": answer,
            "source_chunks": source_chunks,
            "chunk_count": len(source_chunks),
        }

        if evaluate:
            # ── Inline RAGAS evaluation ───────────────────────────────────────
            # Evaluation is opt-in: it makes additional Groq API calls
            # (typically 2-3 per metric) so it meaningfully increases latency.
            # For production workloads, use scripts/eval/batch_eval.py instead.
            from app.evaluation.ragas_evaluator import evaluate_response

            logger.info(
                "Running RAGAS evaluation for question: %.80s...", user_question
            )
            scores = await evaluate_response(user_question, answer, contexts)
            result["evaluation"] = scores

            # Log each individual score so they appear in server logs / APM
            if "error" in scores:
                logger.warning("RAGAS evaluation returned error: %s", scores["error"])
            else:
                score_str = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in scores.items()
                )
                logger.info("RAGAS scores — %s", score_str)

        return result


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


rag_pipeline = get_rag_pipeline()
