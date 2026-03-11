"""
Inline RAGAS evaluation for the Enterprise RAG Platform.

Provides a single async helper ``evaluate_response`` that scores one
question/answer/context triple using three **reference-free** RAGAS metrics:

  - Faithfulness            — Is the answer grounded in the context?
  - ResponseRelevancy       — Is the answer on-topic for the question?
  - LLMContextPrecisionWithoutReference — Are the retrieved chunks useful?

Using reference-free metrics means we need *no pre-built ground-truth
answer*, so evaluation can run inline on every live request.

The evaluator is intentionally fault-tolerant: if RAGAS raises for any
reason (rate limits, network, unexpected schema) the error is caught and
returned as ``{"error": "..."}`` so the pipeline never fails.
"""

from __future__ import annotations

# !! Must be set BEFORE any langchain/ragas imports !!
# LangGraph injects LangSmith callbacks into all nodes; disable entirely
# since we use Langfuse for tracing instead.
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _build_evaluator_llm():
    """Build a RAGAS-compatible LLM wrapper backed by Groq.

    RAGAS uses LangChain's chat model interface internally for metric
    computation.  ``LangchainLLMWrapper`` adapts any LangChain chat model
    to RAGAS without requiring OpenAI.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    from app.core.config import get_settings

    settings = get_settings()
    chat_llm = ChatGroq(
        api_key=settings.groq.api_key,
        model=settings.groq.model,
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat_llm)


def _build_evaluator_embeddings():
    """Build a RAGAS-compatible embeddings wrapper.

    ``ResponseRelevancy`` needs an embedding model to measure cosine
    similarity between the question and generated answer statements.
    We reuse the same ``all-MiniLM-L6-v2`` SentenceTransformer model
    that the ingestion pipeline already has cached locally — no extra
    downloads or API keys required.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


def _run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """Synchronous RAGAS evaluation — runs in a thread via asyncio.to_thread.

    RAGAS's ``evaluate()`` is synchronous and may make several LLM API calls.
    We isolate it here so the async FastAPI event loop never blocks.

    Args:
        question: The original user question.
        answer:   The generated LLM answer.
        contexts: Plain-text strings of the retrieved chunks.

    Returns:
        Dict of {metric_name: float_score}, e.g.
        {"faithfulness": 0.85, "response_relevancy": 0.92, ...}
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )

    evaluator_llm = _build_evaluator_llm()
    evaluator_embeddings = _build_evaluator_embeddings()

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]

    from langchain_core.callbacks import BaseCallbackHandler
    
    class SilentCallbackHandler(BaseCallbackHandler):
        pass

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        # ResponseRelevancy uses embeddings to score answer vs question;
        # without this it silently returns null for that metric.
        embeddings=evaluator_embeddings,
        # raise_exceptions=False keeps RAGAS from crashing on partial failures
        raise_exceptions=False,
        callbacks=[SilentCallbackHandler()],
    )

    # EvaluationResult in RAGAS 0.2 supports subscript access, not .get().
    # result[metric_name] returns a list of scores — one float per sample.
    # Since we always evaluate exactly one sample, we take index [0].
    scores: dict[str, Any] = {}
    for metric in metrics:
        key = metric.name          # e.g. "faithfulness"
        try:
            values = result[key]   # list[float | None], length == num samples
            val = values[0] if values else None
        except (KeyError, IndexError, TypeError):
            val = None
        scores[key] = round(float(val), 4) if val is not None else None

    return scores


async def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """Async entry point for RAGAS evaluation.

    Dispatches the synchronous RAGAS computation to a thread so the
    FastAPI event loop is never blocked.

    Args:
        question: The user's question.
        answer:   The LLM-generated answer.
        contexts: List of retrieved chunk texts used as context.

    Returns:
        Dict with metric scores, or ``{"error": "..."}`` on failure.
        Example success return:
            {
                "faithfulness": 0.8571,
                "response_relevancy": 0.9231,
                "llm_context_precision_without_reference": 0.7500,
            }
    """
    try:
        scores = await asyncio.to_thread(_run_ragas, question, answer, contexts)
        return scores
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc, exc_info=True)
        return {"error": str(exc)}
