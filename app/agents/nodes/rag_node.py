"""
LangGraph RAG_RETRIEVAL node for the Enterprise RAG Platform.

Responsibility
--------------
This node runs when the ROUTER has decided ``routing_decision == "rag"``.
It performs the full two-stage retrieval pipeline using ``retrieve_staged()``
so that both intermediate and final results are captured in state:

  Stage 1 — Hybrid search (dense cosine + sparse BM25 fused with RRF)
             Stored in ``state["retrieved_chunks"]`` — raw candidates before
             reranking.  Useful for debugging, tracing, and evaluation.

  Stage 2 — Cross-encoder reranking (ms-marco-MiniLM-L-6-v2 or fine-tuned)
             Stored in ``state["reranked_chunks"]`` — the final top-k chunks
             ordered by cross-encoder relevance score.  This is what
             ``llm_node`` uses to build the LLM context window.

Langfuse spans
--------------
This node opens two consecutive child spans on the parent trace:

  rag-retrieval — hybrid search stage.
      Logs ``chunk_count`` (candidates returned) and ``character_count``
      (total chars across all candidate texts) so operators can see
      how much raw text is flowing into the reranker.

  reranking — cross-encoder reranking stage.
      Logs ``model_source`` ("finetuned" or "base"), ``candidates_in``,
      ``chunks_out``, and ``chunks_dropped`` (candidates_in − chunks_out).
      High ``chunks_dropped`` with a low ``chunks_out`` is expected behaviour
      — it means the reranker aggressively filtered low-relevance candidates.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.state import AgentState
from app.core.config import get_settings
from app.observability.langfuse_tracer import tracer
from app.rag.retrieval import retrieval_service
from app.rag.reranker import reranker_service

logger = logging.getLogger(__name__)

# Default number of final chunks returned to the GENERATOR node.
DEFAULT_TOP_K: int = 5


async def rag_node(state: AgentState, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """
    LangGraph node: retrieve relevant chunks for the current question.

    Opens two Langfuse child spans: ``rag-retrieval`` (hybrid search) and
    ``reranking`` (cross-encoder), both attached to the parent trace passed
    in via ``state["lf_trace"]``.

    Args:
        state:  The current ``AgentState`` dict.  Must contain:
                - ``current_question`` — the raw user query string.
                - ``routing_decision`` — must equal ``"rag"``.
                - ``lf_trace``         — Langfuse trace for child spans.
        top_k:  Number of final reranked chunks to return.

    Returns:
        Partial state dict with ``retrieved_chunks``, ``reranked_chunks``,
        and ``error``.
    """
    # ------------------------------------------------------------------ #
    # Guard                                                               #
    # ------------------------------------------------------------------ #
    routing_decision: str = state.get("routing_decision", "")
    if routing_decision != "rag":
        msg = (
            f"rag_node invoked with routing_decision={routing_decision!r}. "
            "Expected 'rag'. Skipping retrieval."
        )
        logger.error(msg)
        return {"retrieved_chunks": [], "error": msg}

    question: str = state.get("current_question", "").strip()
    lf_trace = state.get("lf_trace")

    if not question:
        msg = "rag_node received an empty current_question; cannot retrieve."
        logger.warning(msg)
        return {"retrieved_chunks": [], "error": msg}

    # ------------------------------------------------------------------ #
    # Span 1: rag-retrieval (hybrid search)                              #
    # ------------------------------------------------------------------ #
    candidate_k = top_k * 4   # RERANK_FACTOR from retrieval.py

    retrieval_span = tracer.start_span(
        lf_trace,
        "rag-retrieval",
        input={
            "query":           question[:200],
            "candidate_k":     candidate_k,
            "collection_name": get_settings().qdrant.collection_name,
        },
    )

    try:
        logger.debug(
            "rag_node starting staged retrieval for question=%r (top_k=%d)",
            question[:120],
            top_k,
        )

        # retrieve_with_vision returns (raw_candidates, reranked, image_b64_list):
        #   raw_candidates — all hybrid-search hits (top_k * RERANK_FACTOR)
        #   reranked       — final top_k after cross-encoder scoring
        #   image_b64_list — base64 strings of extracted images
        raw_candidates, reranked, image_b64_list = await retrieval_service.retrieve_with_vision(
            query=question,
            top_k=top_k,
        )

        # Compute character_count across all candidate texts.
        # A high character_count with a moderate chunk_count indicates
        # large chunks — consider reducing chunk_size in the ingestion pipeline.
        candidate_char_count = sum(len(c.get("text", "")) for c in raw_candidates)

        tracer.end_span(
            retrieval_span,
            output={
                "chunk_count":   len(raw_candidates),
                # Renamed to context_chars for consistency with llm-generation span.
                # A high value with a moderate chunk_count = large chunks.
                "context_chars": candidate_char_count,
                "top_rrf_scores": [
                    round(float(c.get("rrf_score", 0)), 4)
                    for c in raw_candidates[:5]
                ],
            },
        )

    except Exception as exc:  # noqa: BLE001
        msg = f"rag_node retrieval failed: {exc}"
        logger.exception(msg)
        tracer.end_span(retrieval_span, output={"error": msg})
        return {"retrieved_chunks": [], "reranked_chunks": [], "image_b64_list": [], "error": msg}

    # ------------------------------------------------------------------ #
    # Span 2: reranking (cross-encoder)                                  #
    # ------------------------------------------------------------------ #
    reranking_span = tracer.start_span(
        lf_trace,
        "reranking",
        input={
            "candidates_in": len(raw_candidates),
            "top_k":         top_k,
        },
    )

    # top_reranker_score: the highest score assigned to any chunk by the
    # cross-encoder. A high score (close to 1.0 for a well-calibrated model)
    # means the reranker found a very relevant chunk; a low score across all
    # chunks is a signal that the retrieved candidates are weakly relevant.
    #
    # Note: rerank() returns dicts without scores (score was used for sorting).
    # We re-read the raw scores via the reranker_service's last predict() call
    # as a proxy by checking the first chunk's position — for now we log None
    # when scores aren't directly accessible, and rely on chunks_dropped instead.
    chunks_dropped = len(raw_candidates) - len(reranked)

    # The Jina API currently does not expose the scores after reranking
    # in the public interface. We log None for top_reranker_score.
    top_reranker_score: float | None = None

    tracer.end_span(
        reranking_span,
        output={
            # Confirms which cross-encoder model was used (e.g. from Jina API).
            "model_source":       getattr(reranker_service, "_model", "unavailable") if getattr(reranker_service, "_available", False) else "unavailable",
            "chunks_out":         len(reranked),
            # High chunks_dropped is normal — aggressive reranker filtering.
            "chunks_dropped":     chunks_dropped,
            # Confidence signal: how relevant was the best candidate?
            # Low top_reranker_score (<0.0 for ms-marco logits) means
            # no retrieved chunk is strongly relevant to the query.
            "top_reranker_score": top_reranker_score,
        },
    )

    logger.info(
        "rag_node: %d candidate(s) → %d reranked chunk(s) (dropped=%d, model=%s) "
        "for question=%r",
        len(raw_candidates),
        len(reranked),
        chunks_dropped,
        getattr(reranker_service, "_model", "unavailable") if getattr(reranker_service, "_available", False) else "unavailable",
        question[:80],
    )

    if not reranked:
        logger.warning(
            "rag_node: 0 reranked chunks for question=%r. "
            "The vector store may be empty or the query may not match any document.",
            question[:80],
        )

    return {
        "retrieved_chunks": raw_candidates,
        "reranked_chunks":  reranked,
        "image_b64_list":   image_b64_list,
        "error":            "",
    }
