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

  Stage 2 — Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
             Stored in ``state["reranked_chunks"]`` — the final top-k chunks
             ordered by cross-encoder relevance score.  This is what
             ``llm_node`` uses to build the LLM context window.

Why ``retrieve_staged()`` instead of ``retrieve()``?
-----------------------------------------------------
``retrieve()`` collapses both stages into one list — you only get the final
reranked results with no visibility into the raw hybrid-search candidates.
``retrieve_staged()`` returns ``(candidates, reranked)`` so we can:

  • Write raw candidates → ``retrieved_chunks`` and reranked → ``reranked_chunks``.
  • Let ``llm_node`` always prefer ``reranked_chunks`` (the higher-quality set)
    without any ambiguity about which field holds which stage.
  • Preserve the raw candidates for Langfuse tracing, evaluation, and
    debugging without an extra service call.

Why import the singleton directly?
-----------------------------------
``retrieval_service`` is a module-level singleton created in
``app.rag.retrieval`` at import time.  Importing it directly here means:

  • No repeated instantiation — the same ``RetrievalService`` instance,
    embedding model, and Qdrant connection are reused across every graph run.
  • Easy to mock in tests — replace ``rag_node.retrieval_service`` with a stub.
  • Consistent with the pattern used by ``pipeline.py`` and ``groq_client``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.state import AgentState
from app.rag.retrieval import retrieval_service

logger = logging.getLogger(__name__)

# Default number of final chunks returned to the GENERATOR node.
# Chosen as a balance between context richness and prompt-token cost:
#   • Too few (< 3): insufficient evidence for multi-part questions.
#   • Too many (> 8): risks hitting the LLM context window limit and adds noise.
DEFAULT_TOP_K: int = 5


async def rag_node(state: AgentState, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """
    LangGraph node: retrieve relevant chunks for the current question.

    Guards
    ------
    Returns an error immediately (without calling the vector store) if
    ``routing_decision`` is not ``"rag"``.  This prevents the node from
    doing expensive I/O when called incorrectly during graph development or
    testing.  In production the ROUTER's conditional edges should ensure
    this guard never triggers, but defensive programming here makes bugs
    visible early.

    Args:
        state:  The current ``AgentState`` dict.  Must contain:
                - ``current_question`` — the raw user query string.
                - ``routing_decision`` — must equal ``"rag"``.
        top_k:  Number of final reranked chunks to return.  Defaults to
                ``DEFAULT_TOP_K`` (5).  Pass a different value when calling
                the node directly in tests or scripts.

    Returns:
        A partial state dict containing only the keys this node owns:

        ``retrieved_chunks``
            Raw hybrid-search candidates (before reranking).  List of dicts
            with ``"text"``, ``"metadata"``, and ``"rrf_score"`` keys.
            Useful for tracing and evaluation; NOT what the LLM sees.

        ``reranked_chunks``
            Final top-k chunks after cross-encoder reranking, ordered by
            relevance score (most relevant first).  Each dict has
            ``"text"`` and ``"metadata"`` keys.  This is what ``llm_node``
            uses to build the LLM context window.

        ``error``
            Empty string on success.  Human-readable message on failure —
            a non-empty value causes the OUTPUT node's conditional edge to
            route to the ERROR node for graceful degradation.

    Raises:
        Does NOT raise — all exceptions are caught and written to ``error``.
    """
    # ------------------------------------------------------------------ #
    # Guard: verify this node should be running                           #
    # ------------------------------------------------------------------ #
    routing_decision: str = state.get("routing_decision", "")

    if routing_decision != "rag":
        # This should never happen in a correctly wired graph, but catches
        # misconfigured edges during development.
        msg = (
            f"rag_node invoked with routing_decision={routing_decision!r}. "
            "Expected 'rag'. Skipping retrieval."
        )
        logger.error(msg)
        return {
            "retrieved_chunks": [],
            "error": msg,
        }

    # ------------------------------------------------------------------ #
    # Extract question                                                    #
    # ------------------------------------------------------------------ #
    question: str = state.get("current_question", "").strip()

    if not question:
        msg = "rag_node received an empty current_question; cannot retrieve."
        logger.warning(msg)
        return {
            "retrieved_chunks": [],
            "error": msg,
        }

    # ------------------------------------------------------------------ #
    # Hybrid retrieval + cross-encoder reranking                          #
    # ------------------------------------------------------------------ #
    try:
        logger.debug(
            "rag_node starting staged retrieval for question=%r (top_k=%d)",
            question[:120],
            top_k,
        )

        # ``retrieve_staged`` returns (raw_candidates, reranked):
        #   raw_candidates — all hybrid-search hits (top_k * RERANK_FACTOR)
        #   reranked       — final top_k after cross-encoder scoring
        # Both are async-safe: blocking calls are offloaded via to_thread.
        raw_candidates, reranked = await retrieval_service.retrieve_staged(
            query=question,
            top_k=top_k,
        )

        logger.info(
            "rag_node: %d candidate(s) → %d reranked chunk(s) for question=%r",
            len(raw_candidates),
            len(reranked),
            question[:80],
        )

        if not reranked:
            # Zero results is not an error — the vector store may simply have
            # no relevant documents.  The GENERATOR node will handle this case
            # (e.g., respond "I couldn't find relevant documents").
            logger.warning(
                "rag_node: 0 reranked chunks for question=%r. "
                "The vector store may be empty or the query may not match any document.",
                question[:80],
            )

        return {
            "retrieved_chunks": raw_candidates,   # raw stage-1 candidates
            "reranked_chunks":  reranked,          # final stage-2 context
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001
        msg = f"rag_node retrieval failed: {exc}"
        logger.exception(msg)
        return {
            "retrieved_chunks": [],
            "reranked_chunks":  [],
            "error": msg,
        }
