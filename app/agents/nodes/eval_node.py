"""
LangGraph EVAL node for the Enterprise RAG Platform.

Responsibility
--------------
This node runs **after** ``llm_node`` (and optionally after ``memory_node``).
It scores the current question/answer/context triple using RAGAS reference-free
metrics and writes results to ``state["evaluation_scores"]``.

Evaluation is **optional** — it is a quality signal, not part of the answer
generation flow.  Three design rules enforce this:

  1. ``eval_node`` is always a terminal side-branch in the graph.  It never
     influences routing decisions or answer content.
  2. Any exception inside RAGAS is caught and logged as a WARNING — the node
     always returns a valid partial state dict, never raises.
  3. Skipped evaluation returns ``evaluation_scores={}`` (empty dict), which
     is distinct from a failure (which returns ``{"error": "..."}`` inside
     the scores dict so the caller can distinguish the two cases).

When to skip (no RAGAS call made)
----------------------------------
- ``routing_decision`` is ``'direct'``        — no retrieved context to evaluate.
- ``routing_decision`` is ``'out_of_scope'``  — no answer generated; nothing to score.
- ``reranked_chunks`` is empty                — RAGAS context-precision metric
                                               requires at least one chunk.
- ``final_answer`` is empty                   — evaluation without an answer is meaningless.
- ``evaluate`` flag in state is ``False``     — caller opted out of evaluation
                                               (mirrors ``pipeline.py``'s ``evaluate=False``
                                               default for latency-sensitive paths).

Matching the existing pipeline pattern
---------------------------------------
``app/rag/pipeline.py`` accepts an ``evaluate: bool = False`` parameter and
only calls the evaluator when the caller explicitly passes ``evaluate=True``.
We replicate this behaviour by reading ``state.get("evaluate", False)`` —
meaning evaluation is **opt-in** per request, not automatic.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.state import AgentState
from app.evaluation.ragas_evaluator import evaluate_response

logger = logging.getLogger(__name__)


async def eval_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: run optional RAGAS evaluation on the current turn.

    Evaluation is skipped silently (returns ``evaluation_scores={}``) when:
    - ``state["evaluate"]`` is ``False`` or absent  (opt-in flag)
    - ``routing_decision`` is ``'direct'`` or ``'out_of_scope'``
    - ``reranked_chunks`` is empty (no context to evaluate against)
    - ``final_answer`` is empty (nothing to score)

    If RAGAS raises for any reason (rate limit, network, schema change),
    the exception is logged as a WARNING and ``evaluation_scores`` is set to
    ``{"error": "<message>"}`` so the caller can distinguish a skip from a
    failure without crashing the response flow.

    Args:
        state: The current ``AgentState`` dict.  Reads:
               - ``evaluate``           — bool opt-in flag (default: False).
               - ``routing_decision``   — skip if 'direct' or 'out_of_scope'.
               - ``current_question``   — passed to RAGAS as the user query.
               - ``final_answer``       — passed to RAGAS as the response.
               - ``reranked_chunks``    — text extracted and passed as contexts.

    Returns:
        Partial state dict:

        ``evaluation_scores``
            - ``{}``                   — evaluation skipped (opt-out or non-RAG path).
            - ``{"faithfulness": float, "response_relevancy": float, ...}``
                                       — RAGAS scores on success.
            - ``{"error": str}``       — RAGAS failed; str is the exception message.

        ``error``
            Always ``""`` — eval failures are non-fatal and expressed via
            ``evaluation_scores["error"]`` rather than the graph error channel.

    Raises:
        Does NOT raise.
    """
    # ------------------------------------------------------------------ #
    # Opt-in gate — mirrors pipeline.py's evaluate=False default          #
    # ------------------------------------------------------------------ #
    if not state.get("evaluate", False):
        logger.debug("eval_node: evaluation opted out (evaluate=False) — skipping.")
        return {"evaluation_scores": {}, "error": ""}

    # ------------------------------------------------------------------ #
    # Routing gate — only RAG turns have retrievable context to score     #
    # ------------------------------------------------------------------ #
    routing_decision: str = state.get("routing_decision", "")

    if routing_decision in ("direct", "out_of_scope"):
        logger.info(
            "eval_node: skipping evaluation for routing_decision=%r "
            "(no retrieved context to score).",
            routing_decision,
        )
        return {"evaluation_scores": {}, "error": ""}

    # ------------------------------------------------------------------ #
    # Field guards                                                         #
    # ------------------------------------------------------------------ #
    question: str = state.get("current_question", "").strip()
    answer: str = state.get("final_answer", "").strip()

    # Prefer reranked_chunks (final, cross-encoder ordered) over raw candidates.
    chunks: list[dict[str, Any]] = (
        state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    )

    if not answer:
        logger.warning("eval_node: final_answer is empty — skipping evaluation.")
        return {"evaluation_scores": {}, "error": ""}

    if not chunks:
        logger.warning(
            "eval_node: no reranked_chunks available — skipping evaluation. "
            "RAGAS context-precision metric requires at least one chunk."
        )
        return {"evaluation_scores": {}, "error": ""}

    # ------------------------------------------------------------------ #
    # Extract plain-text contexts for RAGAS                               #
    # ------------------------------------------------------------------ #
    # evaluate_response expects List[str] — one string per chunk.
    # We use the "text" key which is guaranteed by RetrievalService contract.
    contexts: list[str] = [
        chunk.get("text", "").strip()
        for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not contexts:
        logger.warning("eval_node: all chunks have empty text — skipping evaluation.")
        return {"evaluation_scores": {}, "error": ""}

    # ------------------------------------------------------------------ #
    # Run RAGAS (non-fatal — any exception returns error inside scores)   #
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            "eval_node: running RAGAS on question=%r (%d context chunks).",
            question[:80],
            len(contexts),
        )

        # evaluate_response is already async and dispatches the blocking
        # RAGAS compute to a thread via asyncio.to_thread internally.
        # It also catches its own exceptions and returns {"error": "..."}.
        scores: dict[str, Any] = await evaluate_response(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        if "error" in scores:
            # RAGAS's own fault-tolerant path — log but don't escalate.
            logger.warning(
                "eval_node: RAGAS returned an error score: %s", scores["error"]
            )
        else:
            logger.info(
                "eval_node: RAGAS scores for question=%r: %s",
                question[:80],
                {k: v for k, v in scores.items()},
            )

        return {"evaluation_scores": scores, "error": ""}

    except Exception as exc:  # noqa: BLE001
        # Belt-and-suspenders: evaluate_response should never raise, but if
        # our code around it does (e.g. context extraction bug), catch here.
        logger.warning(
            "eval_node: unexpected exception during evaluation: %s", exc, exc_info=True
        )
        return {"evaluation_scores": {"error": str(exc)}, "error": ""}
