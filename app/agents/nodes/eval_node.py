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

EvaluationResult fields logged to Langfuse spans
-------------------------------------------------
``content_types_evaluated``
    List of content-type strings that were actually present in the chunks
    scored by RAGAS (e.g. ``['text']``, ``['text', 'table']``, ``['image']``).
    Derived from the ``content_type`` key in each chunk's metadata; falls
    back to ``'text'`` when the key is absent (legacy chunks).
    Set to ``[]`` when evaluation is skipped.

``skipped_reason``
    ``None`` when RAGAS ran normally.  A short snake_case string when the
    node returned early without calling RAGAS:

    - ``'evaluate_flag_false'``   — opt-in flag not set
    - ``'non_rag_routing'``       — routing_decision was direct/out_of_scope
    - ``'empty_answer'``          — final_answer was blank
    - ``'no_chunks'``             — reranked_chunks list was empty
    - ``'no_text_chunks'``        — all chunks had empty text payloads
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agents.state import AgentState
from app.evaluation.ragas_evaluator import evaluate_response
from app.observability.langfuse_tracer import tracer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured container for the outcome of one eval_node execution.

    Attributes:
        scores: RAGAS metric scores (or ``{"error": str}`` on failure, or ``{}``
                when skipped).
        content_types_evaluated: Content-type labels of the chunks that were
                actually passed to RAGAS (e.g. ``['text', 'table']``).  Empty
                when evaluation was skipped.
        skipped_reason: ``None`` if RAGAS ran normally; otherwise a short
                snake_case string explaining why evaluation was skipped.
    """

    scores: dict[str, Any] = field(default_factory=dict)
    content_types_evaluated: list[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def _derive_content_types(chunks: list[dict[str, Any]]) -> list[str]:
    """Return a sorted, deduplicated list of content_type values from chunks.

    Reads ``chunk["metadata"]["content_type"]`` (the key written by all three
    extractor classes).  Falls back to ``"text"`` for chunks that pre-date the
    multimodal ingestion refactor and therefore lack the key.
    """
    seen: set[str] = set()
    for chunk in chunks:
        ct = (
            chunk.get("metadata", {}).get("content_type")
            or chunk.get("content_type")  # some upstream paths flatten metadata
            or "text"
        )
        seen.add(str(ct))
    return sorted(seen)


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
        return {
            "evaluation_scores": {
                "content_types_evaluated": [],
                "skipped_reason": "evaluate_flag_false",
            },
            "error": "",
        }

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
        return {
            "evaluation_scores": {
                "content_types_evaluated": [],
                "skipped_reason": "non_rag_routing",
            },
            "error": "",
        }

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
        return {
            "evaluation_scores": {
                "content_types_evaluated": [],
                "skipped_reason": "empty_answer",
            },
            "error": "",
        }

    if not chunks:
        logger.warning(
            "eval_node: no reranked_chunks available — skipping evaluation. "
            "RAGAS context-precision metric requires at least one chunk."
        )
        return {
            "evaluation_scores": {
                "content_types_evaluated": [],
                "skipped_reason": "no_chunks",
            },
            "error": "",
        }

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
        return {
            "evaluation_scores": {
                "content_types_evaluated": [],
                "skipped_reason": "no_text_chunks",
            },
            "error": "",
        }

    # Derive content types from the chunks that actually supplied text.
    # This is logged to the Langfuse span so analysts can see which modalities
    # contributed to the scored context (e.g. ['text'], ['text', 'table']).
    scored_chunks = [
        chunk for chunk in chunks if chunk.get("text", "").strip()
    ]
    content_types_evaluated: list[str] = _derive_content_types(scored_chunks)

    # ------------------------------------------------------------------ #
    # Run RAGAS (non-fatal — any exception returns error inside scores)   #
    # ------------------------------------------------------------------ #
    lf_trace = state.get("lf_trace")

    # Open an evaluation span so RAGAS run time and chunk count are visible
    # in Langfuse alongside the generation span.
    eval_span = tracer.start_span(
        lf_trace,
        "evaluation",
        input={
            "question":               question[:200],
            # RAGAS uses an internal LLM judge that makes several API calls per
            # metric. The SDK does not expose those token counts. chunks_evaluated
            # is logged as a proxy: more chunks = more judge calls = more tokens.
            "chunks_evaluated":       len(contexts),
            "question_chars":         len(question),
            "content_types_evaluated": content_types_evaluated,
        },
    )

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

        tracer.end_span(
            eval_span,
            output={
                "chunks_evaluated":       len(contexts),
                "content_types_evaluated": content_types_evaluated,
                "skipped_reason":         None,
                **{k: round(float(v), 4) if isinstance(v, float) else v
                   for k, v in scores.items()},
            },
        )

        return {
            "evaluation_scores": {
                **scores,
                "content_types_evaluated": content_types_evaluated,
                "skipped_reason": None,
            },
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001
        # Belt-and-suspenders: evaluate_response should never raise, but if
        # our code around it does (e.g. context extraction bug), catch here.
        logger.warning(
            "eval_node: unexpected exception during evaluation: %s", exc, exc_info=True
        )
        tracer.end_span(
            eval_span,
            output={
                "content_types_evaluated": content_types_evaluated,
                "skipped_reason": None,
                "error": str(exc),
            },
        )
        return {
            "evaluation_scores": {
                "error": str(exc),
                "content_types_evaluated": content_types_evaluated,
                "skipped_reason": None,
            },
            "error": "",
        }
