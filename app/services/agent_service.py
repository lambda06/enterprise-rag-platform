"""
AgentService — high-level orchestrator for the LangGraph RAG agent.

This service is the single entry point for all agent invocations.  It sits
between FastAPI endpoint handlers and the compiled ``agent_graph``, handling:

  1. Redis cache check  — skip the graph entirely on a cache hit.
  2. AgentState init    — builds the initial state dict from the caller's inputs.
  3. Graph invocation   — calls ``agent_graph.ainvoke`` and awaits completion.
  4. Result extraction  — pulls the fields that callers care about from the
                          completed state into a clean structured response dict.
  5. Cache write        — stores the response in Redis (1 h TTL) on success.

Cache key strategy
------------------
The cache key is an MD5 hash of ``session_id::question`` (lowercase, stripped).
Including ``session_id`` in the key scopes the cache per-session — the same
question asked in different sessions is cached independently.  This is the
correct behaviour for a conversational assistant where the answer may differ
based on session-specific context (uploaded documents, prior turns, etc.).

Why synchronous cache calls are wrapped in asyncio.to_thread
-------------------------------------------------------------
``CacheService`` uses Upstash's REST API via synchronous ``httpx`` calls
(no async client).  Wrapping them in ``asyncio.to_thread`` keeps the
FastAPI event loop unblocked so other requests can proceed concurrently
while a cache GET/SET is in flight.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from app.agents.graph import agent_graph
from app.agents.state import AgentState
from app.cache.redis_client import cache_service
from app.observability.langfuse_tracer import tracer

logger = logging.getLogger(__name__)

# Cache TTL for successful agent responses (seconds).  1 hour is a reasonable
# balance between freshness and avoiding repeated retrieval + LLM calls for
# identical questions.
CACHE_TTL_SECONDS: int = 3600

# Key prefix — makes it easy to identify agent-service keys in Redis and
# distinguish them from pipeline.py keys (which use the "rag:" prefix).
_CACHE_PREFIX = "agent:"


def _make_cache_key(session_id: str, question: str) -> str:
    """
    Return a stable MD5 cache key scoped to this session and question.

    Format before hashing: ``"<session_id>::<question>"`` (lowercase, stripped).

    Args:
        session_id: The caller's session identifier.
        question:   The raw user question.

    Returns:
        32-character lowercase hex MD5 digest prefixed with ``"agent:"``.
        Example: ``"agent:a1b2c3d4e5f6..."``.
    """
    raw = f"{session_id.strip().lower()}::{question.strip().lower()}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def _build_initial_state(
    question: str,
    session_id: str,
    evaluate: bool = False,
) -> AgentState:
    """
    Construct the initial ``AgentState`` dict passed to ``agent_graph.ainvoke``.

    Sets the fields that the graph entry point (``router_node``) needs and
    leaves all node-owned fields at their zero/empty defaults.  LangGraph
    merges each node's partial return dict over this base state.

    Args:
        question:   The user's raw question string.
        session_id: Session identifier — scopes memory and cache per session.
        evaluate:   If ``True``, ``eval_node`` will run RAGAS scoring.
                    Defaults to ``False`` for latency-sensitive paths.

    Returns:
        A fully initialised ``AgentState`` dict ready for graph invocation.
    """
    return AgentState(
        # ── Caller-supplied ──────────────────────────────────────────────
        current_question=question.strip(),
        session_id=session_id.strip(),
        evaluate=evaluate,

        # ── Node-owned — start empty; each node writes its own keys ──────
        messages=[],
        routing_decision="",
        cache_hit=False,
        retrieved_chunks=[],
        reranked_chunks=[],
        final_answer="",
        evaluation_scores={},
        error="",
    )


def _extract_response(completed_state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the caller-facing fields from the completed graph state.

    The full state dict contains many internal fields (raw retrieved_chunks,
    message history, etc.) that are useful for tracing but not for the API
    response.  This function projects only the fields the caller needs.

    Args:
        completed_state: The dict returned by ``agent_graph.ainvoke``.

    Returns:
        A clean response dict with keys:
        - ``answer``            — the final answer string
        - ``routing_decision``  — which path the graph took
        - ``sources``           — list of source metadata from reranked chunks
        - ``evaluation_scores`` — RAGAS metrics (empty dict if not run)
        - ``error``             — empty string on success, message on failure
        - ``cache_hit``         — always False for a graph execution response
    """
    # Extract source metadata from reranked chunks (the LLM-visible ones).
    # Each chunk has {"text": ..., "metadata": {...}}.  We surface only
    # metadata so the API response doesn't include full chunk text (which
    # could be large and is already stored in the DB via memory_node).
    reranked: list[dict] = completed_state.get("reranked_chunks") or []
    sources: list[dict] = [
        chunk.get("metadata", {})
        for chunk in reranked
        if chunk.get("metadata")
    ]

    return {
        "answer":            completed_state.get("final_answer", ""),
        "routing_decision":  completed_state.get("routing_decision", ""),
        "sources":           sources,
        "evaluation_scores": completed_state.get("evaluation_scores", {}),
        "error":             completed_state.get("error", ""),
        "cache_hit":         False,
    }


class AgentService:
    """
    High-level orchestrator for the LangGraph RAG agent.

    This class is stateless — all graph state lives inside ``agent_graph``'s
    invocation scope.  The singleton instance at the bottom of this module
    is safe to share across all FastAPI request handlers.

    Usage
    -----
    ::

        from app.services.agent_service import agent_service

        result = await agent_service.run(
            question="what are the payment terms?",
            session_id="user-123-abc",
        )
        print(result["answer"])
    """

    async def run(
        self,
        question: str,
        session_id: str,
        evaluate: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the RAG agent for a single question and return a structured result.

        Flow
        ----
        1. Validate inputs.
        2. Build the Redis cache key from ``session_id + question``.
        3. Check cache — return immediately on hit (no graph invocation).
        4. Build initial ``AgentState``.
        5. Invoke ``agent_graph.ainvoke`` — runs all nodes sequentially.
        6. Extract the response fields from the completed state.
        7. Write to Redis cache (1 h TTL) if no error occurred.
        8. Return the structured response dict.

        Args:
            question:   The user's raw question.
            session_id: Identifies the conversation session for memory scoping.
            evaluate:   If ``True``, run RAGAS evaluation on this turn.
                        Adds ~2–5 s of latency via extra Groq API calls.

        Returns:
            dict with keys:
            - ``answer``            (str)
            - ``routing_decision``  (str)
            - ``sources``           (list[dict])
            - ``evaluation_scores`` (dict)
            - ``error``             (str)
            - ``cache_hit``         (bool)

        Raises:
            Does NOT raise — all exceptions are caught and returned in ``error``.
        """
        question = question.strip()
        session_id = session_id.strip()

        # ── Input validation ─────────────────────────────────────────────
        if not question:
            return {
                "answer": "",
                "routing_decision": "",
                "sources": [],
                "evaluation_scores": {},
                "error": "Question must not be empty.",
                "cache_hit": False,
            }

        if not session_id:
            return {
                "answer": "",
                "routing_decision": "",
                "sources": [],
                "evaluation_scores": {},
                "error": "session_id must not be empty.",
                "cache_hit": False,
            }

        # ── Langfuse trace — wraps the entire request lifecycle ──────────
        # One trace per user question. Spans inside capture individual stages.
        # All tracing is fault-tolerant (no-op if Langfuse is not configured).
        lf_trace = tracer.start_trace(
            "agent-chat",
            input={"question": question},
            session_id=session_id,
            tags=["langgraph", "agent"],
        )

        # ── Cache key ────────────────────────────────────────────────────
        cache_key = _make_cache_key(session_id, question)

        # ── Cache read ───────────────────────────────────────────────────
        cache_span = tracer.start_span(lf_trace, "cache-check", input={"key": cache_key})
        try:
            cached = await asyncio.to_thread(
                cache_service.get_cached_response, cache_key
            )
            if cached is not None:
                logger.info(
                    "AgentService cache hit for session=%r question=%r",
                    session_id,
                    question[:80],
                )
                tracer.end_span(cache_span, output={"hit": True})
                tracer.end_trace(lf_trace, output=cached, metadata={"cache_hit": True})
                cached["cache_hit"] = True
                return cached
            tracer.end_span(cache_span, output={"hit": False})
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentService: cache read failed: %s", exc)
            tracer.end_span(cache_span, output={"hit": False, "error": str(exc)})

        # ── Graph invocation ─────────────────────────────────────────────
        graph_span = tracer.start_span(
            lf_trace, "langgraph-run",
            input={"question": question, "session_id": session_id, "evaluate": evaluate},
        )
        try:
            logger.info(
                "AgentService: invoking agent_graph for session=%r question=%r",
                session_id,
                question[:80],
            )

            initial_state = _build_initial_state(question, session_id, evaluate)
            completed_state: dict[str, Any] = await agent_graph.ainvoke(initial_state)
            response = _extract_response(completed_state)

            logger.info(
                "AgentService: graph completed — route=%r error=%r",
                response["routing_decision"],
                response["error"] or "(none)",
            )

            tracer.end_span(
                graph_span,
                output={"answer": response["answer"][:200], "route": response["routing_decision"]},
            )

        except Exception as exc:  # noqa: BLE001
            msg = f"AgentService: agent_graph invocation failed: {exc}"
            logger.exception(msg)
            tracer.end_span(graph_span, output={"error": str(exc)})
            tracer.end_trace(lf_trace, output={"error": msg})
            return {
                "answer": "",
                "routing_decision": "",
                "sources": [],
                "evaluation_scores": {},
                "error": msg,
                "cache_hit": False,
            }

        # ── Cache write ──────────────────────────────────────────────────
        should_cache = (
            not response["error"]
            and response["routing_decision"] != "out_of_scope"
            and response["answer"]
        )

        if should_cache:
            write_span = tracer.start_span(lf_trace, "cache-write", input={"key": cache_key})
            try:
                await asyncio.to_thread(
                    cache_service.cache_response,
                    cache_key,
                    response,
                    CACHE_TTL_SECONDS,
                )
                tracer.end_span(write_span, output={"stored": True})
                logger.debug(
                    "AgentService: cached response for session=%r key=%s",
                    session_id,
                    cache_key,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AgentService: cache write failed: %s", exc)
                tracer.end_span(write_span, output={"stored": False, "error": str(exc)})

        # ── End trace ────────────────────────────────────────────────────
        tracer.end_trace(
            lf_trace,
            output={"answer": response["answer"][:200]},
            metadata={
                "routing_decision":  response["routing_decision"],
                "evaluation_scores": response["evaluation_scores"],
                "cache_hit":         False,
                "error":             response["error"],
            },
        )

        return response



# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Return the singleton AgentService instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


agent_service = get_agent_service()
