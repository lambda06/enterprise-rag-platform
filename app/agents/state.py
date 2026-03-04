"""
LangGraph agent state definition for the Enterprise RAG Platform.

This module defines ``AgentState`` — the single shared state object that flows
through every node of the LangGraph ``StateGraph``.  Every node receives a copy
of this dict, mutates only the keys it owns, and returns the delta; LangGraph
merges deltas back into the canonical state automatically.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Why TypedDict and NOT a Pydantic BaseModel?
# ---------------------------------------------------------------------------
# LangGraph's StateGraph machinery works with plain Python dicts under the
# hood.  It merges partial updates (returned by each node) into the running
# state using simple dict-merge semantics rather than model instantiation.
#
# Using a Pydantic BaseModel would mean:
#   1. Every node return value would need to be a fully-valid model instance,
#      making it impossible to return *partial* updates (only changed keys).
#   2. LangGraph cannot directly use a Pydantic instance as a node's return
#      value — it expects either a plain dict or a TypedDict-annotated dict.
#   3. The overhead of Pydantic validation on every graph step is unnecessary:
#      LangGraph nodes are trusted internal functions, not external API inputs.
#
# TypedDict gives us:
#   ✔ Full static type-checking (mypy / Pyright) across all nodes.
#   ✔ IDE autocomplete for every field when writing node logic.
#   ✔  Zero runtime overhead — it's erased at runtime; it's just a dict.
#   ✔  Native compatibility with LangGraph's partial-update / reducer model.
#
# TL;DR — TypedDict is the LangGraph-idiomatic choice; Pydantic is great for
# API boundaries, TypedDict is great for internal graph plumbing.
# ---------------------------------------------------------------------------

from typing import Any

from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# ``total=False`` makes every key optional in the dict so that individual
# nodes can return partial updates without having to re-supply unchanged
# fields.  LangGraph merges the partial dict into the running state.
# ---------------------------------------------------------------------------


class AgentState(dict):
    """
    Typed dict representing all information flowing through the RAG agent graph.

    Field ownership by node
    -----------------------
    ROUTER          → routing_decision
    RAG_RETRIEVAL   → retrieved_chunks, cache_hit
    RERANKER        → reranked_chunks           (future node)
    MEMORY          → messages
    GENERATOR       → final_answer
    EVALUATOR       → evaluation_scores         (future node)
    ANY             → error (set on exception, checked by every node)

    All fields are Optional so a node only needs to return the keys it mutated.
    """

    # ------------------------------------------------------------------ #
    # Conversation history                                                #
    # ------------------------------------------------------------------ #

    messages: list[BaseMessage]
    """
    Full conversation history as LangChain ``BaseMessage`` objects
    (``HumanMessage``, ``AIMessage``, ``SystemMessage``, …).

    Grows with each turn; passed into the LLM as the prompt history.
    The MEMORY node is responsible for loading and persisting this list.
    """

    # ------------------------------------------------------------------ #
    # Current request                                                     #
    # ------------------------------------------------------------------ #

    current_question: str
    """
    The user's raw question for the current turn, extracted by the ROUTER
    node from the latest ``HumanMessage`` in ``messages``.
    """

    session_id: str
    """
    Opaque identifier that groups all turns belonging to one conversation.

    Set by the caller (e.g. the FastAPI endpoint) before invoking the graph.
    The MEMORY node uses it to scope PostgreSQL queries to the correct session
    so that conversation history is isolated per user / chat thread.

    Recommended format: a UUID string (``str(uuid.uuid4())``), but any
    non-empty string that uniquely identifies a session is valid.
    """

    routing_decision: str
    """
    Decision made by the ROUTER node.  Expected values (extend as needed):

    - ``"rag"``          → route to RAG_RETRIEVAL
    - ``"direct"``       → route to DIRECT_ANSWER (LLM parametric knowledge)
    - ``"out_of_scope"`` → route to a polite refusal node
    """

    # ------------------------------------------------------------------ #
    # Retrieval stage                                                     #
    # ------------------------------------------------------------------ #

    cache_hit: bool
    """
    ``True`` when the query was served from the Redis semantic cache,
    bypassing retrieval and LLM generation entirely.

    Set by the RAG_RETRIEVAL node before any vector-store calls are made.
    """

    retrieved_chunks: list[dict[str, Any]]
    """
    Raw candidates returned by the hybrid search (dense cosine + BM25 / RRF).

    Each dict has the shape produced by ``RetrievalService.retrieve_staged``::

        {
            "text":     str,           # chunk content
            "metadata": dict[str, Any] # source doc metadata (id, page, …)
        }

    Populated by the RAG_RETRIEVAL node; consumed by the RERANKER node.
    """

    reranked_chunks: list[dict[str, Any]]
    """
    Top-k chunks after cross-encoder reranking.

    Same schema as ``retrieved_chunks``.  Populated by the RERANKER node;
    consumed by the GENERATOR node to build the LLM context window.
    """

    # ------------------------------------------------------------------ #
    # Generation stage                                                    #
    # ------------------------------------------------------------------ #

    final_answer: str
    """
    The grounded answer produced by the LLM (GENERATOR node) or pulled
    directly from the semantic cache.

    Written once per turn; read by the OUTPUT node for streaming / returning
    to the caller.
    """

    # ------------------------------------------------------------------ #
    # Evaluation & quality signals                                        #
    # ------------------------------------------------------------------ #

    evaluation_scores: dict[str, float]
    """
    Quality metrics produced by the EVALUATOR node (optional, post-generation).

    Expected keys (all floats in [0, 1])::

        {
            "faithfulness":   float,  # answer grounded in retrieved context?
            "response_relevancy": float, # is the answer on-topic?
            "llm_context_precision_without_reference": float,
        }

    Set to ``{}`` when ``evaluate=False`` or evaluation is skipped.
    Set to ``{"error": str}`` when RAGAS raised but the graph continued.
    """

    evaluate: bool
    """
    Opt-in flag that controls whether ``eval_node`` runs RAGAS evaluation.

    Mirrors the ``evaluate: bool = False`` parameter on ``RAGPipeline.query``.
    Set to ``True`` in the initial state to enable per-turn quality scoring.
    Defaults to ``False`` so latency-sensitive paths pay zero evaluation cost.
    """

    # ------------------------------------------------------------------ #
    # Error handling                                                      #
    # ------------------------------------------------------------------ #

    error: str
    """
    Human-readable error message set by any node that raises an exception.

    Convention: a non-empty ``error`` string causes the OUTPUT node's
    conditional edge to route to a dedicated ERROR node instead of END,
    allowing graceful degradation (e.g., return a "I'm sorry" response
    instead of a 500 to the caller).

    Reset to ``""`` at the start of each new turn.
    """
