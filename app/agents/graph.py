"""
LangGraph StateGraph definition for the Enterprise RAG Platform agent.

Graph topology
--------------

                         ┌─────────────┐
              START ────>│ router_node │
                         └──────┬──────┘
                                │
              conditional_edge(route_decision)
                ┌───────────────┼───────────────┐
                │               │               │
             "rag"          "direct"     "out_of_scope"
                │               │               │
                v               │               │
         ┌──────────┐           │               │
         │ rag_node │           │               │
         └────┬─────┘           │               │
              │                 │               │
              └────────>┌───────┴───────┐<──────┘
                         │  llm_node    │
                         └──────┬───────┘
                                │
                         ┌──────v───────┐
                         │ memory_node  │
                         └──────┬───────┘
                                │
                         ┌──────v───────┐
                         │  eval_node   │
                         └──────┬───────┘
                                │
                               END

Routing logic
-------------
``router_node`` sets ``state["routing_decision"]`` to one of:
  "rag"           → rag_node  → llm_node
  "direct"        → llm_node  (skip retrieval)
  "out_of_scope"  → llm_node  (returns canned refusal, no LLM call)

``llm_node``, ``memory_node``, and ``eval_node`` are always linear —
they run unconditionally after the generation stage.

Compilation
-----------
``agent_graph`` is the compiled ``CompiledStateGraph`` ready for ``.invoke()``
or ``.ainvoke()``.  Compile is called once at module import time so the
graph is shared across all requests in the process.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agents.nodes.eval_node import eval_node
from app.agents.nodes.llm_node import llm_node
from app.agents.nodes.memory_node import memory_node
from app.agents.nodes.rag_node import rag_node
from app.agents.nodes.router import router_node
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing constants — match the values documented in router.py
# ---------------------------------------------------------------------------
_ROUTE_RAG = "rag"
_ROUTE_DIRECT = "direct"
_ROUTE_OUT_OF_SCOPE = "out_of_scope"

# ---------------------------------------------------------------------------
# Node name constants — single source of truth for graph wiring
# ---------------------------------------------------------------------------
_NODE_ROUTER = "router"
_NODE_RAG = "rag_retrieval"
_NODE_LLM = "llm"
_NODE_MEMORY = "memory"
_NODE_EVAL = "eval"


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def _route_decision(state: AgentState) -> str:
    """
    Conditional edge function: read routing_decision and return the next node.

    Called by LangGraph after ``router_node`` completes.  The return value
    must match one of the keys in the ``add_conditional_edges`` mapping dict.

    Args:
        state: Current ``AgentState`` — must contain ``routing_decision``.

    Returns:
        One of ``"rag_retrieval"``, ``"llm"``, or ``"llm"`` (out_of_scope
        also routes to ``"llm"`` — the llm_node handles the refusal branch
        internally without making an LLM API call).
    """
    decision = state.get("routing_decision", _ROUTE_DIRECT)

    logger.debug("_route_decision: routing_decision=%r", decision)

    if decision == _ROUTE_RAG:
        return _NODE_RAG

    if decision == _ROUTE_DIRECT:
        return _NODE_LLM

    if decision == _ROUTE_OUT_OF_SCOPE:
        # llm_node detects out_of_scope via routing_decision and returns a
        # canned refusal without calling the Groq API — zero token cost.
        return _NODE_LLM

    # Unknown decision — default to direct LLM answer (safest fallback).
    logger.warning(
        "_route_decision: unknown routing_decision=%r; defaulting to '%s'.",
        decision,
        _NODE_LLM,
    )
    return _NODE_LLM


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """
    Construct and return the uncompiled ``StateGraph``.

    Separated from ``_compile_graph`` so the topology can be inspected
    and unit-tested without triggering the compilation overhead.
    """
    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────
    graph.add_node(_NODE_ROUTER, router_node)
    graph.add_node(_NODE_RAG,    rag_node)
    graph.add_node(_NODE_LLM,    llm_node)
    graph.add_node(_NODE_MEMORY, memory_node)
    graph.add_node(_NODE_EVAL,   eval_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point(_NODE_ROUTER)

    # ── Conditional edge from router ──────────────────────────────────────
    # The mapping tells LangGraph which node to activate for each possible
    # return value of ``_route_decision``.
    graph.add_conditional_edges(
        _NODE_ROUTER,
        _route_decision,
        {
            _NODE_RAG: _NODE_RAG,   # "rag_retrieval" → rag_node
            _NODE_LLM: _NODE_LLM,   # "llm"           → llm_node (direct + out_of_scope)
        },
    )

    # ── Linear edges ──────────────────────────────────────────────────────
    graph.add_edge(_NODE_RAG,    _NODE_LLM)     # retrieval → generation
    graph.add_edge(_NODE_LLM,    _NODE_MEMORY)  # generation → persistence
    graph.add_edge(_NODE_MEMORY, _NODE_EVAL)    # persistence → evaluation
    graph.add_edge(_NODE_EVAL,   END)           # evaluation → graph end

    return graph


def _compile_graph():
    """Compile the StateGraph and return a ``CompiledStateGraph``."""
    graph = _build_graph()
    compiled = graph.compile()
    logger.info(
        "LangGraph agent compiled successfully. "
        "Nodes: %s",
        [_NODE_ROUTER, _NODE_RAG, _NODE_LLM, _NODE_MEMORY, _NODE_EVAL],
    )
    return compiled


# ---------------------------------------------------------------------------
# Module-level singleton — compiled once, shared across all requests
# ---------------------------------------------------------------------------

agent_graph = _compile_graph()
