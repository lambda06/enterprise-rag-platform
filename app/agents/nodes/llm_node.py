"""
LangGraph LLM_NODE for the Enterprise RAG Platform.

Responsibility
--------------
This node is the final generation step.  It handles three distinct paths:

  Path 1 — RAG answer      (reranked_chunks or retrieved_chunks is non-empty)
      Format retrieved chunks as numbered context blocks, instruct the LLM
      to answer *only* from context, cite which chunks it used, and say
      "I don't know" if the answer is not in the context.

  Path 2 — Direct answer   (reranked_chunks is empty or routing_decision == "direct")
      Answer the question directly from general/parametric LLM knowledge.
      No document context is injected.

  Path 3 — Out of scope    (routing_decision == "out_of_scope")
      Return a polite refusal string immediately — no LLM call is made.
      This is intentional: calling the LLM for harmful/irrelevant requests
      wastes tokens and risks generating unsafe content even with guardrails.

Output
------
The node always writes ``final_answer`` (str) to the returned state dict.
An empty ``error`` string indicates success; a non-empty value signals that
the OUTPUT node's conditional edge should route to the ERROR node.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.state import AgentState
from app.llm.groq_client import groq_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level prompt constants
# ---------------------------------------------------------------------------

# Path 1 — RAG system prompt
#
# Design decisions:
#   • "ONLY the context blocks below" is emphasised to minimise hallucination.
#   • "Cite [Context N]" forces the model to attribute claims to specific
#     chunks, making answers auditable and trustworthy in enterprise settings.
#   • The "I don't know" clause is explicit so the model doesn't confabulate
#     an answer when the relevant information is absent from the retrieved chunks.
#   • We do NOT tell the model its general knowledge — keeping it fully grounded
#     is a deliberate tradeoff for factual accuracy over completeness.
RAG_SYSTEM_PROMPT = """\
You are a precise document assistant.  Your answers must be grounded EXCLUSIVELY \
in the context blocks provided below.  Do NOT use any external or general knowledge.

Rules:
  1. Read ALL context blocks before answering.
  2. Cite the source of each claim using [Context N] inline (e.g. "According to [Context 1], ...").
  3. If the answer is not contained in any context block, respond with:
     "I don't know based on the provided documents."
  4. Do NOT speculate, infer beyond what is stated, or add information from outside the context.
  5. Be concise and direct.  Prefer bullet points for multi-part answers.\
"""

# Path 2 — Direct answer system prompt
#
# Design decisions:
#   • Explicitly states "general knowledge" so the model doesn't assume it
#     has documents available (avoids phantom citations).
#   • Keeps the persona consistent ("helpful assistant") while removing the
#     RAG grounding constraint — the model may draw on its parametric memory.
#   • Honesty clause ("say so clearly") mirrors the RAG "I don't know" clause,
#     maintaining a trustworthy user experience across both paths.
DIRECT_SYSTEM_PROMPT = """\
You are a knowledgeable and helpful assistant.  Answer the user's question \
from your general knowledge.  Be accurate, concise, and honest.  \
If you are unsure about something, say so clearly rather than guessing.\
"""

# Path 3 — Out-of-scope refusal (no LLM call; returned verbatim)
#
# Design decisions:
#   • Phrased politely and non-judgementally to avoid embarrassing users who
#     asked an ambiguous question that was incorrectly classified.
#   • Does NOT explain *why* the request is out of scope — doing so can
#     inadvertently teach users how to rephrase harmful prompts.
#   • Offers to help with something else to keep the interaction positive.
OUT_OF_SCOPE_REFUSAL = (
    "I'm sorry, but I'm not able to help with that request. "
    "I'm designed to assist with document-related questions and general knowledge queries. "
    "Is there something else I can help you with?"
)

# Maximum tokens for the generated answer.
# 1024 is a reasonable ceiling for enterprise Q&A: long enough for detailed
# answers, short enough to avoid runaway generation costs.
MAX_ANSWER_TOKENS: int = 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rag_messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Build the Groq message list for a RAG-grounded answer.

    Formats each chunk as a clearly delimited context block so the model can
    reference them by number.  Metadata (source, page) is appended inline to
    allow the model to include useful citations without hallucinating file paths.

    Args:
        question: The user's raw question string.
        chunks:   Reranked chunks — each a dict with ``"text"`` and ``"metadata"`` keys.

    Returns:
        A list of role/content message dicts ready for the Groq chat API.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
    ]

    # Each chunk becomes its own user message so the model processes them as
    # discrete, citable units rather than one undifferentiated blob of text.
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})

        # Build a compact source label from whatever metadata is available.
        source = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("file_path")
            or "Unknown source"
        )
        page = meta.get("page") or meta.get("page_number")
        source_label = f"{source}, page {page}" if page else source

        block = f"[Context {i}] (Source: {source_label})\n{text}"
        messages.append({"role": "user", "content": block})

    # Final user turn: the actual question.
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


def _build_direct_messages(question: str) -> list[dict[str, str]]:
    """
    Build the Groq message list for a direct (no-retrieval) answer.

    Args:
        question: The user's raw question string.

    Returns:
        A minimal two-message list: system prompt + user question.
    """
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------

async def llm_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: generate the final answer via one of three paths.

    Path selection logic:
      1. routing_decision == "out_of_scope"  → return polite refusal (no LLM)
      2. reranked_chunks non-empty           → RAG-grounded answer
      3. otherwise                           → direct parametric answer

    Note: ``reranked_chunks`` takes priority over ``retrieved_chunks`` because
    the RERANKER node (if present) produces higher-quality ordering.  If only
    ``retrieved_chunks`` is populated (RERANKER was skipped), this node falls
    back gracefully by checking both.

    Args:
        state: The current ``AgentState`` dict.  Reads:
               - ``routing_decision``   — to detect out_of_scope early.
               - ``current_question``   — the raw user question.
               - ``reranked_chunks``    — preferred context source.
               - ``retrieved_chunks``   — fallback context source.

    Returns:
        Partial state dict:

        ``final_answer``
            The generated (or canned) answer string.  Never empty on success.

        ``error``
            Empty string on success; human-readable message on failure.

    Raises:
        Does NOT raise — all exceptions are caught and written to ``error``.
    """
    # ------------------------------------------------------------------ #
    # Upstream error propagation                                          #
    # ------------------------------------------------------------------ #
    # If a critical upstream node (router, rag) set an error AND left
    # current_question empty, there is nothing meaningful to generate.
    # Pass the error through rather than producing a blank answer.
    upstream_error: str = state.get("error", "")
    if upstream_error and not state.get("current_question", "").strip():
        logger.warning(
            "llm_node: upstream error detected with no question — skipping generation."
        )
        return {"final_answer": "", "error": upstream_error}

    routing_decision: str = state.get("routing_decision", "")
    question: str = state.get("current_question", "").strip()

    # ------------------------------------------------------------------ #
    # Path 3 — Out of scope: return refusal immediately, no LLM call     #
    # ------------------------------------------------------------------ #
    if routing_decision == "out_of_scope":
        logger.info("llm_node: out_of_scope — returning polite refusal without LLM call.")
        return {
            "final_answer": OUT_OF_SCOPE_REFUSAL,
            "error": "",
        }

    if not question:
        msg = "llm_node received an empty current_question; cannot generate answer."
        logger.warning(msg)
        return {"final_answer": "", "error": msg}

    # ------------------------------------------------------------------ #
    # Determine context: prefer reranked_chunks, fall back to retrieved   #
    # ------------------------------------------------------------------ #
    chunks: list[dict[str, Any]] = (
        state.get("reranked_chunks")
        or state.get("retrieved_chunks")
        or []
    )

    try:
        if chunks:
            # ── Path 1: RAG-grounded answer ──────────────────────────────
            logger.info(
                "llm_node: RAG path — %d context chunks for question=%r",
                len(chunks),
                question[:80],
            )
            messages = _build_rag_messages(question, chunks)

        else:
            # ── Path 2: Direct answer from parametric knowledge ──────────
            logger.info(
                "llm_node: direct path — no context chunks for question=%r",
                question[:80],
            )
            messages = _build_direct_messages(question)

        # Blocking Groq SDK call dispatched to a thread pool so the event
        # loop remains free (same pattern as router_node and rag_node).
        answer: str = await asyncio.to_thread(
            groq_client._call_sync,  # noqa: SLF001
            messages,
            MAX_ANSWER_TOKENS,
        )

        logger.info(
            "llm_node: generated answer (%d chars) for question=%r",
            len(answer),
            question[:80],
        )

        return {
            "final_answer": answer.strip(),
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001
        msg = f"llm_node generation failed: {exc}"
        logger.exception(msg)
        return {
            "final_answer": "",
            "error": msg,
        }
