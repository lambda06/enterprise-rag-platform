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

LLM provider
------------
This node uses **Gemini 2.0 Flash** (via ``GeminiLLMService``) as the
primary generation provider.  Gemini's ``response.usage_metadata`` exposes
real token counts (``prompt_token_count``, ``candidates_token_count``) that
are captured here and forwarded to ``agent_service`` for trace-level aggregation.

Output
------
The node always writes ``final_answer`` (str) to the returned state dict.
An empty ``error`` string indicates success; a non-empty value signals that
the OUTPUT node's conditional edge should route to the ERROR node.

Prompts
-------
System prompts and canned refusals are loaded from the Langfuse-backed registry
at runtime (``prompt_registry``), so they can be updated without redeploying.
The RAG path uses ``get_ab_variant`` for session-stable A/B tests. Resolved
prompt version, source, and variant (when applicable) are recorded on the
``llm-generation`` Langfuse span for every trace.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.state import AgentState
from app.core.prompt_registry import prompt_registry
from app.llm.gemini_client import gemini_llm_service
from app.observability.langfuse_tracer import tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference copies — DEPRECATED - now managed via Langfuse prompt registry
# ---------------------------------------------------------------------------
#
# Path 1 — RAG system prompt (rag-answer-generation)
#
# Design decisions:
#   • "ONLY the context blocks below" is emphasised to minimise hallucination.
#   • "Cite [Context N]" forces the model to attribute claims to specific
#     chunks, making answers auditable and trustworthy in enterprise settings.
#   • The "I don't know" clause is explicit so the model doesn't confabulate
#     an answer when the relevant information is absent from the retrieved chunks.
#   • We do NOT tell the model its general knowledge — keeping it fully grounded
#     is a deliberate tradeoff for factual accuracy over completeness.
#
# # RAG_SYSTEM_PROMPT
# """\
# You are a precise document assistant.  Your answers must be grounded EXCLUSIVELY \
# in the context blocks provided below.  Do NOT use any external or general knowledge.
#
# Rules:
#   1. Read ALL context blocks before answering.
#   2. Cite the source of each claim using [Context N] inline (e.g. "According to [Context 1], ...").
#   3. If the answer is not contained in any context block, respond with:
#      "I don't know based on the provided documents."
#   4. Do NOT speculate, infer beyond what is stated, or add information from outside the context.
#   5. Be concise and direct.  Prefer bullet points for multi-part answers.\
# """
#
# Path 2 — Direct answer system prompt (direct-answer-generation)
#
# Design decisions:
#   • Explicitly states "general knowledge" so the model doesn't assume it
#     has documents available (avoids phantom citations).
#   • Keeps the persona consistent ("helpful assistant") while removing the
#     RAG grounding constraint — the model may draw on its parametric memory.
#   • Honesty clause ("say so clearly") mirrors the RAG "I don't know" clause,
#     maintaining a trustworthy user experience across both paths.
#
# # DIRECT_SYSTEM_PROMPT
# """\
# You are a knowledgeable and helpful assistant.  Answer the user's question \
# from your general knowledge.  Be accurate, concise, and honest.  \
# If you are unsure about something, say so clearly rather than guessing.\
# """
#
# Path 3 — Out-of-scope refusal (out-of-scope-refusal; no LLM call)
#
# Design decisions:
#   • Phrased politely and non-judgementally to avoid embarrassing users who
#     asked an ambiguous question that was incorrectly classified.
#   • Does NOT explain *why* the request is out of scope — doing so can
#     inadvertently teach users how to rephrase harmful prompts.
#   • Offers to help with something else to keep the interaction positive.
#
# # OUT_OF_SCOPE_REFUSAL
# (
#     "I'm sorry, but I'm not able to help with that request. "
#     "I'm designed to assist with document-related questions and general knowledge queries. "
#     "Is there something else I can help you with?"
# )

# Maximum tokens for the generated answer.
# 1024 is a reasonable ceiling for enterprise Q&A: long enough for detailed
# answers, short enough to avoid runaway generation costs.
MAX_ANSWER_TOKENS: int = 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rag_context(question: str, chunks: list[dict[str, Any]]) -> tuple[list[str], str]:
    """
    Build the context chunks list and formatted question string for RAG generation.

    Returns:
        (context_strings, question_str) where context_strings is a list of
        formatted chunk strings, one per chunk, and question_str is the plain
        question ready to append.
    """
    context_strings: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})

        source = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("file_path")
            or "Unknown source"
        )
        page = meta.get("page") or meta.get("page_number")
        source_label = f"{source}, page {page}" if page else source

        context_strings.append(f"[Context {i}] (Source: {source_label})\n{text}")

    return context_strings, question


# ---------------------------------------------------------------------------
# Gemini sync callers (dispatched to thread pool)
# ---------------------------------------------------------------------------

def _call_gemini_rag_sync(
    question: str,
    chunks: list[dict[str, Any]],
    system_instruction: str,
    image_b64_list: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """
    Blocking Gemini call for RAG path — returns (answer, usage).

    Uses a custom system prompt that restricts the model to citing only the
    provided context blocks.  Calls ``_call_text_sync_with_usage`` which
    returns real Gemini token counts from ``response.usage_metadata``.
    """
    from google.genai import types as gtypes

    # Override system instruction for RAG grounding
    config = gtypes.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=MAX_ANSWER_TOKENS,
    )

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})
        source = (
            meta.get("source") or meta.get("filename")
            or meta.get("file_name") or meta.get("file_path") or "Unknown source"
        )
        page = meta.get("page") or meta.get("page_number")
        source_label = f"{source}, page {page}" if page else source
        parts.append(f"[Context {i}] (Source: {source_label})\n{text}")
    
    prompt = "\n\n".join(parts)
    contents: list[Any] = [prompt]

    if image_b64_list:
        import base64
        for idx, b64_str in enumerate(image_b64_list):
            try:
                image_bytes = base64.b64decode(b64_str)
                contents.append(
                    gtypes.Part.from_bytes(data=image_bytes, mime_type="image/png")
                )
            except Exception as exc:
                logger.warning("Skipping image %d in llm_node — decode error: %s", idx + 1, exc)

    contents.append(f"\n\nQuestion: {question}")

    from app.llm.gemini_client import gemini_llm_service as svc
    response = svc._client.models.generate_content(  # noqa: SLF001
        model=svc._model_name,  # noqa: SLF001
        contents=contents,
        config=config,
    )
    usage = svc._parse_usage(getattr(response, "usage_metadata", None))  # noqa: SLF001
    return response.text or "", usage


def _call_gemini_direct_sync(question: str, system_instruction: str) -> tuple[str, dict[str, int]]:
    """
    Blocking Gemini call for direct (no-retrieval) path — returns (answer, usage).

    ``system_instruction`` comes from the registry (direct-answer-generation)
    so the model may use parametric knowledge rather than context blocks.
    """
    from google.genai import types as gtypes
    from app.llm.gemini_client import gemini_llm_service as svc

    config = gtypes.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=MAX_ANSWER_TOKENS,
    )
    response = svc._client.models.generate_content(  # noqa: SLF001
        model=svc._model_name,  # noqa: SLF001
        contents=question,
        config=config,
    )
    usage = svc._parse_usage(getattr(response, "usage_metadata", None))  # noqa: SLF001
    return response.text or "", usage


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------

async def llm_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: generate the final answer via one of three paths.

    Path selection logic:
      1. routing_decision == "out_of_scope"  → return polite refusal (no LLM)
      2. reranked_chunks non-empty           → RAG-grounded answer (Gemini)
      3. otherwise                           → direct parametric answer (Gemini)

    Token tracking
    --------------
    Gemini's ``response.usage_metadata`` returns real token counts
    (``prompt_token_count`` and ``candidates_token_count``).  We extract
    these and compute two additional proxy metrics from character counts:

      context_chars   — total characters across all retrieved chunk texts
      question_chars  — characters in the user's question string alone

    The ``context_chars / question_chars`` ratio reveals how much of the
    prompt window is consumed by retrieval output vs the question itself.
    A ratio > 10:1 suggests chunks are too large; consider reducing
    ``chunk_size`` in the ingestion pipeline to improve prompt efficiency.

    All token data is written to ``state["token_usage"]`` so that
    ``AgentService`` can aggregate a ``total_request_tokens`` into the
    top-level Langfuse trace metadata.

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

        ``token_usage``
            Dict with keys ``input_tokens``, ``output_tokens``,
            ``total_tokens``, ``context_chars``, ``question_chars``.
            All zeros for the out_of_scope path (no LLM call).

        ``error``
            Empty string on success; human-readable message on failure.

    Raises:
        Does NOT raise — all exceptions are caught and written to ``error``.
    """
    # ------------------------------------------------------------------ #
    # Upstream error propagation                                          #
    # ------------------------------------------------------------------ #
    upstream_error: str = state.get("error", "")
    if upstream_error and not state.get("current_question", "").strip():
        logger.warning(
            "llm_node: upstream error detected with no question — skipping generation."
        )
        return {"final_answer": "", "token_usage": {}, "error": upstream_error}

    routing_decision: str = state.get("routing_decision", "")
    question: str = state.get("current_question", "").strip()
    lf_trace = state.get("lf_trace")

    # ------------------------------------------------------------------ #
    # Path 3 — Out of scope: return refusal immediately, no LLM call     #
    # ------------------------------------------------------------------ #
    if routing_decision == "out_of_scope":
        logger.info("llm_node: out_of_scope — returning polite refusal without LLM call.")
        refusal = prompt_registry.get_prompt("out-of-scope-refusal")
        return {
            "final_answer": refusal.text,
            "token_usage": {},
            "error": "",
        }

    if not question:
        msg = "llm_node received an empty current_question; cannot generate answer."
        logger.warning(msg)
        return {"final_answer": "", "token_usage": {}, "error": msg}

    # ------------------------------------------------------------------ #
    # Determine context: prefer reranked_chunks, fall back to retrieved   #
    # ------------------------------------------------------------------ #
    chunks: list[dict[str, Any]] = (
        state.get("reranked_chunks")
        or state.get("retrieved_chunks")
        or []
    )

    # Pre-compute character proxies (before the async call so they appear in
    # the span input even if generation fails).
    context_chars = sum(len(c.get("text", "")) for c in chunks)
    question_chars = len(question)
    path = "rag" if chunks else "direct"

    # Unique content types present in context (e.g. ['text', 'image', 'table']).
    # Used in the span to show whether the LLM received multimodal input.
    content_types_in_context: list[str] = list({
        c.get("metadata", {}).get("content_type", "text")
        for c in chunks
    })
    multimodal_used: bool = any(
        c.get("metadata", {}).get("content_type") == "image"
        for c in chunks
    )

    if chunks:
        # A/B test concluded — Variant A (version 1) won on answer_relevancy
        # and faithfulness. All traffic now uses version 1 directly.
        prompt_res = prompt_registry.get_prompt("rag-answer-generation", version=1)
    else:
        prompt_res = prompt_registry.get_prompt("direct-answer-generation")

    # Open the llm-generation span BEFORE the Gemini call so latency is captured.
    llm_span = tracer.start_generation(
        lf_trace,
        "llm-generation",
        model=gemini_llm_service._model_name,  # noqa: SLF001
        model_parameters={"temperature": 0.0, "max_output_tokens": MAX_ANSWER_TOKENS},
        input={
            "path":                   path,
            "question":               question[:200],
            "chunk_count":            len(chunks),
            "context_chars":          context_chars,
            "question_chars":         question_chars,
            "content_types_in_context": content_types_in_context,
            "multimodal_used":         multimodal_used,
        },
        metadata={
            "prompt_version": prompt_res.version,
            "prompt_source": prompt_res.source,
            "prompt_variant": prompt_res.variant,
        },
    )

    try:
        if chunks:
            # ── Path 1: RAG-grounded answer ──────────────────────────────
            image_b64_list = state.get("image_b64_list") or []
            logger.info(
                "llm_node: RAG path (Gemini) — %d context chunks + %d image(s) for question=%r",
                len(chunks),
                len(image_b64_list),
                question[:80],
            )
            answer, usage = await asyncio.to_thread(
                _call_gemini_rag_sync, question, chunks, prompt_res.text, image_b64_list
            )

        else:
            # ── Path 2: Direct answer from parametric knowledge ──────────
            logger.info(
                "llm_node: direct path (Gemini) — no context chunks for question=%r",
                question[:80],
            )
            answer, usage = await asyncio.to_thread(
                _call_gemini_direct_sync, question, prompt_res.text
            )

        # ── Assemble token metadata ──────────────────────────────────────
        token_usage: dict[str, Any] = {
            **usage,                          # input_tokens, output_tokens, total_tokens
            "context_chars":  context_chars,  # chars consumed by retrieved chunks
            "question_chars": question_chars, # chars from the user's question
        }

        logger.info(
            "llm_node: generated answer (%d chars) | tokens in=%d out=%d total=%d "
            "context_chars=%d question_chars=%d",
            len(answer),
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("total_tokens", 0),
            context_chars,
            question_chars,
        )

        tracer.end_span(
            llm_span,
            output={
                # Real Gemini token counts from response.usage_metadata.
                "input_tokens":  usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens":  usage.get("total_tokens", 0),
                # context_chars vs question_chars shows how much of the prompt
                # window is consumed by retrieval vs the user's own question.
                # A ratio > 10:1 suggests chunks are too large — reduce chunk_size
                # in the ingestion pipeline to improve prompt efficiency.
                "context_chars":  context_chars,
                "question_chars": question_chars,
                # token_efficiency = output_tokens / input_tokens.
                # Low values mean the model is consuming many tokens to produce
                # a short answer — could indicate prompt bloat.
                "token_efficiency": round(
                    usage.get("output_tokens", 0) / usage.get("input_tokens", 1),
                    3,
                ),
            },
        )

        return {
            "final_answer": answer.strip(),
            "token_usage":  token_usage,
            "error":        "",
        }

    except Exception as exc:  # noqa: BLE001
        msg = f"llm_node generation failed: {exc}"
        logger.exception(msg)
        tracer.end_span(llm_span, output={"error": msg})
        return {
            "final_answer": "",
            "token_usage":  {},
            "error":        msg,
        }
