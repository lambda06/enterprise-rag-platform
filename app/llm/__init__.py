"""
LLM provider factory for the Enterprise RAG Platform.

Reads ``LLM_PROVIDER`` from settings (default: ``"gemini"``) and returns
the appropriate singleton LLM service.  Currently supported values:

  - ``"gemini"`` → :class:`~app.llm.gemini_client.GeminiLLMService`
         Uses Google Gemini 2.0 Flash via the ``google-generativeai`` SDK.
         Primary provider — same SDK/API key as embeddings.
  - ``"groq"``   → :class:`~app.llm.groq_client.GroqLLMService`
         Fallback option using the Groq SDK (llama-3.3-70b-versatile).

Usage::

    from app.llm import get_llm_service

    llm = get_llm_service()
    answer = await llm.generate_text_response(question, context_chunks)
    # or — for multimodal:
    answer = await llm.generate_multimodal_response(question, chunks, images)
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_llm_service():
    """Return the configured LLM service singleton.

    Reads ``LLM_PROVIDER`` env var (via ``Settings.llm_provider``).
    Defaults to Gemini if the var is unset or unrecognised.

    Returns:
        GeminiLLMService or GroqLLMService instance.
    """
    settings = get_settings()
    provider = getattr(settings, "llm_provider", "gemini").lower().strip()

    if provider == "groq":
        logger.info("LLM provider: groq (fallback)")
        from app.llm.groq_client import get_groq_client
        return get_groq_client()

    if provider != "gemini":
        logger.warning(
            "Unknown LLM_PROVIDER=%r, falling back to gemini.", provider
        )

    logger.info("LLM provider: gemini (primary)")
    from app.llm.gemini_client import get_gemini_llm_service
    return get_gemini_llm_service()
