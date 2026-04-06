from __future__ import annotations

import asyncio
import base64
import logging
from typing import Iterable, List

from google import genai
from google.genai import types

from app.core.config import get_settings

"""
Gemini 2.0 Flash LLM client — primary generation provider.

Uses the ``google-genai`` SDK (``from google import genai``) — the same library
already used for embeddings in ``app/rag/embeddings.py'

Two public async methods are exposed:

  generate_text_response(question, context_chunks)
      Pure text generation: system prompt + context chunks + question.

  generate_multimodal_response(question, context_chunks, image_b64_list)
      Single multimodal request combining text context AND inline images.

Why one multimodal request beats calling a vision model per image
-----------------------------------------------------------------
Gemini 2.0 Flash supports mixed text+image parts in a single request.
Sending all retrieved images together with all text chunks in *one* call
means the model has cross-attention over the full context simultaneously —
it can reason about relationships between charts, tables, and prose in a
single forward pass.  Calling a separate vision model for each image and
then concatenating the captions loses that cross-modal coherence, and
adds N−1 extra API round-trips (latency × N).  The single-multimodal-call
approach is both faster and produces more coherent, grounded answers.
"""

logger = logging.getLogger(__name__)

# System prompt: instructs the model to only use the provided context.
SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly from the provided "
    "context. Do NOT use any external knowledge beyond what is included "
    "in the context blocks. If the answer is not contained in the context, "
    "respond with 'I don't know based on the provided context.'"
)


class GeminiLLMService:
    """Google Gemini 2.0 Flash wrapper for text and multimodal generation.

    Uses the same ``genai.Client`` pattern as ``EmbeddingService`` so the
    platform only pays for one authenticated Google client instance style.

    Example::

        service = GeminiLLMService()
        # Text-only
        answer = await service.generate_text_response(query, context_chunks)
        # Multimodal (text + images)
        answer = await service.generate_multimodal_response(
            query, context_chunks, image_b64_list
        )
    """

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.gemini.api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not configured in settings")

        self._client = genai.Client(api_key=api_key)
        self._model_name = settings.gemini.generation_model
        logger.info("GeminiLLMService initialised with model: %s", self._model_name)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _generation_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=1024,
        )

    @staticmethod
    def _parse_usage(usage_metadata: object | None) -> dict[str, int]:
        """Extract token counts from Gemini response usage_metadata.

        Gemini's ``usage_metadata`` exposes:
          - ``prompt_token_count``      — tokens consumed by the prompt (input)
          - ``candidates_token_count``  — tokens in the generated response (output)
          - ``total_token_count``       — sum of both

        Returns a normalised dict with keys ``input_tokens``, ``output_tokens``,
        and ``total_tokens``.  All values default to 0 if the attribute is absent
        (e.g. when Gemini does not return usage data for a given model tier).
        """
        if usage_metadata is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_t  = getattr(usage_metadata, "prompt_token_count",     0) or 0
        output_t = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_t  = getattr(usage_metadata, "total_token_count",      0) or (input_t + output_t)
        return {
            "input_tokens":  input_t,
            "output_tokens": output_t,
            "total_tokens":  total_t,
        }

    def _call_text_sync(
        self, question: str, context_chunks: Iterable[str]
    ) -> str:
        """Blocking text-only generation call (returns content string only)."""
        content, _ = self._call_text_sync_with_usage(question, context_chunks)
        return content

    def _call_text_sync_with_usage(
        self, question: str, context_chunks: Iterable[str]
    ) -> tuple[str, dict[str, int]]:
        """Blocking text-only generation call that also returns token usage.

        Returns:
            (content, usage) where usage has keys
            ``input_tokens``, ``output_tokens``, ``total_tokens``.
            All integers.  Falls back to zeros if the API returns no usage data.
        """
        parts: list[str] = []
        for i, chunk in enumerate(context_chunks, start=1):
            parts.append(f"Context {i}:\n{chunk}")
        parts.append(f"Question: {question}")
        prompt = "\n\n".join(parts)

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._generation_config(),
            )
            usage = self._parse_usage(getattr(response, "usage_metadata", None))
            return response.text or "", usage
        except Exception as exc:
            logger.exception("Gemini text generation failed: %s", exc)
            raise

    def _call_classification_sync_with_usage(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: int = 10,
    ) -> tuple[str, dict[str, int]]:
        """Blocking classification call for the router — returns (label, usage).

        Sends a minimal contents list built from ``messages`` dicts
        (role/content pairs) as a single concatenated prompt to match
        the Groq message-list pattern used by router_node.

        Args:
            messages:          List of {"role": str, "content": str} dicts.
            max_output_tokens: Upper bound on the generated label (default 10).

        Returns:
            (raw_label, usage_dict) — label is the raw model text (usually
            one word); usage_dict has ``input_tokens``, ``output_tokens``,
            ``total_tokens``.
        """
        # Flatten the message list into a single prompt string so we can
        # use the unified generate_content API without function-calling or
        # chat sessions (which are unnecessarily complex for one-shot classification).
        combined = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=combined,
                config=config,
            )
            usage = self._parse_usage(getattr(response, "usage_metadata", None))
            return response.text or "", usage
        except Exception as exc:
            logger.exception("Gemini classification call failed: %s", exc)
            raise

    def _call_multimodal_sync(
        self,
        question: str,
        context_chunks: Iterable[str],
        image_b64_list: List[str],
    ) -> str:
        """Blocking multimodal generation call (text + inline images).

        All images and text contexts are bundled into a single content list so
        Gemini can reason over them jointly in one forward pass.
        """
        content, _ = self._call_multimodal_sync_with_usage(question, context_chunks, image_b64_list)
        return content

    def _call_multimodal_sync_with_usage(
        self,
        question: str,
        context_chunks: Iterable[str],
        image_b64_list: List[str],
    ) -> tuple[str, dict[str, int]]:
        """Blocking multimodal generation call that also returns token usage.

        Returns:
            (content, usage) — same usage dict shape as
            ``_call_text_sync_with_usage``.
        """
        # Build a contents list: text chunks first, then images, then question.
        contents: list = []

        for i, chunk in enumerate(context_chunks, start=1):
            contents.append(types.Part.from_text(text=f"Context {i}:\n{chunk}"))

        for idx, b64_str in enumerate(image_b64_list):
            try:
                image_bytes = base64.b64decode(b64_str)
                contents.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                )
                logger.debug(
                    "Added image %d to multimodal request (%d bytes)",
                    idx + 1,
                    len(image_bytes),
                )
            except Exception as exc:
                logger.warning("Skipping image %d — decode error: %s", idx + 1, exc)

        contents.append(types.Part.from_text(text=f"Question: {question}"))

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._generation_config(),
            )
            usage = self._parse_usage(getattr(response, "usage_metadata", None))
            return response.text or "", usage
        except Exception as exc:
            logger.exception("Gemini multimodal generation failed: %s", exc)
            raise

    # ── Public async API ───────────────────────────────────────────────────────

    async def generate_text_response(
        self, question: str, context_chunks: Iterable[str]
    ) -> str:
        """Generate a grounded answer from text/table context chunks only."""
        return await asyncio.to_thread(self._call_text_sync, question, list(context_chunks))

    async def generate_multimodal_response(
        self,
        question: str,
        context_chunks: Iterable[str],
        image_b64_list: List[str],
    ) -> str:
        """Generate a grounded answer using text context AND inline images.

        Falls back to text-only generation if ``image_b64_list`` is empty.
        """
        chunks = list(context_chunks)
        if not image_b64_list:
            return await asyncio.to_thread(self._call_text_sync, question, chunks)
        return await asyncio.to_thread(
            self._call_multimodal_sync, question, chunks, image_b64_list
        )

    # Convenience alias so this class matches the GroqLLMService interface
    async def generate(self, query: str, context_chunks: Iterable[str]) -> str:
        """Alias for generate_text_response (backward-compatible)."""
        return await self.generate_text_response(query, context_chunks)


# ── Module-level singleton ─────────────────────────────────────────────────────
_gemini_llm_service: GeminiLLMService | None = None


def get_gemini_llm_service() -> GeminiLLMService:
    global _gemini_llm_service
    if _gemini_llm_service is None:
        _gemini_llm_service = GeminiLLMService()
    return _gemini_llm_service


gemini_llm_service = get_gemini_llm_service()
