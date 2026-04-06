from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List

from groq import Groq

from app.core.config import get_settings

"""
Groq LLM client — FALLBACK provider.

Use this when LLM_PROVIDER=groq is set in the environment.  The primary
provider is GeminiLLMService (see app/llm/gemini_client.py), which re-uses
the same google-generativeai SDK already used for embeddings and adds native
multimodal support.  Groq is kept here as a lightweight text-only fallback
(llama-3.3-70b-versatile) for situations where Gemini is unavailable.

This module initializes the official `Groq` client and provides an
async `GroqLLMService.generate_text_response` method which calls
`client.chat.completions.create()` under the hood. Blocking SDK calls
are executed in a thread via `asyncio.to_thread` so the API can be used
from FastAPI endpoints safely.
"""


# System prompt: instructs the model to only use the provided context
# when answering. Grounding the LLM to context reduces hallucinations
# and ensures responses can be traced back to source material — this
# is important in production when factual accuracy and provenance matter.
SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly from the provided "
    "context. Do NOT use any external knowledge beyond what is included "
    "in the context blocks. If the answer is not contained in the context, "
    "respond with 'I don't know based on the provided context.'"
)


class GroqLLMService:
    """Official Groq SDK wrapper.

    Example:
        service = GroqLLMService()
        answer = await service.generate(query, context_chunks)
    """

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.groq.api_key
        model = settings.groq.model
        if not api_key:
            raise RuntimeError("GROQ API key not configured in settings")

        # Initialize the official Groq client
        self._client = Groq(api_key=api_key)
        self._model = model

    def _build_messages(self, query: str, context_chunks: Iterable[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add each context chunk as a user message for clarity and traceability
        for i, c in enumerate(context_chunks, start=1):
            messages.append({"role": "user", "content": f"Context {i}:\n{c}"})

        # Then add the user question
        messages.append({"role": "user", "content": f"Question: {query}"})

        return messages

    def _call_sync(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        # Use the official SDK chat completions API
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Try common response shapes from SDKs
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content:
                        return content

            if isinstance(resp, dict):
                if "choices" in resp and resp["choices"]:
                    c0 = resp["choices"][0]
                    if isinstance(c0, dict) and "message" in c0:
                        return c0["message"].get("content", "")
                if "output" in resp and resp["output"]:
                    out0 = resp["output"][0]
                    if isinstance(out0, dict):
                        return out0.get("content", "")

        except Exception:
            pass

        return str(resp)

    async def generate_text_response(self, query: str, context_chunks: Iterable[str]) -> str:
        """Async wrapper that generates a grounded answer from text context."""
        messages = self._build_messages(query, context_chunks)
        return await asyncio.to_thread(self._call_sync, messages)

    # Backward-compatible alias
    async def generate(self, query: str, context_chunks: Iterable[str]) -> str:
        return await self.generate_text_response(query, context_chunks)

    async def generate_multimodal_response(
        self,
        query: str,
        context_chunks: Iterable[str],
        image_b64_list: list[str],
    ) -> str:
        """Fallback: Groq is text-only so images are ignored.

        Logs a warning so operators know image context is being dropped,
        then delegates to the text-only path.
        """
        import logging
        logging.getLogger(__name__).warning(
            "GroqLLMService does not support image inputs — "
            "%d image(s) will be ignored. Switch LLM_PROVIDER=gemini for vision.",
            len(image_b64_list),
        )
        return await self.generate_text_response(query, context_chunks)


# Module-level singleton
_groq_client: GroqLLMService | None = None


def get_groq_client() -> GroqLLMService:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqLLMService()
    return _groq_client


groq_client = get_groq_client()
