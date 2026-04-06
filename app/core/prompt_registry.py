"""
Central registry for production prompts with Langfuse, in-process cache, and hardcoded fallbacks.

**5-minute cache TTL — why it matters**

We balance freshness against Langfuse API load and latency. Prompts change infrequently compared to
request volume; caching avoids a remote round-trip on every turn while still picking up new
``production`` versions within a few minutes after publish. A shorter TTL would increase API calls
and tail latency; a longer TTL would delay rollout visibility. Five minutes is a practical default
for admin-driven prompt updates without hammering Langfuse.

**Session-stable A/B assignment (MD5 of ``session_id``)**

``get_ab_variant`` hashes ``session_id`` and maps the first hex digit to variant A or B. The same
``session_id`` always yields the same branch, so a user does not flip between experiments mid-session
(which would confuse both UX and metrics). New sessions get a pseudo-random 50/50 split across the
hash space.

**Local fallback is non-negotiable in production**

Langfuse may be misconfigured, rate-limited, or unreachable. Shipping prompts only from a remote
registry would take down answering and routing when Langfuse fails. Bundling the four baseline
prompts guarantees the platform keeps serving safe defaults with predictable behavior while
operators restore connectivity or fix credentials.

**``PromptResult.source`` — debugging semantics**

- ``cache``: Returned from the in-memory TTL cache (no Langfuse call this hit). Use to confirm
  caching is working and to rule out remote failures.
- ``langfuse``: Fetched from Langfuse and then cached. Confirms the live registry was reachable
  for this resolution path.
- ``fallback``: Langfuse fetch failed or prompt missing remotely; bundled text was used. Investigate
  credentials, network, prompt name/version, or Langfuse project state when you see this in logs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from langfuse import Langfuse

from app.core.config import get_settings
from app.core.prompt_fallbacks import (
    DIRECT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REFUSAL,
    RAG_SYSTEM_PROMPT,
    ROUTER_CLASSIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)

PromptSource = Literal["cache", "langfuse", "fallback"]

# Align with ``get_prompt`` cache key convention: ``None`` version means latest production label.
_LATEST = "latest"
_CACHE_TTL = timedelta(minutes=5)


@dataclass(frozen=True)
class PromptResult:
    """Resolved prompt text plus provenance for observability and A/B testing."""

    text: str
    version: int
    source: PromptSource
    variant: Optional[str] = None  # 'A', 'B', or None when not in an A/B path


class PromptRegistry:
    """
    Loads prompts from Langfuse (production label or explicit version), with TTL cache and
    hardcoded fallbacks mirroring ``app.core.prompt_fallbacks``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._langfuse = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )
        # Keeps cached prompts: `{"prompt_name:version": (PromptResult, cache_timestamp)}`
        self._cache: dict[str, tuple[PromptResult, datetime]] = {}
        
        # Hardcoded default prompts if Langfuse is unreachable
        self._fallback: dict[str, str] = {
            "router-classification": ROUTER_CLASSIFICATION_PROMPT,
            "rag-answer-generation": RAG_SYSTEM_PROMPT,
            "direct-answer-generation": DIRECT_SYSTEM_PROMPT,
            "out-of-scope-refusal": OUT_OF_SCOPE_REFUSAL,
        }

    def _cache_key(self, prompt_name: str, version: int | None) -> str:
        return f"{prompt_name}:{version if version is not None else _LATEST}"

    def _get_from_fallback(self, prompt_name: str) -> PromptResult:
        """Returns the hardcoded prompt text when Langfuse is unavailable."""
        text = self._fallback.get(prompt_name)
        if text is None:
            logger.warning("Prompt fallback missing for name=%r", prompt_name)
        return PromptResult(text=text or "", version=0, source="fallback")

    @staticmethod
    def _langfuse_prompt_to_text_and_version(lf_prompt: Any, requested_version: int | None) -> tuple[str, int]:
        """Extracts plain text and integer version from Langfuse prompt objects."""
        prompt_content = getattr(lf_prompt, "prompt", lf_prompt)
        
        # If it's a chat prompt (list of dictionaries representing roles/messages)
        if isinstance(prompt_content, list):
            parts = []
            for item in prompt_content:
                # Extract 'content' whether it's a dict or object
                content = item.get("content", item) if isinstance(item, dict) else getattr(item, "content", item)
                parts.append(str(content))
            text = "\n".join(parts)
        else:
            text = str(prompt_content)

        # Resolve the version (defaulting to 0 if unknown)
        ver = getattr(lf_prompt, "version", requested_version)
        version_out = int(ver) if ver is not None else 0
        
        return text, version_out

    def get_prompt(self, prompt_name: str, version: int | None = None) -> PromptResult:
        """Fetch prompt from cache, or if expired, from Langfuse, with local fallback."""
        key = self._cache_key(prompt_name, version)
        now = datetime.now(timezone.utc)

        # 1. Return from cache if it hasn't expired
        if key in self._cache:
            result, cached_at = self._cache[key]
            if now - cached_at < _CACHE_TTL:
                return PromptResult(
                    text=result.text, version=result.version, source="cache", variant=result.variant
                )

        # 2. Otherwise fetch freshly from Langfuse
        try:
            if version is not None:
                lf_prompt = self._langfuse.get_prompt(name=prompt_name, version=version)
            else:
                lf_prompt = self._langfuse.get_prompt(name=prompt_name, label="production")

            text, ver = self._langfuse_prompt_to_text_and_version(lf_prompt, version)

            logger.info(
                "PromptRegistry: fetched %r v%s from langfuse",
                prompt_name,
                ver,
            )

            # Save into cache and return
            out = PromptResult(text=text, version=ver, source="langfuse")
            self._cache[key] = (out, now)
            return out
            
        except Exception as exc:
            logger.warning("Langfuse fetch failed for %r (v=%r): %s — using local fallback.", prompt_name, version, exc)
            return self._get_from_fallback(prompt_name)

    def get_ab_variant(
        self, prompt_name: str, session_id: str, version_a: int, version_b: int
    ) -> PromptResult:
        """Consistently pick either version A or B for a user's session over time."""
        digest = hashlib.md5(session_id.encode()).hexdigest()
        is_variant_a = digest[0] in "01234567" # First Hex char '0'-'7' -> 50% split
        
        chosen_version = version_a if is_variant_a else version_b
        result = self.get_prompt(prompt_name, version=chosen_version)
        
        return PromptResult(
            text=result.text, 
            version=result.version, 
            source=result.source, 
            variant="A" if is_variant_a else "B"
        )

    def invalidate_cache(self, prompt_name: str | None = None) -> None:
        """Clear cache completely, or for a specific prompt."""
        if prompt_name is None:
            self._cache.clear()
        else:
            prefix = f"{prompt_name}:"
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(prefix)}


prompt_registry = PromptRegistry()
