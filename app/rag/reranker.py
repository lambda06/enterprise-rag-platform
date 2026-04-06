"""
Cross-encoder reranking via the Jina Rerank API.

Replaces the previous sentence-transformers / PyTorch CrossEncoder with an
HTTP call to https://api.jina.ai/v1/rerank.  The public interface is identical:

    reranker_service.rerank(query, candidates, top_k) -> list[dict]

so retrieval.py and all callers require zero changes.

Why Jina?
---------
* Free tier: 1 million tokens / month — sufficient for development and
  light production traffic.
* No local model weights: removes torch / transformers / sentence-transformers
  (~1.4 GB) from the Docker image, taking the image from ~2.5 GB → ~500 MB.
* No RAM at startup: the container no longer loads a neural network on import,
  so memory footprint drops from ~1.2 GB → ~300 MB, fitting GCP Cloud Run's
  free tier (256 MiB minimum, configurable up to 32 GiB).

Fallback behaviour
------------------
If JINA_API_KEY is missing or the API call fails, the service falls back to
returning the first top_k candidates in their original hybrid-search (RRF)
order — identical to the previous CrossEncoder fallback path.

Environment variables
---------------------
JINA_API_KEY        Required. Get a free key at https://jina.ai/
JINA_RERANKER_MODEL Optional. Default: jina-reranker-v2-base-multilingual
"""

import logging
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_JINA_API_URL = "https://api.jina.ai/v1/rerank"
_REQUEST_TIMEOUT = 30  # seconds


class RerankerService:
    """Cross-encoder reranking via the Jina Rerank HTTP API.

    Loaded once as a module-level singleton.  Each ``rerank()`` call issues
    a single HTTPS POST with all (query, document) pairs; Jina returns
    relevance scores which we use to re-sort the candidates.

    If JINA_API_KEY is unset or the API is unreachable, ``rerank()`` degrades
    gracefully to returning candidates in their original RRF order.
    """

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.jina.api_key.strip()
        self._model: str = settings.jina.reranker_model
        self._available: bool = bool(api_key)

        if self._available:
            self._headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            logger.info(
                "RerankerService initialised: provider=jina model=%s", self._model
            )
        else:
            self._headers = {}
            logger.warning(
                "JINA_API_KEY not set — reranker disabled. "
                "Retrieval will use hybrid-search (RRF) ranking only. "
                "Set JINA_API_KEY to enable cross-encoder reranking."
            )

    # ── Public API (identical contract to the previous CrossEncoder version) ──

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score and re-sort candidates using the Jina Rerank API (synchronous).

        If the API is unavailable (missing key or network error), falls back to
        returning the first ``top_k`` candidates in their original RRF order.

        Args:
            query:      The raw user query string.
            candidates: List of dicts, each with at least a ``"text"`` key.
            top_k:      Number of results to return after reranking.

        Returns:
            Top ``top_k`` dicts from ``candidates``, re-sorted by Jina relevance
            score (most relevant first), or by original RRF order on fallback.
        """
        if not candidates:
            return []

        limit = min(len(candidates), top_k)

        if not self._available:
            logger.debug(
                "Reranker unavailable (no API key) — returning top-%d by RRF order.",
                top_k,
            )
            return [candidates[i] for i in range(limit)]

        # Build the Jina API request payload.
        # The API accepts a list of plain strings as "documents"; we extract
        # the "text" field from each candidate dict.
        documents = [hit.get("text", "") for hit in candidates]

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": limit,
        }

        try:
            response = httpx.post(
                _JINA_API_URL,
                headers=self._headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException:
            logger.warning(
                "Jina Rerank API timed out after %ds — falling back to RRF order.",
                _REQUEST_TIMEOUT,
            )
            return [candidates[i] for i in range(limit)]

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Jina Rerank API returned HTTP %d — falling back to RRF order. "
                "Response: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return [candidates[i] for i in range(limit)]

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Jina Rerank API call failed (%s) — falling back to RRF order.",
                exc,
            )
            return [candidates[i] for i in range(limit)]

        # Jina returns results[] sorted by relevance_score descending.
        # Each result has an "index" field pointing into the original documents list.
        results = data.get("results", [])
        if not results:
            logger.warning(
                "Jina Rerank returned empty results — falling back to RRF order."
            )
            return [candidates[i] for i in range(limit)]

        reranked = [candidates[r["index"]] for r in results]

        logger.debug(
            "Jina Reranker top scores: %s",
            [round(r.get("relevance_score", 0.0), 3) for r in results],
        )

        return reranked


# ── Module-level singleton (same pattern as previous CrossEncoder version) ────
reranker_service = RerankerService()
