"""
Upstash Redis cache service for the Enterprise RAG Platform.

Provides a ``CacheService`` that stores and retrieves RAG responses using
Upstash Redis over its serverless REST API — no persistent TCP connection,
no Redis server to manage, works behind firewalls and in serverless runtimes.

Cache key strategy
------------------
Keys are MD5 hashes of ``"{question}::{collection_name}"``.  MD5 is used
purely for key shortening (not security) — it produces a compact, fixed-length
hex string that fits comfortably within Redis key limits.

Configuration (via .env)
------------------------
    UPSTASH_REDIS_REST_URL=https://<your-db>.upstash.io
    UPSTASH_REDIS_REST_TOKEN=<your-token>

If either value is missing, every cache call is a no-op (miss / silent skip).
This keeps the application fully functional even before Redis is configured.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def make_cache_key(question: str, collection_name: str) -> str:
    """Return a stable MD5 cache key for a question + collection pair.

    The key format before hashing is ``"<question>::<collection_name>"``.
    MD5 is chosen for speed and compactness — security is not a concern here.

    Args:
        question:        The raw user query string.
        collection_name: The Qdrant collection being searched.

    Returns:
        32-character lowercase hex MD5 digest, prefixed with ``"rag:"``.
        Example: ``"rag:a1b2c3d4e5f6..."``.
    """
    raw = f"{question}::{collection_name}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"rag:{digest}"


class CacheService:
    """Upstash Redis cache for RAG pipeline responses.

    All network calls use the Upstash REST API via ``httpx``.  Methods are
    synchronous because the Qdrant/embedding pipeline is also synchronous;
    callers should dispatch to a thread via ``asyncio.to_thread`` if needed.

    Usage::

        cache = CacheService()
        key = make_cache_key(question, collection)

        hit = cache.get_cached_response(key)
        if hit is not None:
            return hit

        result = pipeline.query(question)
        cache.cache_response(key, result, ttl=3600)
        return result
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._url = settings.redis.rest_url.rstrip("/")
        self._token = settings.redis.rest_token
        self._enabled = bool(self._url and self._token)

        if not self._enabled:
            logger.warning(
                "CacheService: UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN "
                "is not set. Caching is disabled — all requests will be cache misses."
            )

    # ── Internal REST helper ───────────────────────────────────────────────────

    def _rest(self, *command: str) -> Any:
        """Execute one Redis command via the Upstash REST API.

        Upstash's REST protocol accepts commands as JSON arrays POSTed to the
        base URL.  For example, ``SET key value EX 3600`` becomes::

            POST /  body: ["SET", "key", "value", "EX", "3600"]

        Args:
            *command: Redis command tokens, e.g. ``("GET", "rag:abc123")``.

        Returns:
            The ``result`` field from the Upstash JSON response.

        Raises:
            httpx.HTTPError: On network or HTTP-level failure (caller catches).
        """
        response = httpx.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            content=json.dumps(list(command)),
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json().get("result")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        """Return cached pipeline response for ``cache_key``, or ``None``.

        A return value of ``None`` means cache miss (key absent or expired).
        The stored value is expected to be a JSON-serialised dict as written
        by :meth:`cache_response`.

        Args:
            cache_key: The Redis key to look up (typically from
                       :func:`make_cache_key`).

        Returns:
            The deserialised response dict, or ``None`` on miss / error.
        """
        if not self._enabled:
            return None

        try:
            raw = self._rest("GET", cache_key)
            if raw is None:
                logger.debug("Cache miss: %s", cache_key)
                return None

            data: dict[str, Any] = json.loads(raw)
            logger.info("Cache hit: %s", cache_key)
            return data

        except Exception as exc:
            # Never let a cache failure propagate — fall through to live query
            logger.warning("Cache GET failed for key %s: %s", cache_key, exc)
            return None

    def cache_response(
        self,
        cache_key: str,
        data: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """Store ``data`` in Redis under ``cache_key`` with a TTL.

        The dict is serialised to JSON before storage.  If serialisation or
        the network call fails, the error is logged but never re-raised so
        the pipeline response is still returned to the caller.

        Args:
            cache_key: The Redis key to write (typically from
                       :func:`make_cache_key`).
            data:      The pipeline response dict to cache.
            ttl:       Time-to-live in seconds (default 3600 = 1 hour).
                       Set to 0 for no expiry (not recommended in production).
        """
        if not self._enabled:
            return

        try:
            serialised = json.dumps(data, ensure_ascii=False, default=str)
            if ttl > 0:
                self._rest("SET", cache_key, serialised, "EX", str(ttl))
            else:
                self._rest("SET", cache_key, serialised)

            logger.info(
                "Cached response under key %s (TTL=%ds, %d bytes)",
                cache_key,
                ttl,
                len(serialised),
            )

        except Exception as exc:
            logger.warning("Cache SET failed for key %s: %s", cache_key, exc)

    def clear_cache(self, pattern: str = "rag:*") -> int:
        """Delete all cache keys matching ``pattern`` (development use only).

        Uses Redis ``SCAN`` + ``DEL`` to avoid blocking the server with a
        naive ``KEYS *`` call.  In production, prefer key-specific invalidation
        or let TTLs expire naturally.

        Args:
            pattern: Glob pattern to match keys. Defaults to ``"rag:*"``
                     which clears all keys written by this service.

        Returns:
            Number of keys deleted.
        """
        if not self._enabled:
            logger.warning("clear_cache called but caching is disabled.")
            return 0

        deleted = 0
        cursor = "0"

        try:
            while True:
                # SCAN returns [next_cursor, [key1, key2, ...]]
                result = self._rest("SCAN", cursor, "MATCH", pattern, "COUNT", "100")
                cursor, keys = str(result[0]), result[1]

                if keys:
                    self._rest("DEL", *keys)
                    deleted += len(keys)
                    logger.debug("Deleted %d keys (batch)", len(keys))

                if cursor == "0":
                    break  # Full scan complete

            logger.info("clear_cache: deleted %d keys matching '%s'", deleted, pattern)

        except Exception as exc:
            logger.warning("clear_cache failed: %s", exc)

        return deleted


# ── Module-level singleton ─────────────────────────────────────────────────────

_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Return the singleton CacheService instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


cache_service = get_cache_service()
