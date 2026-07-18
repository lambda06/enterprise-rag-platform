"""
Embedding service for the Enterprise RAG Platform.

Uses Google's text-embedding-004 (or configured Gemini embedding model) for dense
vector representations of text, table summaries, and vision captions for semantic
search in the Qdrant vector store.

Architecture overview
─────────────────────
This module uses the Google Gemini / Vertex AI (`google-genai`) SDK for embeddings.
Key architectural properties:

  1. **Task types** — Distinguishes between documents being indexed
     (RETRIEVAL_DOCUMENT) and queries issued at search time (RETRIEVAL_QUERY).
     The model bends its internal vector space so that a QUERY vector points
     toward the region where matching DOCUMENT vectors live.

  2. **Dimensionality** — For text-embedding-004, 768 is its
     native full-capacity dimension (`output_dimensionality=768`). 

  3. **Multimodal Captions** — Images (`ImageExtractor`) and structured tables
     (`TableExtractor`) are converted to natural language Vision captions before
     embedding (`embed_chunks([caption])`), ensuring high-recall cross-modal
     retrieval and reranking over a clean text vector space.

  4. **Rate limiting** — All API calls are wrapped with exponential back-off via
     ``tenacity`` (retries on ``google.api_core.exceptions.ResourceExhausted`` / HTTP 429).

Requires: pip install google-genai tenacity Pillow
"""

from __future__ import annotations

import logging

import numpy as np
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ─── Retry policy ─────────────────────────────────────────────────────────────
# On ResourceExhausted (HTTP 429) we back off exponentially:
#   attempt 1 → wait 1 s, attempt 2 → 2 s, … capped at 60 s, up to 6 tries.
# google.api_core.exceptions is re-exported through google.genai so we catch
# it via the string name to avoid a hard dependency on google-api-core internals.
try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
    _RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (_ResourceExhausted,)
except ImportError:
    # Fallback: catch generic Exception subclass named ResourceExhausted
    _RETRY_EXCEPTIONS = (Exception,)


def _gemini_retry(fn):  # type: ignore[no-untyped-def]
    """Decorator: exponential back-off retry for Gemini API rate-limit errors."""
    return retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(fn)


# ─── EmbeddingService ─────────────────────────────────────────────────────────

class EmbeddingService:
    """Service for generating dense text embeddings via Google Gemini / Vertex AI.

    Exposes two methods that the rest of the platform calls:

    * ``embed_chunks(texts)``  — bulk ingestion of document chunks, vision captions, and table summaries
    * ``embed_query(query)``   — single query at retrieval time

    All returned arrays are float32 vectors of length
    ``settings.gemini.embedding_dimensions`` (default 768, natively normalized by the API).
    """

    def __init__(self) -> None:
        """Initialise the Gemini client from application settings."""
        settings = get_settings()
        self._model: str = settings.gemini.embedding_model
        self._dims: int = settings.gemini.embedding_dimensions
        
        if settings.gcp.use_vertex_ai:
            self._client = genai.Client(
                vertexai=True, 
                project=settings.gcp.project_id, 
                location=settings.gcp.region
            )
            logger.info(
                "EmbeddingService initialised with Vertex AI (ADC): model=%s dims=%d",
                self._model,
                self._dims,
            )
        else:
            api_key = settings.gemini.api_key
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not configured in settings (and USE_VERTEX_AI is False)")
            self._client = genai.Client(api_key=api_key)
            logger.info(
                "EmbeddingService initialised with API Key: model=%s dims=%d",
                self._model,
                self._dims,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_chunks(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of document chunks with RETRIEVAL_DOCUMENT task type.

        Uses task_type=RETRIEVAL_DOCUMENT because these strings are pieces of
        factual content that will sit in Qdrant waiting to be found.  The model
        encodes them in a region of the vector space that RETRIEVAL_QUERY
        vectors are trained to point toward.

        Args:
            texts: List of chunk content strings to embed.

        Returns:
            List of 1-D float32 numpy arrays, one per input text, each of
            length ``self._dims``.
            Returns an empty list when *texts* is empty.
        """
        if not texts:
            return []

        logger.debug("embed_chunks: embedding %d chunks", len(texts))
        raw_embeddings = self._embed_texts_with_retry(
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        result = [np.asarray(e, dtype=np.float32) for e in raw_embeddings]
        logger.debug("embed_chunks: done, %d vectors of dim %d", len(result), self._dims)
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query with RETRIEVAL_QUERY task type.

        Uses task_type=RETRIEVAL_QUERY so the model positions this vector in
        the direction of relevant RETRIEVAL_DOCUMENT vectors rather than in the
        cluster where similar-length-question vectors live.  Using
        RETRIEVAL_DOCUMENT here would degrade recall measurably.

        Args:
            query: The raw user search string.

        Returns:
            1-D float32 numpy array of length ``self._dims``.
        """
        logger.debug("embed_query: '%s'", query[:80])
        raw_embeddings = self._embed_texts_with_retry(
            texts=[query],
            task_type="RETRIEVAL_QUERY",
        )
        return np.asarray(raw_embeddings[0], dtype=np.float32)

    # ── Private retry-wrapped API callers ──────────────────────────────────────

    @_gemini_retry
    def _embed_texts_with_retry(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """Call the Gemini embedding API for a list of text strings.

        Wrapped with ``@_gemini_retry`` so that ResourceExhausted (HTTP 429)
        errors are retried automatically with exponential back-off.

        Args:
            texts:     One or more strings to embed.
            task_type: Gemini task type string, e.g. ``"RETRIEVAL_DOCUMENT"``.

        Returns:
            List of raw float lists, one per input text.
        """
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dims,
            ),
        )
        # response.embeddings is a list of ContentEmbedding objects, each with
        # a .values attribute that is a list of floats.
        return [emb.values for emb in response.embeddings]


# ─── Singleton ────────────────────────────────────────────────────────────────
# Instantiated once per process on first import.  The Gemini client is
# stateless (no in-process model weights), so the singleton only holds the
# configured API key reference and model name — it is cheap.

_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# Module-level singleton instance (kept for backward-compatible imports
# such as ``from app.rag.embeddings import embedding_service``).
embedding_service = get_embedding_service()
