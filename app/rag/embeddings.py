"""
Embedding service for the Enterprise RAG Platform.

Uses Google's gemini-embedding-2-preview model for multimodal dense vector
representations of text and images for semantic search in the vector store.

Architecture overview
─────────────────────
This module replaces sentence-transformers (BAAI/bge-small-en-v1.5) with the
Gemini multimodal embedding API.  The key differences:

  1. **Task types** — Gemini distinguishes between documents being indexed
     (RETRIEVAL_DOCUMENT) and queries issued at search time (RETRIEVAL_QUERY).
     The model bends its internal vector space so that a QUERY vector points
     toward the region where matching DOCUMENT vectors live.  Using the wrong
     task type (e.g. embedding a user query as RETRIEVAL_DOCUMENT) will
     noticeably hurt recall.

  2. **MRL / dimensionality** — The model's native space is 3072-dimensional.
     We request 768 dimensions via ``output_dimensionality=768``.  Because MRL
     truncates the vector (it drops the last 2304 components), the remaining
     768-component vector is no longer unit-length.  L2 normalisation must be
     applied manually before storing in Qdrant; this module does it uniformly
     on every API response.

  3. **Multimodal** — Images are passed as base64-encoded bytes parts in the
     exact same API call.  The resulting image vector lives in the same space
     as text vectors, enabling genuine cross-modal retrieval (text query →
     matching image chunk) without any intermediate captioning step.

  4. **Rate limiting** — Preview models have stricter RPM caps than GA models.
     All API calls are wrapped with exponential back-off via ``tenacity``
     (retries on ``google.api_core.exceptions.ResourceExhausted`` / HTTP 429).

Requires: pip install google-genai tenacity Pillow
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from PIL import Image
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
# gemini-embedding-2-preview is a Preview model with stricter QPM/RPM limits.
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
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(fn)


# ─── L2 normalisation ─────────────────────────────────────────────────────────

def _l2_normalize(vector: list[float] | np.ndarray) -> np.ndarray:
    """Return the L2-normalised version of *vector* as a 1-D float32 array.

    When output_dimensionality < 3072 (MRL truncation), the Gemini API returns
    a vector whose magnitude is no longer 1.0.  Without normalisation, Qdrant's
    cosine distance calculations are skewed because they assume unit vectors.

    Args:
        vector: Raw embedding values from the API response.

    Returns:
        1-D numpy float32 array with norm ≈ 1.0.  If the input is all zeros
        (degenerate edge-case), the zero vector is returned unchanged.
    """
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        logger.warning("L2 normalisation got a zero-norm vector — returning as-is.")
        return arr
    return arr / norm


# ─── EmbeddingService ─────────────────────────────────────────────────────────

class EmbeddingService:
    """Service for generating dense embeddings via Google Gemini.

    Exposes three methods that the rest of the platform calls:

    * ``embed_chunks(texts)``  — bulk ingestion of document chunks
    * ``embed_query(query)``   — single query at retrieval time
    * ``embed_image(image)``   — PIL Image for multimodal ingestion

    All returned arrays are L2-normalised float32 vectors of length
    ``settings.gemini.embedding_dimensions`` (default 768).
    """

    def __init__(self) -> None:
        """Initialise the Gemini client from application settings."""
        settings = get_settings()
        self._model: str = settings.gemini.embedding_model
        self._dims: int = settings.gemini.embedding_dimensions
        self._client = genai.Client(api_key=settings.gemini.api_key)
        logger.info(
            "EmbeddingService initialised: model=%s dims=%d",
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
            length ``self._dims`` and L2-normalised (norm ≈ 1.0).
            Returns an empty list when *texts* is empty.
        """
        if not texts:
            return []

        logger.debug("embed_chunks: embedding %d chunks", len(texts))
        raw_embeddings = self._embed_texts_with_retry(
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        result = [_l2_normalize(e) for e in raw_embeddings]
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
            1-D float32 numpy array of length ``self._dims``, L2-normalised.
        """
        logger.debug("embed_query: '%s'", query[:80])
        raw_embeddings = self._embed_texts_with_retry(
            texts=[query],
            task_type="RETRIEVAL_QUERY",
        )
        return _l2_normalize(raw_embeddings[0])

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a PIL Image using Gemini's multimodal embedding API.

        The image is sent as an inline base64-encoded PNG part in the same
        Gemini embed_content call — no separate captioning step required.
        task_type=RETRIEVAL_DOCUMENT because images are indexed content, not
        query intent.

        The resulting vector lives in the same embedding space as text vectors,
        enabling genuine cross-modal retrieval: a text query such as
        "red sports car" will match a stored image embedding of a red sports car
        without any caption acting as an intermediary.

        Args:
            image: A PIL Image object (any mode; converted to PNG internally).

        Returns:
            1-D float32 numpy array of length ``self._dims``, L2-normalised.
        """
        logger.debug("embed_image: %s %s", image.mode, image.size)

        # Convert PIL Image → PNG bytes → base64 string
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        image_bytes = buf.getvalue()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        raw_embedding = self._embed_image_with_retry(b64_data=b64_data)
        return _l2_normalize(raw_embedding)

    # ── Private retry-wrapped API callers ──────────────────────────────────────

    @_gemini_retry
    def _embed_texts_with_retry(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """Call the Gemini embedding API for a list of text strings.

        Wrapped with ``@_gemini_retry`` so that ResourceExhausted (HTTP 429)
        errors from the Preview model's tighter rate limits are retried
        automatically with exponential back-off.

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

    @_gemini_retry
    def _embed_image_with_retry(self, b64_data: str) -> list[float]:
        """Call the Gemini embedding API for a single base64-encoded PNG image.

        Args:
            b64_data: Base64-encoded PNG image bytes.

        Returns:
            Raw float list from the API (not yet L2-normalised).
        """
        image_part = types.Part.from_bytes(
            data=base64.b64decode(b64_data),
            mime_type="image/png",
        )
        response = self._client.models.embed_content(
            model=self._model,
            contents=[image_part],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dims,
            ),
        )
        return response.embeddings[0].values


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
