"""
Unit tests for EmbeddingService (Gemini Embedding 2 preview).

All Gemini API calls are mocked — no real API key or network access is needed.
The tests verify:

  - Correct vector shapes and L2 normalisation for each method.
  - Correct task_type passed to the API per method.
  - embed_image correctly converts a PIL Image and uses RETRIEVAL_DOCUMENT.
  - Retry logic surfaces success after a transient ResourceExhausted error.
"""

from __future__ import annotations

import base64
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image


# ─── Stub helpers ─────────────────────────────────────────────────────────────

_DIM = 768  # target embedding dimension


def _make_raw_vector(dim: int = _DIM, *, norm: float = 2.5) -> list[float]:
    """Return a synthetic float vector with a controlled non-unit norm.

    The norm is intentionally not 1.0 to verify that _l2_normalize is applied.
    """
    v = np.ones(dim, dtype=np.float64) * (norm / math.sqrt(dim))
    return v.tolist()


def _make_embed_response(n: int = 1, dim: int = _DIM) -> MagicMock:
    """Build a mock response object matching google.genai's embed_content shape."""
    embeddings = [
        SimpleNamespace(values=_make_raw_vector(dim)) for _ in range(n)
    ]
    return SimpleNamespace(embeddings=embeddings)


def _patch_client(return_value):
    """Return a context-manager patcher for genai.Client that injects a mock."""
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = return_value
    return patch("app.rag.embeddings.genai.Client", return_value=mock_client), mock_client


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_settings(monkeypatch):
    """Patch get_settings() to return a minimal stub with Gemini config."""
    settings = MagicMock()
    settings.gemini.api_key = "fake-key"
    settings.gemini.embedding_model = "gemini-embedding-2-preview"
    settings.gemini.embedding_dimensions = _DIM
    monkeypatch.setattr("app.rag.embeddings.get_settings", lambda: settings)
    return settings


# ─── embed_query ──────────────────────────────────────────────────────────────

class TestEmbedQuery:
    def test_returns_1d_array_of_correct_dim(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            # Reset module-level singleton so it re-initialises with mock client
            import app.rag.embeddings as emb_mod
            emb_mod._embedding_service = None

            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_query("What is our refund policy?")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (_DIM,)

    def test_l2_norm_is_unit(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_query("test query")

        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_uses_retrieval_query_task_type(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            svc.embed_query("find me something")

        call_kwargs = mock_client.models.embed_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2] if call_kwargs.args else None
        # The config is a types.EmbedContentConfig; check its task_type attribute
        assert "RETRIEVAL_QUERY" in str(call_kwargs)


# ─── embed_chunks ─────────────────────────────────────────────────────────────

class TestEmbedChunks:
    def test_returns_list_of_arrays(self, mock_settings):
        n = 4
        patcher, mock_client = _patch_client(_make_embed_response(n=n))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            results = svc.embed_chunks([f"chunk {i}" for i in range(n)])

        assert isinstance(results, list)
        assert len(results) == n
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (_DIM,)

    def test_all_vectors_are_l2_normalised(self, mock_settings):
        n = 3
        patcher, mock_client = _patch_client(_make_embed_response(n=n))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            results = svc.embed_chunks(["alpha", "beta", "gamma"])

        for arr in results:
            norm = float(np.linalg.norm(arr))
            assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_empty_input_returns_empty_list(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=0))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_chunks([])

        assert result == []
        mock_client.models.embed_content.assert_not_called()

    def test_uses_retrieval_document_task_type(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=2))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            svc.embed_chunks(["doc one", "doc two"])

        call_kwargs = mock_client.models.embed_content.call_args
        assert "RETRIEVAL_DOCUMENT" in str(call_kwargs)


# ─── embed_image ──────────────────────────────────────────────────────────────

class TestEmbedImage:
    def _make_image(self, width: int = 32, height: int = 32) -> Image.Image:
        return Image.new("RGB", (width, height), color=(128, 64, 32))

    def test_returns_1d_array_of_correct_dim(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_image(self._make_image())

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (_DIM,)

    def test_l2_norm_is_unit(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_image(self._make_image())

        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_uses_retrieval_document_task_type(self, mock_settings):
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            svc.embed_image(self._make_image())

        call_kwargs = mock_client.models.embed_content.call_args
        assert "RETRIEVAL_DOCUMENT" in str(call_kwargs)

    def test_accepts_rgba_image(self, mock_settings):
        """RGBA images should be converted to RGB before encoding."""
        patcher, mock_client = _patch_client(_make_embed_response(n=1))
        with patcher:
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            rgba_image = Image.new("RGBA", (16, 16))
            result = svc.embed_image(rgba_image)  # must not raise

        assert result.shape == (_DIM,)


# ─── Retry logic ──────────────────────────────────────────────────────────────

class TestRetryLogic:
    def test_retries_on_resource_exhausted_then_succeeds(self, mock_settings):
        """First call raises ResourceExhausted; second call succeeds."""
        try:
            from google.api_core.exceptions import ResourceExhausted
        except ImportError:
            pytest.skip("google-api-core not installed")

        good_response = _make_embed_response(n=1)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ResourceExhausted("rate limit")
            return good_response

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = side_effect

        with patch("app.rag.embeddings.genai.Client", return_value=mock_client), \
             patch("tenacity.nap.time.sleep"):  # skip actual sleep in tests
            from app.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_query("retry test")

        assert call_count == 2
        assert result.shape == (_DIM,)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5
