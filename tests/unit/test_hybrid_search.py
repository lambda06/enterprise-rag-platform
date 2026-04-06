"""
Unit tests for RRF fusion logic in QdrantService.hybrid_search.

These tests are fully self-contained — they mock the QdrantClient so no
running Qdrant instance is required.  The focus is on verifying the
correctness of:

  1. RRF score computation (1 / (k + rank) per list)
  2. Fusion ordering (doc in both lists beats doc in one list)
  3. top_k truncation
  4. Edge cases (empty legs, identical results across legs)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake Qdrant hit objects
# ---------------------------------------------------------------------------

def _make_hit(point_id: str, text: str = "", **meta) -> SimpleNamespace:
    """Create a minimal fake Qdrant hit with .id and .payload attributes."""
    payload = {"text": text, **meta}
    return SimpleNamespace(id=point_id, payload=payload)


# ---------------------------------------------------------------------------
# Isolated RRF logic tests  (no QdrantService instantiation needed)
# ---------------------------------------------------------------------------

class TestRRFFusion:
    """Test the Reciprocal Rank Fusion scoring algorithm in isolation."""

    def _rrf(
        self,
        dense_ids: list[str],
        sparse_ids: list[str],
        top_k: int = 10,
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Pure-Python RRF that mirrors the implementation in hybrid_search."""
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, doc_id in enumerate(sparse_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def test_single_doc_in_both_lists_gets_double_contribution(self):
        """A doc ranked 1st in both legs should earn 2 × (1/(k+1))."""
        results = self._rrf(dense_ids=["doc_a"], sparse_ids=["doc_a"])
        assert len(results) == 1
        doc_id, score = results[0]
        assert doc_id == "doc_a"
        expected = 2.0 / (60 + 1)
        assert abs(score - expected) < 1e-9

    def test_doc_in_both_lists_beats_doc_in_one_list(self):
        """
        'doc_both' appears rank-3 in dense and rank-3 in sparse.
        'doc_dense_only' appears rank-1 in dense only.

        Even though doc_dense_only has a better per-list position, doc_both's
        combined score should exceed it once its two contributions are summed.
        """
        # doc_dense_only: 1/(60+1) ≈ 0.01639
        # doc_both:       1/(60+3) + 1/(60+3) = 2/63 ≈ 0.03175
        dense_ids = ["doc_dense_only", "doc_x", "doc_both"]
        sparse_ids = ["doc_sparse_only", "doc_y", "doc_both"]
        results = self._rrf(dense_ids=dense_ids, sparse_ids=sparse_ids)

        ids_in_order = [r[0] for r in results]
        assert ids_in_order[0] == "doc_both", (
            "doc_both (appears in both lists) should outrank doc_dense_only "
            "(appears in only one list, even at rank-1)"
        )

    def test_top_k_truncates_correctly(self):
        """Results should be limited to top_k entries."""
        dense_ids = [f"d{i}" for i in range(20)]
        sparse_ids = [f"s{i}" for i in range(20)]
        results = self._rrf(dense_ids=dense_ids, sparse_ids=sparse_ids, top_k=5)
        assert len(results) == 5

    def test_empty_dense_leg(self):
        """If dense search returns nothing, sparse results fill the output."""
        sparse_ids = ["s1", "s2", "s3"]
        results = self._rrf(dense_ids=[], sparse_ids=sparse_ids)
        ids = [r[0] for r in results]
        assert ids == sparse_ids

    def test_empty_sparse_leg(self):
        """If sparse search returns nothing, dense results fill the output."""
        dense_ids = ["d1", "d2", "d3"]
        results = self._rrf(dense_ids=dense_ids, sparse_ids=[])
        ids = [r[0] for r in results]
        assert ids == dense_ids

    def test_both_legs_empty(self):
        """If both legs are empty, hybrid search returns an empty list."""
        results = self._rrf(dense_ids=[], sparse_ids=[])
        assert results == []

    def test_scores_decrease_monotonically(self):
        """Output should be ordered highest score first."""
        dense_ids = ["a", "b", "c", "d"]
        sparse_ids = ["c", "d", "a", "b"]
        results = self._rrf(dense_ids=dense_ids, sparse_ids=sparse_ids)
        scores_only = [r[1] for r in results]
        assert scores_only == sorted(scores_only, reverse=True)

    def test_rank1_both_legs_is_highest_score(self):
        """A doc at rank-1 in both lists must be the top result."""
        dense_ids = ["winner", "a", "b"]
        sparse_ids = ["winner", "c", "d"]
        results = self._rrf(dense_ids=dense_ids, sparse_ids=sparse_ids)
        assert results[0][0] == "winner"


# ---------------------------------------------------------------------------
# Integration-style tests: QdrantService.hybrid_search with mocked client
# ---------------------------------------------------------------------------

class TestQdrantServiceHybridSearch:
    """Test QdrantService.hybrid_search end-to-end with a mocked Qdrant client."""

    def _make_service(self, dense_hits, sparse_hits):
        """Return a QdrantService whose internal client is fully mocked."""
        with patch("app.vectorstore.qdrant_client.get_settings") as mock_settings, \
             patch("app.vectorstore.qdrant_client.QdrantClient") as MockClient:

            # Minimal settings stub
            settings = MagicMock()
            settings.qdrant.url = "http://localhost:6333"
            settings.qdrant.api_key = None
            settings.qdrant.collection_name = "test_collection"
            mock_settings.return_value = settings

            # Mock the query_points method to return our test hits wrapped in a container
            mock_client = MagicMock()
            mock_client.query_points.side_effect = [
                SimpleNamespace(points=dense_hits),
                SimpleNamespace(points=sparse_hits)
            ]
            MockClient.return_value = mock_client

            from app.vectorstore.qdrant_client import QdrantService
            svc = QdrantService()
            svc._client = mock_client  # override with pre-configured mock
            return svc

    def test_hybrid_search_returns_correct_keys(self):
        """Each result dict must have 'text', 'metadata', and 'rrf_score'."""
        from app.vectorstore.qdrant_client import QdrantService

        svc = QdrantService.__new__(QdrantService)
        svc._collection = "test"
        svc._client = MagicMock()

        dense_hits = [_make_hit("doc1", text="The XT-500 sensor calibration guide.")]
        sparse_hits = [_make_hit("doc1", text="The XT-500 sensor calibration guide.")]

        svc._client.query_points.side_effect = [
            SimpleNamespace(points=dense_hits),
            SimpleNamespace(points=sparse_hits)
        ]

        query_emb = np.zeros(384)
        results = svc.hybrid_search("XT-500", query_emb, top_k=1)

        assert len(results) == 1
        result = results[0]
        assert "text" in result
        assert "metadata" in result
        assert "rrf_score" in result
        assert isinstance(result["rrf_score"], float)

    def test_hybrid_search_top_k_limits_output(self):
        """hybrid_search must never return more than top_k results."""
        from app.vectorstore.qdrant_client import QdrantService

        svc = QdrantService.__new__(QdrantService)
        svc._collection = "test"
        svc._client = MagicMock()

        dense_hits = [_make_hit(f"doc{i}", text=f"chunk {i}") for i in range(10)]
        sparse_hits = [_make_hit(f"doc{i+5}", text=f"chunk {i+5}") for i in range(10)]

        svc._client.query_points.side_effect = [
            SimpleNamespace(points=dense_hits),
            SimpleNamespace(points=sparse_hits)
        ]

        results = svc.hybrid_search("something", np.zeros(384), top_k=3)
        assert len(results) <= 3

    def test_hybrid_search_rrf_score_is_positive(self):
        """All RRF scores must be strictly positive."""
        from app.vectorstore.qdrant_client import QdrantService

        svc = QdrantService.__new__(QdrantService)
        svc._collection = "test"
        svc._client = MagicMock()

        dense_hits = [_make_hit("a", text="doc a"), _make_hit("b", text="doc b")]
        sparse_hits = [_make_hit("b", text="doc b"), _make_hit("c", text="doc c")]

        svc._client.query_points.side_effect = [
            SimpleNamespace(points=dense_hits),
            SimpleNamespace(points=sparse_hits)
        ]

        results = svc.hybrid_search("hello", np.zeros(384), top_k=5)
        for r in results:
            assert r["rrf_score"] > 0.0

    def test_hybrid_search_doc_in_both_lists_ranked_first(self):
        """
        'doc_b' is rank-1 sparse, rank-2 dense.
        'doc_a' is rank-1 dense only.

        doc_b's combined score: 1/62 + 1/61 ≈ 0.0161 + 0.0164 = 0.0325
        doc_a's score:          1/61           ≈ 0.0164

        doc_b should be ranked first.
        """
        from app.vectorstore.qdrant_client import QdrantService

        svc = QdrantService.__new__(QdrantService)
        svc._collection = "test"
        svc._client = MagicMock()

        dense_hits = [
            _make_hit("doc_a", text="only in dense"),   # rank 1 dense
            _make_hit("doc_b", text="in both lists"),   # rank 2 dense
        ]
        sparse_hits = [
            _make_hit("doc_b", text="in both lists"),   # rank 1 sparse
        ]

        svc._client.query_points.side_effect = [
            SimpleNamespace(points=dense_hits),
            SimpleNamespace(points=sparse_hits)
        ]

        results = svc.hybrid_search("query", np.zeros(384), top_k=2)
        assert results[0]["text"] == "in both lists", (
            "doc_b should be first because it appears in both lists"
        )
