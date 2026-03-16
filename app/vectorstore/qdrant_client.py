"""
Qdrant vector store client for the Enterprise RAG Platform.

Handles collection management, chunk upserts, dense semantic search,
and hybrid (dense + sparse BM25) search with Reciprocal Rank Fusion.

─── Architectural Note: Embedding Dimension Change (384 → 768) ──────────────
Previously this platform used BAAI/bge-small-en-v1.5 (sentence-transformers),
a text-only model producing 384-dimensional vectors.  Multimodal support
required three separate pipelines: bge-small-en for text, GPT-4o-mini vision
API to caption images (then embed the captions as text), and markdown
conversion for tables followed by text embedding.  This was complex, had
multiple failure points, and still produced text-to-text retrieval of image
descriptions rather than genuine cross-modal retrieval.

On 2026-03-10 Google released `gemini-embedding-2-preview`, a fully multimodal
embedding model that maps text, images, video, audio, and PDFs into a single
unified 3072-dimensional embedding space.  When using output_dimensionality
below 3072 (e.g., 768) L2 normalisation must be applied manually — the
embedding service handles this.  This collapses the three-pipeline approach
into one, eliminates the captioning latency/cost, and enables genuine
cross-modal retrieval (text query → image chunk) without any intermediate step.

Consequence: the Qdrant collection schema is INCOMPATIBLE with the old 384-dim
vectors.  Use `QdrantService.reset_collection()` (or the reset_collection
CLI/startup hook) to drop and recreate the collection before re-ingesting.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings


# ─── Vector name constants ────────────────────────────────────────────────────
# Qdrant supports multiple named vectors per point. We use two:
#   "dense"  — 768-dim float vector from gemini-embedding-2-preview (cosine)
#              (L2-normalised by the embedding service before storage)
#   "sparse" — variable-length BM25 bag-of-words vector (dot product)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ─── Cosine distance rationale ────────────────────────────────────────────────
# Cosine distance = 1 − cosine_similarity. It measures the angle between
# two vectors, ignoring magnitude. Range: 0 (identical) → 2 (opposite).
# Preferred over Euclidean for embeddings because:
#   • Models produce L2-normalised outputs; magnitude has no semantic meaning.
#   • Cosine focuses on *direction* (shared concept), which is orientation-
#     independent with respect to text length.
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser for BM25.

    A production system might strip punctuation or apply stemming, but
    lowercase whitespace split is fast, repeatable, and sufficient for
    most English enterprise documents.
    """
    return text.lower().split()


def _build_sparse_vector(
    query_tokens: list[str],
    corpus_tokens: list[list[str]],
) -> SparseVector:
    """Compute a BM25 sparse query vector over the given corpus.

    BM25Okapi scores each token against every document in the corpus, then
    we aggregate those scores per *unique token* to produce a bag-of-weights
    representation.  The indices are stable integer hashes of the token
    strings so they are comparable across separate calls.

    Args:
        query_tokens:  Tokenised query string.
        corpus_tokens: All tokenised documents in this batch (for IDF).

    Returns:
        SparseVector with ``indices`` (token hashes) and ``values`` (BM25 weights).
    """
    if not corpus_tokens:
        return SparseVector(indices=[], values=[])

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    # Map each corpus-token → cumulative BM25 weight.
    # We use a stable positive-integer hash (modulo 2^20 to keep indices small).
    token_weights: dict[int, float] = {}
    for doc_tokens, score in zip(corpus_tokens, scores):
        if score > 0:
            for token in set(doc_tokens):
                token_id = abs(hash(token)) % (2**20)
                token_weights[token_id] = token_weights.get(token_id, 0.0) + float(score)

    if not token_weights:
        return SparseVector(indices=[], values=[])

    indices = list(token_weights.keys())
    values = [token_weights[i] for i in indices]
    return SparseVector(indices=indices, values=values)


def _query_sparse_vector(query_tokens: list[str]) -> SparseVector:
    """Build a sparse query vector from query tokens alone.

    When querying, we don't have a corpus; we assign each token a weight
    of 1.0 so that the dot-product with stored BM25 chunk vectors acts as a
    keyword-overlap score.

    Args:
        query_tokens: Tokens from the user query.

    Returns:
        SparseVector suitable for passing to Qdrant's sparse search.
    """
    token_weights: dict[int, float] = {}
    for token in set(query_tokens):
        token_id = abs(hash(token)) % (2**20)
        token_weights[token_id] = 1.0

    indices = list(token_weights.keys())
    values = [token_weights[i] for i in indices]
    return SparseVector(indices=indices, values=values)


class QdrantService:
    """Service for interacting with Qdrant vector store.

    Manages collections, upserts chunks with both dense and sparse embeddings,
    and performs semantic, keyword, and hybrid search.
    """

    VECTOR_SIZE = 768  # gemini-embedding-2-preview output dimension (L2-normalised sub-3072)

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialise the Qdrant client.

        Args:
            url: Qdrant URL. If None, uses settings.
            api_key: API key for Qdrant Cloud. If None, uses settings.
            collection_name: Collection name. If None, uses settings.
        """
        settings = get_settings()
        self._client = QdrantClient(
            url=url or settings.qdrant.url,
            api_key=api_key or settings.qdrant.api_key,
            timeout=settings.qdrant.timeout,
        )
        self._collection = collection_name or settings.qdrant.collection_name

    # ─── Collection Management ────────────────────────────────────────────────

    def ensure_collection(self, vector_size: int | None = None) -> None:
        """Ensure the collection exists with dense AND sparse vector configs.

        If the collection already exists (e.g., an older dense-only schema)
        it is **deleted and re-created** so that the sparse index is always
        present.  This keeps the method idempotent with respect to the target
        schema: callers can call it on every startup without worrying about
        stale configs.

        Args:
            vector_size: Dense embedding dimension.  Defaults to
                ``settings.qdrant.vector_size`` (768 for
                gemini-embedding-2-preview with L2 normalisation).
        """
        if vector_size is None:
            vector_size = get_settings().qdrant.vector_size

        collections = self._client.get_collections().collections
        names = [c.name for c in collections]

        if self._collection in names:
            # Drop the existing collection so we can apply the new schema
            # (dense + sparse).  This is acceptable because:
            #   1. The ingestion pipeline will re-populate on next upload.
            #   2. Schema mismatches between dense-only and hybrid collections
            #      silently break searches — a clean slate is safer.
            self._client.delete_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            # Named vectors dict: supports multiple independent vector spaces
            # per point.  "dense" uses cosine distance (semantic similarity);
            # "sparse" will be configured separately below.
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            },
            # Sparse vectors are stored in a separate, dedicated index.
            # on_disk=False keeps the index in RAM for fast lookup.
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

    # ─── Upsert ───────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Upsert chunks with dense and sparse vectors into the collection.

        Each chunk is stored as a point containing:
          • A deterministic UUID (from source+page+chunk_index) for idempotency.
          • A named "dense" float vector (semantic embedding).
          • A named "sparse" BM25 vector (keyword index).
          • The full chunk payload (text + metadata).

        Args:
            chunks: List of chunk dicts with keys ``text`` and ``metadata``.
            embeddings: Numpy array of shape (n, dim) matching chunk order.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )

        # Build sparse vectors for all chunks in one BM25 pass so IDF is
        # computed over the full batch (better weighting than per-chunk).
        texts = [c.get("text", "") for c in chunks]
        corpus_tokens = [_tokenize(t) for t in texts]

        points = []
        for chunk, embedding, chunk_tokens in zip(chunks, embeddings, corpus_tokens):
            # ── Deterministic UUID ────────────────────────────────────────────
            # Derive a content-based UUID so that repeated ingestion of the
            # same source does not create duplicate points.
            # Fix: In a multimodal pipeline, a text chunk, an image, and a table
            # on the same page might all have an index of 0. If we only hash
            # source, page, and index, they will collide and overwrite each other.
            # We fix this by including the content_type and the correct index key.
            meta = chunk.get("metadata", {})
            source = meta.get("source_filename", "")
            page = meta.get("page_number", 0)
            idx = meta.get("chunk_index", meta.get("image_index", meta.get("table_index", 0)))
            content_type = meta.get("content_type", "text")
            hash_input = f"{source}:{page}:{idx}:{content_type}".encode("utf-8")
            sha_bytes = hashlib.sha256(hash_input).digest()
            chunk_uuid = uuid.UUID(bytes=bytes(sha_bytes[:16]))

            # ── Sparse BM25 vector ────────────────────────────────────────────
            # We compute the sparse vector for this chunk using the full corpus
            # so IDF scores are normalised across all chunks being upserted.
            sparse_vec = _build_sparse_vector(chunk_tokens, corpus_tokens)

            payload = {
                "text": chunk.get("text", ""),
                **meta,
            }
            points.append(
                PointStruct(
                    id=str(chunk_uuid),
                    # Named vectors: each key maps to its own index/distance metric
                    vector={
                        DENSE_VECTOR_NAME: embedding.tolist(),
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    # ─── Dense Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the most similar chunks by dense query embedding.

        Args:
            query_embedding: Query vector of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of dicts with ``text`` and ``metadata``, ordered by
            cosine similarity (most similar first).
        """
        # query_points is the universal search method in qdrant-client v1.x.
        # Pass the raw vector as `query` and select which named vector to search
        # against with `using`. The response has a `.points` attribute.
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return self._format_hits(response.points)

    # ─── Hybrid Search ────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        rrf_k: int = 60,
        prefetch_multiplier: int = 3,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining dense vectors + sparse BM25, fused via RRF.

        Two independent searches are performed against the same collection:
          1. **Dense search** — embeds the query as a float vector and finds
             semantically similar chunks using cosine distance.
          2. **Sparse search** — represents the query as a BM25 bag-of-words
             vector and finds chunks with high keyword overlap.

        The two ranked lists are then fused with **Reciprocal Rank Fusion**:

            RRF_score(doc) = Σ  1 / (k + rank_i)

        where the sum runs over every list in which the doc appears, ``rank_i``
        is its 1-based rank in that list, and ``k`` (default 60) is a constant
        that dampens the advantage of very high ranks. Documents appearing in
        *both* lists receive contributions from both terms, naturally boosting
        results that are both semantically relevant AND keyword-matching.

        Why k=60?
          The value 60 was validated empirically by Cormack et al. (2009) and
          is the de-facto standard in information retrieval.  It prevents a
          rank-1 result from dominating completely (1/61 ≈ 0.016 vs 1/1=1).

        Args:
            query:               Raw user query string (used for BM25 tokens).
            query_embedding:     Dense query embedding of shape (dim,).
            top_k:               Number of fused results to return.
            rrf_k:               RRF constant (default 60).
            prefetch_multiplier: Each leg fetches ``top_k * prefetch_multiplier``
                                 candidates before fusion.  Larger values give
                                 RRF more to work with at the cost of latency.

        Returns:
            List of dicts with ``text``, ``metadata``, and ``rrf_score``,
            sorted descending by fused RRF score.
        """
        candidate_limit = top_k * prefetch_multiplier
        query_tokens = _tokenize(query)

        # ── Leg 1: Dense semantic search ─────────────────────────────────────
        # query_points() replaces the removed .search() in qdrant-client v1.x.
        # `using` selects which named vector index to query.
        dense_resp = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=candidate_limit,
            with_payload=True,
        )
        dense_hits = dense_resp.points

        # ── Leg 2: Sparse BM25 keyword search ────────────────────────────────
        # Pass SparseVector directly as `query`; Qdrant detects it's sparse.
        query_sparse_vec = _query_sparse_vector(query_tokens)
        sparse_resp = self._client.query_points(
            collection_name=self._collection,
            query=query_sparse_vec,
            using=SPARSE_VECTOR_NAME,
            limit=candidate_limit,
            with_payload=True,
        )
        sparse_hits = sparse_resp.points

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        # Build a map of point_id → accumulated RRF score.
        # We also preserve the payload for each unique ID so we can reconstruct
        # the result dicts after scoring.
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            pt_id = str(hit.id)
            # Each leg contributes 1 / (k + rank) to the fused score.
            # A document ranking 1st in dense search gets 1/(60+1) ≈ 0.0164.
            rrf_scores[pt_id] = rrf_scores.get(pt_id, 0.0) + 1.0 / (rrf_k + rank)
            if pt_id not in payloads:
                payloads[pt_id] = hit.payload or {}

        for rank, hit in enumerate(sparse_hits, start=1):
            pt_id = str(hit.id)
            # A document found in BOTH lists gets contributions from BOTH ranks.
            # e.g., rank-1 dense + rank-1 sparse → 2 × (1/61) ≈ 0.033
            # This is the key property that makes hybrid search powerful:
            # exact-match + semantic-match simultaneously → highest score.
            rrf_scores[pt_id] = rrf_scores.get(pt_id, 0.0) + 1.0 / (rrf_k + rank)
            if pt_id not in payloads:
                payloads[pt_id] = hit.payload or {}

        # Sort all unique docs by fused score (descending) and take top_k
        sorted_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        top_ids = sorted_ids[:top_k]

        results = []
        for pt_id in top_ids:
            payload = payloads[pt_id]
            results.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "rrf_score": round(rrf_scores[pt_id], 6),
                }
            )

        return results

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_hits(hits: list[Any]) -> list[dict[str, Any]]:
        """Normalise Qdrant hit objects into plain dicts."""
        formatted: list[dict[str, Any]] = []
        for hit in hits:
            payload: dict[str, Any] = {}
            if hasattr(hit, "payload"):
                payload = hit.payload or {}
            elif isinstance(hit, dict):
                payload = hit.get("payload", {}) or {}

            formatted.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )
        return formatted
