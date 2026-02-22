"""
Qdrant vector store client for the Enterprise RAG Platform.

Handles collection management, chunk upserts, and semantic search.
"""

from __future__ import annotations

import uuid
import hashlib
from typing import Any

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings


# Cosine distance and why we use it
# ----------------------------------
# Cosine distance = 1 - cosine_similarity. It measures the angle between two
# vectors, ignoring their magnitude. Range: 0 (identical direction) to 2
# (opposite direction). Smaller distance = more similar.
#
# Why cosine over Euclidean?
# - Embeddings are often L2-normalized; magnitude carries little meaning.
# - Cosine focuses on direction (semantic orientation), which matters for
#   "similar meaning" regardless of text length.
# - Robust to document length: a short and long text with the same meaning
#   can have very different Euclidean distances but similar cosine distance.
#
# Qdrant uses cosine as the default for semantic search; it aligns with how
# embedding models are trained (similar texts → similar directions).

# Derive a content-based UUID so that repeated ingestion of the
# same source does not create duplicate points.  In production the
# ingestion pipeline may run multiple times for the same
# document; using a deterministic ID computed from the
# (source_filename, page_number, chunk_index) combo makes the
# process *idempotent* – re‑upserting the same chunk simply
# overwrites the existing point instead of adding a new one.
# This avoids vector store bloat and ensures stable references.

class QdrantService:
    """
    Service for interacting with Qdrant vector store.

    Manages collections, upserts chunks with embeddings, and performs
    semantic search.
    """

    VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize the Qdrant client.

        Args:
            url: Qdrant URL. If None, uses settings.
            api_key: API key for Qdrant Cloud. If None, uses settings.
            collection_name: Collection name. If None, uses settings.
        """
        settings = get_settings()
        self._client = QdrantClient(
            url=url or settings.qdrant.url,
            api_key=api_key or settings.qdrant.api_key,
        )
        self._collection = collection_name or settings.qdrant.collection_name

    def ensure_collection(self, vector_size: int = VECTOR_SIZE) -> None:
        """
        Create the collection if it does not exist.

        Uses cosine distance for semantic similarity search. Existing
        collections are left unchanged.

        Args:
            vector_size: Embedding dimension (default 384 for bge-small-en).
        """
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if self._collection in names:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Upsert chunks with their embeddings into the collection.

        Each chunk is stored as a point with a UUID, its embedding as the
        vector, and text + metadata in the payload.

        Args:
            chunks: List of chunk dicts with keys "text" and "metadata".
            embeddings: Numpy array of shape (n, dim) matching chunk order.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Derive a content-based UUID so that repeated ingestion of the
            # same source does not create duplicate points.  In production the
            # ingestion pipeline may run multiple times for the same
            # document; using a deterministic ID computed from the
            # (source_filename, page_number, chunk_index) combo makes the
            # process *idempotent* – re‑upserting the same chunk simply
            # overwrites the existing point instead of adding a new one.
            # This avoids vector store bloat and ensures stable references.
            meta = chunk.get("metadata", {})
            source = meta.get("source_filename", "")
            page = meta.get("page_number", 0)
            idx = meta.get("chunk_index", 0)
            hash_input = f"{source}:{page}:{idx}".encode("utf-8")
            sha_bytes = hashlib.sha256(hash_input).digest()
            # take first 16 bytes of the hash to create a deterministic UUID
            chunk_uuid = uuid.UUID(bytes=sha_bytes[:16])

            payload = {
                "text": chunk.get("text", ""),
                **meta,
            }
            points.append(
                PointStruct(
                    id=str(chunk_uuid),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar chunks by query embedding.

        Args:
            query_embedding: Query vector of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of dicts with "text" and "metadata" for each result,
            ordered by relevance (most similar first).
        """
        # qdrant-client has evolved across versions: older/newer clients
        # expose different method names. Try common variants and normalize
        # the returned hits to a simple list of dicts with `text` and
        # `metadata` keys.
        results = None
        try:
            if hasattr(self._client, "search"):
                results = self._client.search(
                    collection_name=self._collection,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                )
            elif hasattr(self._client, "search_points"):
                # Newer qdrant-client versions use `search_points`
                results = self._client.search_points(
                    collection_name=self._collection,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
            else:
                # Fall back to calling the Qdrant HTTP API directly. This
                # covers environments where the installed qdrant-client
                # version doesn't expose a compatible search method.
                settings = get_settings()
                base = settings.qdrant.url.rstrip("/")
                url = f"{base}/collections/{self._collection}/points/search"
                headers = {"Content-Type": "application/json"}
                if settings.qdrant.api_key:
                    # Accept both common header styles to be permissive
                    headers["api-key"] = settings.qdrant.api_key
                    headers["Authorization"] = f"Bearer {settings.qdrant.api_key}"

                payload = {
                    "vector": query_embedding.tolist(),
                    "limit": top_k,
                    "with_payload": True,
                }

                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # Qdrant HTTP returns {'result': [...]} or similar
                results = data.get("result") if isinstance(data, dict) else data
        except Exception:
            # Re-raise a clearer error for the API callers
            raise

        def _extract_payload(hit: Any) -> dict[str, Any]:
            # Compatible extraction for different result shapes
            if hasattr(hit, "payload"):
                return hit.payload or {}
            if isinstance(hit, dict):
                return hit.get("payload", {}) or {}
            return {}

        formatted: list[dict[str, Any]] = []
        for hit in results:
            payload = _extract_payload(hit)
            formatted.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )

        return formatted
