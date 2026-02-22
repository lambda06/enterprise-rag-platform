"""
Retrieval service for the Enterprise RAG Platform.

Provides an async `RetrievalService` that embeds a query and searches
the Qdrant vector store for the top results, returning a list of
dicts with `text` and `metadata`.

The embedding and Qdrant clients in this code are synchronous, so
work is dispatched to a thread pool using ``asyncio.to_thread`` to
avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.rag.embeddings import embedding_service
from app.vectorstore.qdrant_client import QdrantService


class RetrievalService:
    """Async retrieval wrapper that embeds queries and searches Qdrant.

    Usage:
        service = RetrievalService()
        results = await service.retrieve("search text")
    """

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        embed_service=None,
    ) -> None:
        self._qdrant = qdrant_service or QdrantService()
        self._embed = embed_service or embedding_service

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k search results for the given query.

        Args:
            query: The user query string.
            top_k: Number of top results to return (default 5).

        Returns:
            List of dicts with keys `text` and `metadata` ordered by
            relevance (most similar first).
        """
        # Generate query embedding in a thread to avoid blocking
        query_embedding: np.ndarray = await asyncio.to_thread(
            self._embed.embed_query, query
        )

        # Perform vector search in a thread as the Qdrant client is sync
        raw_results = await asyncio.to_thread(
            self._qdrant.search, query_embedding, top_k
        )

        # Normalize/format results
        formatted = [
            {
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
            }
            for hit in raw_results
        ]

        return formatted


# Module-level singleton
_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """Return the singleton RetrievalService instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


retrieval_service = get_retrieval_service()
