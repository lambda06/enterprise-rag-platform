"""
Embedding service for the Enterprise RAG Platform.

Uses sentence-transformers to produce dense vector representations of text
for semantic search in the vector store.

Requires: pip install sentence-transformers
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


# Why use the same model for both document chunks and queries?
# ------------------------------------------------------------
# Embeddings enable semantic search by mapping text to vectors in a shared
# space. Similar meaning → similar vectors → high cosine similarity.
#
# If we used different models for chunks vs queries, they would produce
# vectors in different spaces. A query embedding would be incomparable to
# chunk embeddings—retrieval would be meaningless.
#
# The same model ensures both are in the same space. The query "password
# requirements" and the chunk "Passwords must be 12+ characters..." get
# nearby vectors because the model was trained to align semantically
# similar text. This is the foundation of vector search.


class EmbeddingService:
    """
    Service for generating text embeddings using a sentence-transformers model.

    Uses the embedding model configured in settings (e.g., BAAI/bge-small-en-v1.5).
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: Override for the model name. If None, uses settings.
        """
        settings = get_settings()
        name = model_name or settings.huggingface.embedding_model
        self._model = SentenceTransformer(name)

    def  embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of chunk texts.

        Args:
            texts: List of chunk strings to embed.

        Returns:
            Numpy array of shape (n, dim) where n = len(texts) and dim is
            the model's embedding dimension (e.g., 384 for bge-small-en-v1.5).
        """
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()))
        return self._model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The search query to embed.

        Returns:
            Numpy array of shape (dim,) — the query embedding vector.
        """
        embedding = self._model.encode(query, convert_to_numpy=True)
        # Ensure 1D for single query (encode may return (1, dim) for list input)
        return np.squeeze(embedding)


# Module-level singleton (loaded on first import of this module).
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# Module-level singleton instance
embedding_service = get_embedding_service()
