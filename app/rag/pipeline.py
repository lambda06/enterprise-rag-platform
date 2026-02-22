"""
RAG pipeline that combines retrieval and LLM generation.

Provides `RAGPipeline` which uses the async `RetrievalService` to fetch
relevant context chunks and the `GroqLLMService` to generate a grounded
answer. The public `query` method is async and returns a dict with:

- `answer`: LLM response string
- `source_chunks`: list of retrieved chunks (each with `text` and `metadata`)
- `chunk_count`: number of returned chunks
"""

from __future__ import annotations

from typing import Any, Iterable

from app.llm.groq_client import groq_client
from app.rag.retrieval import retrieval_service


class RAGPipeline:
    """Combine retrieval and LLM to answer queries using retrieved context.

    The pipeline keeps the LLM grounded by providing only the retrieved
    chunks as context. This helps reduce hallucination risk and provides
    traceability for sourced answers.
    """

    def __init__(
        self,
        retrieval=None,
        llm=None,
    ) -> None:
        self._retrieval = retrieval or retrieval_service
        self._llm = llm or groq_client

    async def query(self, user_question: str, top_k: int = 5) -> dict[str, Any]:
        """Answer a user question using retrieval + LLM.

        Steps:
        1. Retrieve top_k relevant chunks for the question.
        2. Pass the chunks to the LLM (grounded prompt) with the question.
        3. Return the answer along with the source chunks and count.
        """
        # Retrieve relevant chunks (async)
        source_chunks = await self._retrieval.retrieve(user_question, top_k)

        # Extract the plain text pieces to pass as context to the LLM
        contexts: Iterable[str] = (c.get("text", "") for c in source_chunks)

        # Generate answer (async wrapper over sync HTTP/API call)
        answer = await self._llm.generate(user_question, contexts)

        return {
            "answer": answer,
            "source_chunks": source_chunks,
            "chunk_count": len(source_chunks),
        }


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


rag_pipeline = get_rag_pipeline()
