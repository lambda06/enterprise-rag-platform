"""
Ingestion pipeline for the Enterprise RAG Platform.

Orchestrates parsing, chunking, embedding, and vector store upsert.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.ingestion.chunker import chunk_pages
from app.ingestion.parser import parse_pdf
from app.rag.embeddings import get_embedding_service
from app.vectorstore.qdrant_client import QdrantService

logger = logging.getLogger(__name__)


async def ingest(
    file_path: str | Path,
    filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    """
    Run the full ingestion pipeline: parse → chunk → embed → upsert.

    Args:
        file_path: Path to the PDF file.
        filename: Display name for the document (defaults to file_path name).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        Summary dict with filename, total_pages, total_chunks, status.
    """
    path = Path(file_path)
    display_name = filename or path.name

    summary: dict[str, Any] = {
        "filename": display_name,
        "total_pages": 0,
        "total_chunks": 0,
        "status": "failed",
    }

    try:
        # Step 1: Parse PDF
        logger.info("Ingestion started: %s", display_name)
        pages = await asyncio.to_thread(parse_pdf, path)
        total_pages = len(pages)
        summary["total_pages"] = total_pages
        logger.info("Parsed %d pages from %s", total_pages, display_name)

        # Step 2: Chunk
        chunks = await asyncio.to_thread(
            chunk_pages,
            pages,
            display_name,
            chunk_size,
            chunk_overlap,
        )
        total_chunks = len(chunks)
        summary["total_chunks"] = total_chunks
        logger.info("Created %d chunks from %s", total_chunks, display_name)

        if not chunks:
            summary["status"] = "completed"
            logger.warning("No chunks produced for %s (empty or image-only PDF)", display_name)
            return summary

        # Step 3: Embed (single batch)
        logger.info("Embedding %d chunks...", total_chunks)
        texts = [c["text"] for c in chunks]
        embeddings = await asyncio.to_thread(
            get_embedding_service().embed_chunks,
            texts,
        )
        logger.info("Embedded %d chunks", len(embeddings))

        # Step 4: Upsert to Qdrant
        qdrant = QdrantService()
        await asyncio.to_thread(qdrant.ensure_collection)
        await asyncio.to_thread(qdrant.upsert_chunks, chunks, embeddings)
        logger.info("Upserted %d chunks to Qdrant", total_chunks)

        summary["status"] = "completed"
        logger.info("Ingestion completed: %s (%d pages, %d chunks)", display_name, total_pages, total_chunks)

    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", display_name, e)
        summary["status"] = "failed"
        summary["error"] = str(e)

    return summary
