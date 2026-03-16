"""
Ingestion pipeline for the Enterprise RAG Platform.

Orchestrates parsing, chunking, embedding, and vector store upsert.
Runs text, image, and table extraction in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.ingestion.chunker import chunk_pages
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.parser import parse_pdf
from app.ingestion.table_extractor import TableExtractor
from app.rag.embeddings import get_embedding_service
from app.vectorstore.qdrant_client import QdrantService

logger = logging.getLogger(__name__)


async def _extract_text(
    path: Path,
    display_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict[str, Any]], int]:
    """Extract, chunk, and embed regular text from the PDF.
    
    Returns:
        (records, total_pages)
    """
    logger.info("Text extraction started for %s", display_name)
    pages = await asyncio.to_thread(parse_pdf, path)
    total_pages = len(pages)
    
    chunks = await asyncio.to_thread(
        chunk_pages,
        pages,
        display_name,
        chunk_size,
        chunk_overlap,
    )
    
    if not chunks:
        logger.warning("No text chunks produced for %s", display_name)
        return [], total_pages
        
    texts = [c["text"] for c in chunks]
    embeddings = await asyncio.to_thread(
        get_embedding_service().embed_chunks,
        texts,
    )
    
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": emb,
        })
        
    logger.info("Text extraction finished for %s (%d chunks)", display_name, len(records))
    return records, total_pages


async def _extract_images(path: Path, display_name: str) -> list[dict[str, Any]]:
    """Extract and embed images from the PDF.

    Option B — vision at query time: ``image_base64`` is stored *inside*
    ``metadata`` so it lands in the Qdrant point payload and is returned with
    every search hit.  At query time the RAG pipeline detects image chunks
    (``metadata.content_type == 'image'``) and passes the raw base64 bytes
    to Gemini in a single multimodal request alongside the text context.
    """
    logger.info("Image extraction started for %s", display_name)
    extractor = ImageExtractor()
    raw_records = await asyncio.to_thread(extractor.extract, path, display_name)

    records = []
    for rec in raw_records:
        # Merge image_base64 into metadata so it is serialised into the
        # Qdrant payload automatically — accessible on every search hit.
        metadata = dict(rec["metadata"])
        metadata["image_base64"] = rec.get("image_base64", "")

        records.append({
            "text": "",   # Empty text — BM25 will score this near zero
            "metadata": metadata,
            "embedding": rec["embedding"],
        })

    logger.info("Image extraction finished for %s (%d images)", display_name, len(records))
    return records


async def _extract_tables(path: Path, display_name: str) -> list[dict[str, Any]]:
    """Extract and embed tables from the PDF."""
    logger.info("Table extraction started for %s", display_name)
    extractor = TableExtractor()
    # TableExtractor converts to markdown and computes embeddings
    raw_records = await asyncio.to_thread(extractor.extract, path, display_name)
    
    records = []
    for rec in raw_records:
        records.append({
            "text": rec["markdown_text"], # Use markdown string as the primary text payload
            "metadata": rec["metadata"],
            "embedding": rec["embedding"],
        })
        
    logger.info("Table extraction finished for %s (%d tables)", display_name, len(records))
    return records


async def ingest(
    file_path: str | Path,
    filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    """
    Run the full ingestion pipeline: text, images, and tables in parallel.

    Args:
        file_path: Path to the PDF file.
        filename: Display name for the document (defaults to file_path name).
        chunk_size: Max characters per chunk (for text only).
        chunk_overlap: Overlap between adjacent chunks (for text only).

    Returns:
        Summary dict with counts for each content type.
    """
    path = Path(file_path)
    display_name = filename or path.name

    summary: dict[str, Any] = {
        "filename": display_name,
        "total_pages": 0,
        "text_chunks": 0,
        "image_chunks": 0,
        "table_chunks": 0,
        "total_chunks": 0,
        "status": "failed",
    }

    try:
        logger.info("Parallel ingestion started: %s", display_name)
        
        # Run all three extraction phases concurrently using asyncio.gather.
        # This significantly reduces overall ingestion latency because:
        # 1. Image and table extractors can perform heavy I/O and CPU work
        #    (via to_thread) at the same time text parsing is happening.
        # 2. Network calls to the Gemini API for text, image, and table
        #    embeddings occur in parallel rather than blocking each other.
        text_task = _extract_text(path, display_name, chunk_size, chunk_overlap)
        image_task = _extract_images(path, display_name)
        table_task = _extract_tables(path, display_name)
        
        # Allow independent tasks to succeed/fail. We track exceptions
        # to ensure we don't swallow critical errors (e.g. file missing).
        results = await asyncio.gather(
            text_task, image_task, table_task, return_exceptions=True
        )
        
        # Check for critical errors (e.g. file not found)
        for res in results:
            if isinstance(res, Exception):
                raise res # Re-raise the first major error
                
        text_result, image_records, table_records = results
        text_records, total_pages = text_result
        
        summary["total_pages"] = total_pages
        summary["text_chunks"] = len(text_records)
        summary["image_chunks"] = len(image_records)
        summary["table_chunks"] = len(table_records)
        summary["total_chunks"] = len(text_records) + len(image_records) + len(table_records)
        
        if summary["total_chunks"] == 0:
            summary["status"] = "completed"
            logger.warning("No chunks of any type produced for %s", display_name)
            return summary

        # Combine all records into single lists for upsert
        all_records = text_records + image_records + table_records
        
        # Qdrant upsert expects `chunks` (list of dicts with text/metadata) 
        # and a strictly corresponding `embeddings` array (shape N, 768).
        chunks_for_upsert = []
        embeddings_list = []
        
        for rec in all_records:
            chunk_dict = {
                "text": rec["text"],
                "metadata": rec["metadata"],
            }
            # Add image base64 if it's an image record so UI can render it later
            if "image_base64" in rec:
                chunk_dict["image_base64"] = rec["image_base64"]
                
            chunks_for_upsert.append(chunk_dict)
            embeddings_list.append(rec["embedding"])
            
        stacked_embeddings = np.stack(embeddings_list)
        
        # Upsert to Qdrant using the pre-computed embeddings array
        qdrant = QdrantService()
        await asyncio.to_thread(qdrant.ensure_collection)
        
        logger.info("Upserting combined batch of %d items to Qdrant...", len(chunks_for_upsert))
        await asyncio.to_thread(qdrant.upsert_chunks, chunks_for_upsert, stacked_embeddings)
        
        summary["status"] = "completed"
        logger.info(
            "Ingestion completed: %s (%d text, %d images, %d tables)", 
            display_name, summary["text_chunks"], summary["image_chunks"], summary["table_chunks"]
        )

    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", display_name, e)
        summary["status"] = "failed"
        summary["error"] = str(e)

    return summary
