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

from app.ingestion.chunker import chunk_pages, chunk_pages_hierarchical
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.parser import parse_document, parse_pdf
from app.ingestion.table_extractor import TableExtractor
from app.rag.embeddings import get_embedding_service
from app.vectorstore.qdrant_client import QdrantService

logger = logging.getLogger(__name__)


async def _persist_parent_chunks(parent_records: list[dict[str, Any]]) -> None:
    """Persist ~2,000 character parent chunk records into PostgreSQL using async_session."""
    if not parent_records:
        return
    try:
        from app.db.session import async_session
        from app.models.document import ParentChunk
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        async with async_session() as db:
            for rec in parent_records:
                stmt = pg_insert(ParentChunk).values(
                    id=rec["id"],
                    document_id=rec["document_id"],
                    source_filename=rec["source_filename"],
                    page_number=rec["page_number"],
                    section_title=rec.get("section_title"),
                    content=rec["content"],
                    metadata_json=rec.get("metadata_json"),
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "content": rec["content"],
                        "section_title": rec.get("section_title"),
                        "metadata_json": rec.get("metadata_json"),
                    },
                )
                await db.execute(stmt)
            await db.commit()
            logger.info("Persisted %d parent chunks to PostgreSQL (`parent_chunks` table)", len(parent_records))
    except Exception as exc:
        logger.warning("Failed to persist parent chunks to PostgreSQL (`parent_chunks`): %s", exc)


async def _extract_text(
    path: Path,
    display_name: str,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 350,
    child_chunk_overlap: int = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Extract, hierarchically chunk, and embed text from multi-format documents.

    Returns:
        `(parent_records, child_records, total_pages)`
    """
    logger.info("Text extraction started for %s", display_name)
    pages = await asyncio.to_thread(parse_document, path)
    total_pages = len(pages)

    parent_records, child_chunks = await asyncio.to_thread(
        chunk_pages_hierarchical,
        pages,
        display_name,
        None,
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
    )

    if not child_chunks:
        logger.warning("No text chunks produced for %s", display_name)
        return parent_records, [], total_pages

    texts = [c["text"] for c in child_chunks]
    embeddings = await asyncio.to_thread(
        get_embedding_service().embed_chunks,
        texts,
    )

    records = []
    for chunk, emb in zip(child_chunks, embeddings):
        records.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": emb,
        })

    logger.info(
        "Text extraction finished for %s (%d parent chunks, %d child chunks)",
        display_name,
        len(parent_records),
        len(records),
    )
    return parent_records, records, total_pages


async def _extract_images(path: Path, display_name: str) -> list[dict[str, Any]]:
    """Extract, caption, and embed images from documents.

    Option B — vision at query time: `image_base64` is stored *inside* `metadata`
    so it lands in the Qdrant point payload and is returned with every search hit.
    We index the descriptive Vision-to-Text caption (`text`) so exact BM25 keyword
    search and Jina cross-encoder rerankers work seamlessly.
    """
    logger.info("Image extraction started for %s", display_name)
    extractor = ImageExtractor()
    raw_records = await asyncio.to_thread(extractor.extract, path, display_name)

    records = []
    for rec in raw_records:
        metadata = dict(rec["metadata"])
        metadata["image_base64"] = rec.get("image_base64", "")
        caption_text = rec.get("text", "")
        metadata["text"] = caption_text

        records.append({
            "text": caption_text,  # Vision-to-Text caption for BM25 and cross-encoder
            "metadata": metadata,
            "embedding": rec["embedding"],
        })

    logger.info("Image extraction finished for %s (%d images)", display_name, len(records))
    return records


async def _extract_tables(path: Path, display_name: str) -> list[dict[str, Any]]:
    """Extract, summarize, and embed tables from documents."""
    logger.info("Table extraction started for %s", display_name)
    extractor = TableExtractor()
    raw_records = await asyncio.to_thread(extractor.extract, path, display_name)

    records = []
    for rec in raw_records:
        metadata = dict(rec["metadata"])
        raw_md = rec.get("markdown_text", rec.get("raw_table_content", ""))
        metadata["raw_table_content"] = raw_md
        summary_text = rec.get("text", raw_md)
        metadata["text"] = summary_text

        records.append({
            "text": summary_text,  # Table summary as primary text chunk
            "metadata": metadata,
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
    Run the hierarchical ingestion pipeline: text, images, and tables in parallel.

    Stage 1: Parent chunks (~2,000 chars) persisted directly into PostgreSQL.
    Stage 2: Child chunks (~350 chars), image captions, and table summaries
             upserted into Qdrant vector store.

    Args:
        file_path: Path to the document (.pdf, .docx, .xlsx, .pptx).
        filename: Display name for the document (defaults to file_path name).
        chunk_size: Legacy parameter kept for backward compatibility.
        chunk_overlap: Legacy parameter kept for backward compatibility.

    Returns:
        Summary dict with counts for each content type.
    """
    path = Path(file_path)
    display_name = filename or path.name

    summary: dict[str, Any] = {
        "filename": display_name,
        "total_pages": 0,
        "parent_chunks": 0,
        "text_chunks": 0,
        "image_chunks": 0,
        "table_chunks": 0,
        "total_chunks": 0,
        "status": "failed",
    }

    try:
        logger.info("Parallel hierarchical ingestion started: %s", display_name)

        text_task = _extract_text(path, display_name)
        image_task = _extract_images(path, display_name)
        table_task = _extract_tables(path, display_name)

        results = await asyncio.gather(
            text_task, image_task, table_task, return_exceptions=True
        )

        for res in results:
            if isinstance(res, Exception):
                raise res

        text_result, image_records, table_records = results
        parent_records, text_records, total_pages = text_result

        summary["total_pages"] = total_pages
        summary["parent_chunks"] = len(parent_records)
        summary["text_chunks"] = len(text_records)
        summary["image_chunks"] = len(image_records)
        summary["table_chunks"] = len(table_records)
        summary["total_chunks"] = len(text_records) + len(image_records) + len(table_records)

        if summary["total_chunks"] == 0 and summary["parent_chunks"] == 0:
            summary["status"] = "completed"
            logger.warning("No chunks of any type produced for %s", display_name)
            return summary

        # Persist parent chunks directly into PostgreSQL relational store
        await _persist_parent_chunks(parent_records)

        # Combine all child records into single lists for Qdrant vector store upsert
        all_records = text_records + image_records + table_records

        chunks_for_upsert = []
        embeddings_list = []

        for rec in all_records:
            chunk_dict = {
                "text": rec["text"],
                "metadata": rec["metadata"],
            }
            if "image_base64" in rec:
                chunk_dict["image_base64"] = rec["image_base64"]

            chunks_for_upsert.append(chunk_dict)
            embeddings_list.append(rec["embedding"])

        if chunks_for_upsert:
            stacked_embeddings = np.stack(embeddings_list)
            qdrant = QdrantService()

            logger.info("Upserting combined batch of %d items to Qdrant...", len(chunks_for_upsert))
            await asyncio.to_thread(qdrant.upsert_chunks, chunks_for_upsert, stacked_embeddings)

        summary["status"] = "completed"
        logger.info(
            "Hierarchical ingestion completed: %s (%d parents, %d text children, %d images, %d tables)",
            display_name,
            summary["parent_chunks"],
            summary["text_chunks"],
            summary["image_chunks"],
            summary["table_chunks"],
        )

    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", display_name, e)
        summary["status"] = "failed"
        summary["error"] = str(e)

    return summary
