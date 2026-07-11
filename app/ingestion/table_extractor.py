"""
Table extraction and summarization for the Enterprise RAG Platform.

Extracts structured tables from multi-format documents (.pdf, .docx, .xlsx, .pptx)
using IBM Docling (`DocumentConverter`) as the primary structure-aware extractor,
with `pdfplumber` retained as a fast fallback/legacy handler for standard PDFs.

Each extracted table is summarized via Gemini 2.5 Flash Lite (`_generate_table_summary`).
The natural language summary is indexed into the vector store (`embed_chunks`) for high
semantic recall and BM25 exact matching, while the full, exact GitHub-Flavored Markdown
table grid (`raw_table_content`) is stored inside the payload for precise LLM generation.

Requires: pip install docling pdfplumber google-genai
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber

from app.rag.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

# Try importing Docling (if installed)
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("IBM Docling not found (`pip install docling`). Falling back to pdfplumber for table extraction.")


class TableExtractor:
    """Extract, summarize, and embed tables from documents.

    Iterates through document tables using IBM Docling (or pdfplumber fallback),
    generates a 3-sentence summary of each table using Gemini 2.5 Flash Lite,
    embeds the summary via `EmbeddingService.embed_chunks()`, and returns
    records ready for upsert into Qdrant alongside text chunks.
    """

    def __init__(self, embed_service=None) -> None:
        """Initialise the extractor.

        Args:
            embed_service: Optional `EmbeddingService` instance. If None,
                the module-level singleton (`get_embedding_service()`) is used.
        """
        self._embed = embed_service or get_embedding_service()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        file_path: str | Path,
        source_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract, summarize, and embed all tables from a document.

        Args:
            file_path: Path to the document (.pdf, .docx, .xlsx, etc.).
            source_filename: Human-readable name stored in metadata.

        Returns:
            List of dicts, one per qualifying table, containing both summary (`text`)
            and full raw markdown (`raw_table_content` / `markdown_text`).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        display_name = source_filename or path.name
        records: list[dict[str, Any]] = []

        ext = path.suffix.lower()

        # Step 1: Try Docling if available
        if DOCLING_AVAILABLE:
            try:
                logger.info("TableExtractor: scanning '%s' using IBM Docling...", display_name)
                converter = DocumentConverter()
                conversion_result = converter.convert(path)
                doc = conversion_result.document

                # If Docling detected explicit tables
                if hasattr(doc, "tables") and doc.tables:
                    for idx, table_obj in enumerate(doc.tables):
                        try:
                            md_text = table_obj.export_to_markdown().strip() if hasattr(table_obj, "export_to_markdown") else str(table_obj).strip()
                            if not md_text:
                                continue
                            
                            # Page number if available, else 1
                            page_no = getattr(table_obj, "page_number", 1) or 1
                            record = self._process_markdown_table(
                                markdown_text=md_text,
                                page_number=int(page_no),
                                table_index=idx,
                                source_filename=display_name,
                                num_rows=md_text.count("\n"),
                                num_cols=md_text.split("\n")[0].count("|") - 1 if "|" in md_text else 1,
                            )
                            if record:
                                records.append(record)
                        except Exception as exc:
                            logger.warning("TableExtractor: skipping Docling table %d (%s)", idx, exc)
                    
                    if records:
                        return records
            except Exception as exc:
                logger.warning("TableExtractor: Docling extraction failed (%s). Attempting fallback...", exc)
                if ext != ".pdf":
                    raise RuntimeError(f"Failed extracting tables from non-PDF '{path.name}' via Docling: {exc}") from exc

        # Step 2: Fallback to pdfplumber for standard PDFs
        if ext == ".pdf":
            return self._extract_pdfplumber(path, display_name)
        else:
            raise RuntimeError(f"Cannot extract tables from '{ext}' without IBM Docling (`pip install docling`).")

    def _extract_pdfplumber(self, path: Path, display_name: str) -> list[dict[str, Any]]:
        """Extract tables via pdfplumber (legacy/fast fallback for PDFs)."""
        records: list[dict[str, Any]] = []
        try:
            pdf = pdfplumber.open(path)
        except Exception as exc:
            raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

        try:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                try:
                    tables = page.extract_tables()
                except Exception as exc:
                    logger.warning("TableExtractor: failed page %d (%s)", page_number, exc)
                    continue

                for table_index, raw_table in enumerate(tables):
                    try:
                        record = self._process_table(
                            raw_table=raw_table,
                            page_number=page_number,
                            table_index=table_index,
                            source_filename=display_name,
                        )
                        if record is not None:
                            records.append(record)
                    except Exception as exc:
                        logger.warning("TableExtractor: skipping table %d on page %d (%s)", table_index, page_number, exc)
        finally:
            pdf.close()

        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_table(
        self,
        raw_table: list[list[str | None]],
        page_number: int,
        table_index: int,
        source_filename: str,
    ) -> dict[str, Any] | None:
        """Process a single raw pdfplumber table and return a record or None."""
        markdown_text = self._table_to_markdown(raw_table)
        if not markdown_text:
            return None

        num_cols = len(raw_table[0]) if raw_table else 0
        num_rows = max(0, len(raw_table) - 1)

        return self._process_markdown_table(
            markdown_text=markdown_text,
            page_number=page_number,
            table_index=table_index,
            source_filename=source_filename,
            num_rows=num_rows,
            num_cols=num_cols,
        )

    def _process_markdown_table(
        self,
        markdown_text: str,
        page_number: int,
        table_index: int,
        source_filename: str,
        num_rows: int,
        num_cols: int,
    ) -> dict[str, Any]:
        """Summarize a Markdown table via Gemini and produce the embedding record."""
        # ── Generate Table Summary via Gemini 2.5 Flash Lite ──────────────────
        summary: str = self._generate_table_summary(markdown_text)

        # ── Embed the summary directly as a text chunk ────────────────────────
        embeddings: list[np.ndarray] = self._embed.embed_chunks([summary])
        embedding: np.ndarray = embeddings[0]

        metadata = {
            "source_filename": source_filename,
            "page_number": page_number,
            "table_index": table_index,
            "content_type": "table",
            "raw_table_content": markdown_text,
            "text": summary,
        }

        return {
            "page_number": page_number,
            "table_index": table_index,
            "markdown_text": markdown_text,
            "text": summary,
            "embedding": embedding,
            "row_count": num_rows,
            "col_count": num_cols,
            "content_type": "table",
            "metadata": metadata,
        }

    def _generate_table_summary(self, markdown_table: str) -> str:
        """Generate a natural language summary of the table using Gemini 2.5 Flash Lite."""
        try:
            from google import genai
            from app.core.config import get_settings

            settings = get_settings()
            if not settings.gemini.api_key:
                # Structural fallback summary if API key is absent
                first_line = markdown_table.split("\n")[0] if markdown_table else "Data table"
                return f"Table Summary: {first_line}\n\nRaw Table:\n{markdown_table[:300]}..."

            client = genai.Client(api_key=settings.gemini.api_key)
            prompt = (
                "Summarize this structured table in 3-4 clear sentences. Highlight key trends, "
                "outliers, column names, and important figures so the table can be easily searched via keyword and semantic search:\n\n"
                f"{markdown_table}"
            )
            response = client.models.generate_content(
                model=settings.gemini.generation_model or "gemini-3.1-flash-lite-preview",
                contents=prompt,
            )
            summary = (response.text or "").strip()
            return summary if summary else f"Table Summary:\n{markdown_table[:300]}..."
        except Exception as exc:
            logger.warning("TableExtractor: summary generation failed (%s). Using fallback.", exc)
            return f"Table Summary:\n{markdown_table[:300]}..."

    @staticmethod
    def _table_to_markdown(rows: list[list[str | None]]) -> str:
        """Convert a pdfplumber table (list of rows) to a GFM Markdown string."""
        if not rows:
            return ""

        def clean(cell: str | None) -> str:
            if cell is None:
                return ""
            return " ".join(cell.split())

        cleaned_rows = [[clean(cell) for cell in row] for row in rows]
        if all(cell == "" for row in cleaned_rows for cell in row):
            return ""

        max_cols = max(len(row) for row in cleaned_rows) if cleaned_rows else 0
        if max_cols == 0:
            return ""

        def pad_row(row: list[str]) -> list[str]:
            return row + [""] * (max_cols - len(row))

        lines: list[str] = []
        header = pad_row(cleaned_rows[0])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in cleaned_rows[1:]:
            padded = pad_row(row)
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)
