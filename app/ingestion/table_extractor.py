"""
Table extraction for the Enterprise RAG Platform.

Extracts tables from PDF pages using pdfplumber, converts each table to
GitHub-Flavored Markdown, and embeds the Markdown text with the Gemini
EmbeddingService so tables live in the same vector space as text chunks.

Design rationale
----------------
Tables are a common carrier of structured, high-value information in enterprise
documents (financial summaries, specification matrices, SLA tables, etc.).
Plain PDF text extraction via PyMuPDF collapses table cells into unstructured
strings, destroying row/column relationships and making retrieval inaccurate.

pdfplumber uses character-level bounding boxes from the PDF to reconstruct the
cell structure, yielding ``list[list[str | None]]`` where each inner list is
one row.  We convert this structure to Markdown, which:
  - Preserves column alignment in a human-readable format.
  - Can be embedded as text via ``EmbeddingService.embed_chunks()`` (Gemini
    RETRIEVAL_DOCUMENT task type) — tables and text chunks share the same vector
    space, so a text query naturally retrieves relevant tables.
  - Can be stored verbatim in Qdrant payloads for rendering in the UI.

No separate vision-model or OCR step is needed: pdfplumber works purely from
the PDF's internal character positions; it requires that the PDF contains
selectable text (not scanned images).

Requires: pip install pdfplumber
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber

from app.rag.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract and embed tables from PDF files using pdfplumber.

    Iterates through every page of the PDF, detects embedded tables using
    pdfplumber's layout analysis, converts each valid table to GFM Markdown,
    and embeds the Markdown via ``EmbeddingService.embed_chunks()``.  The
    returned records are structurally compatible with text chunk records and
    can be upserted into Qdrant alongside them.

    Usage::

        extractor = TableExtractor()
        records = extractor.extract("annual_report.pdf")
        # Each record has: page_number, table_index, markdown_text,
        #                   embedding, row_count, col_count, content_type,
        #                   metadata.
    """

    def __init__(self, embed_service=None) -> None:
        """Initialise the extractor.

        Args:
            embed_service: Optional ``EmbeddingService`` instance.  If None,
                the module-level singleton (``get_embedding_service()``) is
                used.  Pass a mock here in unit tests to avoid real API calls.
        """
        self._embed = embed_service or get_embedding_service()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        file_path: str | Path,
        source_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract, convert, and embed all tables from a PDF.

        For each page, ``page.extract_tables()`` returns a list of tables,
        where each table is ``list[list[str | None]]``.  Each table is
        converted to Markdown (via ``_table_to_markdown``), and the Markdown
        string is embedded with ``embed_chunks([markdown_text])`` so that
        table records share the same dense vector space as prose chunks.

        A malformed table on one page never aborts the rest — errors are
        caught per-table and logged at WARNING level.

        Args:
            file_path:       Path to the PDF file (str or Path).
            source_filename: Human-readable name stored in metadata.
                             Defaults to the file's basename.

        Returns:
            List of dicts, one per qualifying table, with keys:

            - ``page_number``   (int)       — 1-based page index.
            - ``table_index``   (int)       — 0-based table index on the page.
            - ``markdown_text`` (str)       — GFM Markdown table string.
            - ``embedding``     (np.ndarray)— 768-dim L2-normalised float32
                                             vector from ``embed_chunks()``.
            - ``row_count``     (int)       — number of data rows (excl. header).
            - ``col_count``     (int)       — number of columns.
            - ``content_type``  (str)       — Always ``"table"``.
            - ``metadata``      (dict)      — source_filename, page_number,
                                             table_index, content_type.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError:        If the file cannot be opened by pdfplumber.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        display_name = source_filename or path.name
        records: list[dict[str, Any]] = []

        try:
            pdf = pdfplumber.open(path)
        except Exception as exc:
            raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

        try:
            total_pages = len(pdf.pages)
            logger.info(
                "TableExtractor: scanning %d pages of '%s' for tables",
                total_pages,
                display_name,
            )

            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1  # 1-based

                # extract_tables() returns list[list[list[str | None]]]
                # Each element is one table; each inner list is one row.
                try:
                    tables = page.extract_tables()
                except Exception as exc:
                    logger.warning(
                        "TableExtractor: failed extracting tables from page %d of '%s': %s",
                        page_number,
                        display_name,
                        exc,
                    )
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
                        logger.warning(
                            "TableExtractor: skipping table %d on page %d of '%s': %s",
                            table_index,
                            page_number,
                            display_name,
                            exc,
                        )

        finally:
            pdf.close()

        logger.info(
            "TableExtractor: extracted %d tables from '%s'",
            len(records),
            display_name,
        )
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_table(
        self,
        raw_table: list[list[str | None]],
        page_number: int,
        table_index: int,
        source_filename: str,
    ) -> dict[str, Any] | None:
        """Process a single raw pdfplumber table and return a record or None.

        Steps:
        1. Convert to Markdown via ``_table_to_markdown``.
        2. Skip tables that produce empty Markdown (all-empty cells).
        3. Embed the Markdown string via ``embed_chunks([markdown_text])``.
        4. Assemble the record dict.

        Args:
            raw_table:       Rows from pdfplumber, ``list[list[str | None]]``.
            page_number:     1-based page number.
            table_index:     0-based index of this table within the page.
            source_filename: Display name of the source PDF.

        Returns:
            Dict with table record fields, or ``None`` if the table is empty.
        """
        markdown_text = self._table_to_markdown(raw_table)
        if not markdown_text:
            logger.debug(
                "Skipping empty table %d on page %d", table_index, page_number
            )
            return None

        # ── Embed the Markdown as a text chunk ────────────────────────────────
        # embed_chunks() takes a list[str] and returns list[np.ndarray].
        # Passing a single-element list gives us one 768-dim vector.
        embeddings: list[np.ndarray] = self._embed.embed_chunks([markdown_text])
        embedding: np.ndarray = embeddings[0]

        # Row/column counts: first row is treated as header
        # row_count = data rows only (excludes header)
        num_cols = len(raw_table[0]) if raw_table else 0
        num_rows = max(0, len(raw_table) - 1)  # subtract header row

        metadata = {
            "source_filename": source_filename,
            "page_number": page_number,
            "table_index": table_index,
            "content_type": "table",
        }

        return {
            "page_number": page_number,
            "table_index": table_index,
            "markdown_text": markdown_text,
            "embedding": embedding,
            "row_count": num_rows,
            "col_count": num_cols,
            "content_type": "table",
            "metadata": metadata,
        }

    @staticmethod
    def _table_to_markdown(rows: list[list[str | None]]) -> str:
        """Convert a pdfplumber table (list of rows) to a GFM Markdown string.

        Handles:
        - ``None`` cells (merged cells or missing values) → empty string ``""``.
        - Whitespace-only cells → empty string.
        - Tables where every cell is empty → returns ``""`` (caller skips it).
        - Single-row tables (header only, no data) → still rendered.

        The first row is always treated as the column header.  A GFM separator
        row (``| --- | --- |``) is inserted after the header so the table
        renders correctly in Markdown viewers and is unambiguous to the LLM.

        Args:
            rows: ``list[list[str | None]]`` from ``page.extract_tables()``.

        Returns:
            GFM Markdown table string, or ``""`` if the table has no content.
        """
        if not rows:
            return ""

        def clean(cell: str | None) -> str:
            """Normalise a single cell value."""
            if cell is None:
                return ""
            # Collapse internal newlines (common in multi-line PDF cells)
            cleaned = " ".join(cell.split())
            return cleaned

        # Normalise every row — applies clean() to each cell
        cleaned_rows = [[clean(cell) for cell in row] for row in rows]

        # Guard: skip if every cell across all rows is empty
        if all(cell == "" for row in cleaned_rows for cell in row):
            return ""

        # Determine column width as the maximum number of cells in any row
        # (rows can have ragged lengths if pdfplumber detection is imperfect)
        max_cols = max(len(row) for row in cleaned_rows) if cleaned_rows else 0
        if max_cols == 0:
            return ""

        def pad_row(row: list[str]) -> list[str]:
            """Right-pad a row with empty strings to max_cols."""
            return row + [""] * (max_cols - len(row))

        # ── Build Markdown lines ──────────────────────────────────────────────
        lines: list[str] = []

        # Header row (first row of the table)
        header = pad_row(cleaned_rows[0])
        lines.append("| " + " | ".join(header) + " |")

        # GFM separator row — required for the table to be recognised as a table
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        # Data rows
        for row in cleaned_rows[1:]:
            padded = pad_row(row)
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)
