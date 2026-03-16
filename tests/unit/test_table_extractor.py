"""
Unit tests for TableExtractor.

All I/O (pdfplumber.open) and Gemini API calls (EmbeddingService.embed_chunks)
are fully mocked — no real PDF file or API key is required.

Tests verify:
  - A valid table is extracted and returned as a record.
  - Tables where every cell is empty are silently skipped.
  - Returned records contain all required keys with correct types.
  - Markdown output has a GFM header-separator row.
  - None cells become empty strings (not the literal "None").
  - Ragged rows (varying column counts) are padded correctly.
  - Embedding shape is (768,).
  - row_count and col_count values are correct.
  - content_type is always "table".
  - A bad table (embed raises) is skipped without crashing.
  - FileNotFoundError is raised for a missing file path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

_DIM = 768


def _fake_embedding() -> np.ndarray:
    """Return a deterministic unit-norm float32 vector."""
    v = np.ones(_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_pdf_mock(tables_per_page: list[list[list[list[str | None]]]]) -> MagicMock:
    """Build a minimal pdfplumber PDF mock.

    Args:
        tables_per_page: Outer list = pages; each element = list of tables
            for that page; each table = list[list[str | None]].
    """
    pages = []
    for page_tables in tables_per_page:
        page = MagicMock()
        page.extract_tables.return_value = page_tables
        pages.append(page)

    pdf = MagicMock()
    pdf.pages = pages
    pdf.__len__ = MagicMock(return_value=len(pages))
    # Make pdf work as a context manager (pdfplumber.open returns one)
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


def _sample_table() -> list[list[str | None]]:
    return [
        ["Name", "Q1", "Q2"],
        ["Revenue", "1.2M", "1.5M"],
        ["Profit", "0.3M", "0.4M"],
    ]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_embed_service():
    """Mock EmbeddingService.embed_chunks returning a unit-norm vector."""
    svc = MagicMock()
    svc.embed_chunks.return_value = [_fake_embedding()]
    return svc


@pytest.fixture()
def extractor(mock_embed_service):
    from app.ingestion.table_extractor import TableExtractor
    return TableExtractor(embed_service=mock_embed_service)


def _open_patch(pdf_mock):
    """Helper: patch pdfplumber.open to return our mock."""
    return patch("app.ingestion.table_extractor.pdfplumber.open", return_value=pdf_mock)


def _exists_patch(value: bool = True):
    return patch("app.ingestion.table_extractor.Path.exists", return_value=value)


# ─── Extraction ───────────────────────────────────────────────────────────────

class TestExtraction:
    def test_valid_table_returned(self, extractor):
        pdf = _make_pdf_mock([[_sample_table()]])
        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")
        assert len(records) == 1

    def test_multiple_tables_on_same_page(self, extractor):
        pdf = _make_pdf_mock([[_sample_table(), _sample_table()]])
        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")
        assert len(records) == 2

    def test_tables_across_multiple_pages(self, extractor):
        pdf = _make_pdf_mock([[_sample_table()], [_sample_table()]])
        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")
        assert len(records) == 2
        assert records[0]["page_number"] == 1
        assert records[1]["page_number"] == 2

    def test_empty_table_skipped(self, extractor):
        """A table where every cell is None/empty must be silently dropped."""
        all_empty = [[None, None], [None, None]]
        pdf = _make_pdf_mock([[all_empty]])
        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")
        assert records == []

    def test_page_with_no_tables_ok(self, extractor):
        pdf = _make_pdf_mock([[]])  # one page, zero tables
        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")
        assert records == []


# ─── Record Schema ────────────────────────────────────────────────────────────

class TestRecordSchema:
    def _get_record(self, extractor) -> dict:
        pdf = _make_pdf_mock([[_sample_table()]])
        with _exists_patch(), _open_patch(pdf):
            return extractor.extract("report.pdf")[0]

    def test_all_required_keys_present(self, extractor):
        record = self._get_record(extractor)
        for key in ("page_number", "table_index", "markdown_text",
                    "embedding", "row_count", "col_count",
                    "content_type", "metadata"):
            assert key in record, f"Missing key: {key}"

    def test_page_number_is_one_based(self, extractor):
        assert self._get_record(extractor)["page_number"] == 1

    def test_table_index_is_zero_based(self, extractor):
        assert self._get_record(extractor)["table_index"] == 0

    def test_content_type_is_table(self, extractor):
        assert self._get_record(extractor)["content_type"] == "table"

    def test_metadata_fields(self, extractor):
        meta = self._get_record(extractor)["metadata"]
        assert meta["content_type"] == "table"
        assert meta["page_number"] == 1
        assert meta["table_index"] == 0
        assert "source_filename" in meta

    def test_row_count_excludes_header(self, extractor):
        # _sample_table has 3 rows: 1 header + 2 data rows
        assert self._get_record(extractor)["row_count"] == 2

    def test_col_count(self, extractor):
        assert self._get_record(extractor)["col_count"] == 3

    def test_embedding_shape(self, extractor):
        assert self._get_record(extractor)["embedding"].shape == (_DIM,)

    def test_embedding_dtype(self, extractor):
        assert self._get_record(extractor)["embedding"].dtype == np.float32


# ─── Markdown Conversion ──────────────────────────────────────────────────────

class TestMarkdownConversion:
    """Tests for the _table_to_markdown static method directly."""

    def _md(self, rows):
        from app.ingestion.table_extractor import TableExtractor
        return TableExtractor._table_to_markdown(rows)

    def test_empty_rows_returns_empty_string(self):
        assert self._md([]) == ""

    def test_all_none_cells_returns_empty_string(self):
        assert self._md([[None, None], [None, None]]) == ""

    def test_has_gfm_separator_row(self):
        md = self._md(_sample_table())
        lines = md.splitlines()
        assert any("---" in line for line in lines), "Expected GFM separator row"

    def test_none_cell_becomes_empty_string_not_literal_none(self):
        rows = [["Header", None], ["Value", None]]
        md = self._md(rows)
        assert "None" not in md

    def test_whitespace_cell_becomes_empty(self):
        rows = [["A", "B"], ["   ", "val"]]
        md = self._md(rows)
        assert "   " not in md

    def test_multiline_cell_collapsed_to_single_line(self):
        rows = [["Col"], ["line1\nline2"]]
        md = self._md(rows)
        assert "\n" not in md.split("\n", 2)[2]  # data row has no embedded newline

    def test_ragged_rows_padded(self):
        """Rows with fewer columns than the widest row must be padded."""
        rows = [["A", "B", "C"], ["only_one"]]
        md = self._md(rows)
        # Every pipe-delimited row should have the same column count
        lines = [l for l in md.splitlines() if l.startswith("|")]
        col_counts = [l.count("|") for l in lines]
        assert len(set(col_counts)) == 1, f"Column counts differ: {col_counts}"

    def test_header_is_first_row(self):
        rows = [["Name", "Score"], ["Alice", "95"]]
        md = self._md(rows)
        first_line = md.splitlines()[0]
        assert "Name" in first_line and "Score" in first_line

    def test_single_row_table_rendered(self):
        """A header-only table (no data rows) should still render."""
        rows = [["Col1", "Col2"]]
        md = self._md(rows)
        assert md != ""
        assert "Col1" in md


# ─── Error Handling ───────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_file_not_found_raises(self, extractor):
        with _exists_patch(False):
            with pytest.raises(FileNotFoundError):
                extractor.extract("no_such_file.pdf")

    def test_embed_error_skips_table_gracefully(self, extractor):
        """If embed_chunks raises for one table, extractor skips it and continues."""
        table1 = _sample_table()
        table2 = _sample_table()
        pdf = _make_pdf_mock([[table1, table2]])

        # First call raises; second call succeeds
        extractor._embed.embed_chunks.side_effect = [
            RuntimeError("Gemini API error"),
            [_fake_embedding()],
        ]

        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")

        assert len(records) == 1
        assert records[0]["table_index"] == 1

    def test_extract_tables_error_skips_page(self, extractor):
        """If extract_tables raises on a page, that page is skipped gracefully."""
        pdf = _make_pdf_mock([[_sample_table()]])
        # Override extract_tables to raise on the first page
        pdf.pages[0].extract_tables.side_effect = RuntimeError("parse error")

        with _exists_patch(), _open_patch(pdf):
            records = extractor.extract("report.pdf")

        assert records == []
