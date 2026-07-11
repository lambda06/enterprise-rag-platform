"""
Unit tests for document parsing (parser.py).

Tests verify:
  - parse_pdf properly extracts text using PyMuPDF (fitz) and raises on bad extension.
  - parse_document delegates to Docling (DocumentConverter) when available.
  - parse_document falls back to parse_pdf for .pdf when Docling fails or is unavailable.
  - parse_document raises clear error when non-PDF is provided and Docling is unavailable.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.parser import parse_document, parse_pdf


# ─── Helpers & Mocks ──────────────────────────────────────────────────────────

def _make_fitz_doc_mock(pages_text: list[str]) -> MagicMock:
    """Build a mock fitz.Document yielding the specified text per page."""
    doc = MagicMock()
    doc.__len__.return_value = len(pages_text)
    
    pages = []
    for txt in pages_text:
        page = MagicMock()
        page.get_text.return_value = txt
        pages.append(page)
        
    doc.__getitem__.side_effect = lambda idx: pages[idx]
    return doc


def _make_docling_result_mock(pages_dict: dict[int, str] | None = None, full_md: str = "") -> MagicMock:
    """Build a mock DocumentConverter conversion result."""
    result = MagicMock()
    doc = MagicMock()
    
    if pages_dict:
        # Mock doc.pages dictionary
        pages_map = {}
        for p_no, txt in pages_dict.items():
            page_obj = MagicMock()
            page_obj.text = txt
            pages_map[p_no] = page_obj
        doc.pages = pages_map
        doc.export_to_markdown.side_effect = lambda: full_md or list(pages_dict.values())[0]
    else:
        doc.pages = {}
        doc.export_to_markdown.return_value = full_md
        
    result.document = doc
    return result


# ─── Tests for parse_pdf ──────────────────────────────────────────────────────

def test_parse_pdf_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_pdf(non_existent)


def test_parse_pdf_invalid_extension(tmp_path: Path) -> None:
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError, match="Expected a PDF file"):
        parse_pdf(txt_file)


@patch("fitz.open")
def test_parse_pdf_success(mock_fitz_open: MagicMock, tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy")
    
    mock_doc = _make_fitz_doc_mock(["Page 1 content", "Page 2 content"])
    mock_fitz_open.return_value = mock_doc
    
    records = parse_pdf(pdf_path)
    assert len(records) == 2
    assert records[0] == {"page_number": 1, "text": "Page 1 content", "char_count": 14}
    assert records[1] == {"page_number": 2, "text": "Page 2 content", "char_count": 14}
    mock_doc.close.assert_called_once()


# ─── Tests for parse_document ─────────────────────────────────────────────────

def test_parse_document_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.docx"
    with pytest.raises(FileNotFoundError, match="Document file not found"):
        parse_document(non_existent)


@patch("app.ingestion.parser.DOCLING_AVAILABLE", True)
@patch("app.ingestion.parser.DocumentConverter", create=True)
def test_parse_document_docling_multi_page(mock_converter_cls: MagicMock, tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    docx_path.write_text("dummy")
    
    mock_converter = MagicMock()
    mock_converter_cls.return_value = mock_converter
    mock_converter.convert.return_value = _make_docling_result_mock(
        pages_dict={1: "# Header\nPage 1 text", 2: "## Section 2\nPage 2 text"}
    )
    
    records = parse_document(docx_path)
    assert len(records) == 2
    assert records[0] == {
        "page_number": 1,
        "text": "# Header\nPage 1 text",
        "char_count": 20,
        "format": "docx",
    }
    assert records[1] == {
        "page_number": 2,
        "text": "## Section 2\nPage 2 text",
        "char_count": 24,
        "format": "docx",
    }


@patch("app.ingestion.parser.DOCLING_AVAILABLE", True)
@patch("app.ingestion.parser.DocumentConverter", create=True)
def test_parse_document_docling_single_continuous(mock_converter_cls: MagicMock, tmp_path: Path) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    xlsx_path.write_text("dummy")
    
    mock_converter = MagicMock()
    mock_converter_cls.return_value = mock_converter
    # No pages dict, just full markdown export
    mock_converter.convert.return_value = _make_docling_result_mock(full_md="| A | B |\n| --- | --- |\n| 1 | 2 |")
    
    records = parse_document(xlsx_path)
    assert len(records) == 1
    assert records[0]["page_number"] == 1
    assert "| A | B |" in records[0]["text"]
    assert records[0]["format"] == "xlsx"


@patch("app.ingestion.parser.DOCLING_AVAILABLE", False)
@patch("app.ingestion.parser.parse_pdf")
def test_parse_document_docling_unavailable_pdf_fallback(mock_parse_pdf: MagicMock, tmp_path: Path) -> None:
    pdf_path = tmp_path / "legacy.pdf"
    pdf_path.write_text("dummy")
    
    mock_parse_pdf.return_value = [{"page_number": 1, "text": "fallback text", "char_count": 13}]
    
    records = parse_document(pdf_path)
    assert len(records) == 1
    assert records[0]["text"] == "fallback text"
    mock_parse_pdf.assert_called_once_with(pdf_path)


@patch("app.ingestion.parser.DOCLING_AVAILABLE", False)
def test_parse_document_docling_unavailable_non_pdf_raises(tmp_path: Path) -> None:
    docx_path = tmp_path / "unsupported.docx"
    docx_path.write_text("dummy")
    
    with pytest.raises(RuntimeError, match="Cannot parse format '.docx' without IBM Docling"):
        parse_document(docx_path)
