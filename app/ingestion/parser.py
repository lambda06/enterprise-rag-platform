"""
Multi-format document and PDF text extraction for the Enterprise RAG Platform.

Supports machine-readable and scanned documents (.pdf, .docx, .xlsx, .pptx, etc.)
using IBM Docling as the primary structure-aware document converter, with PyMuPDF
(fitz) retained as a fast fallback/legacy handler for standard PDFs.

Requires: pip install docling pymupdf
"""

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Try importing Docling (if installed)
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("IBM Docling not found (`pip install docling`). Falling back to PyMuPDF for PDF extraction.")


def parse_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract structured text from multi-format documents (.pdf, .docx, .xlsx, .pptx)
    using IBM Docling (`DocumentConverter`).

    Docling parses both machine-readable and OCR/scanned documents, converting complex
    tables and multi-column layouts into high-fidelity Markdown while preserving page
    numbers and structural elements.

    Args:
        file_path: Path to the document (.pdf, .docx, .xlsx, .pptx, etc.).

    Returns:
        List of dicts, one per page (or sheet/section), with keys:
            - page_number: 1-based page index
            - text: Extracted Markdown/text for that page
            - char_count: Number of characters
            - format: File format extension (e.g. 'docx', 'xlsx', 'pdf')

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If Docling is not installed and file is not a PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    ext = path.suffix.lower()

    # If Docling is available, use it for multi-format and scanned documents
    if DOCLING_AVAILABLE:
        try:
            logger.info("Parsing '%s' using IBM Docling DocumentConverter...", path.name)
            converter = DocumentConverter()
            conversion_result = converter.convert(path)
            doc = conversion_result.document

            pages: list[dict[str, Any]] = []
            
            # Check if document has explicit page-level text representation
            # Docling allows exporting full markdown or iterating over items/pages
            if hasattr(doc, "pages") and doc.pages:
                for page_no, page_obj in doc.pages.items():
                    page_text = ""
                    if hasattr(doc, "export_to_markdown"):
                        page_text = str(page_obj.text or "").strip() if hasattr(page_obj, "text") else ""
                    
                    if not page_text and hasattr(doc, "export_to_markdown"):
                        if page_no == 1:
                            page_text = doc.export_to_markdown().strip()

                    # Structured deduplication: Replace table grids with [TABLE_ANCHOR]
                    if hasattr(doc, "tables") and doc.tables:
                        for idx, table_obj in enumerate(doc.tables):
                            try:
                                tbl_md = table_obj.export_to_markdown().strip() if hasattr(table_obj, "export_to_markdown") else str(table_obj).strip()
                                if tbl_md and tbl_md in page_text:
                                    anchor = f"\n\n[TABLE_ANCHOR: Table {idx} on Page {int(page_no)} — See dedicated Table Summary for structured metrics]\n\n"
                                    page_text = page_text.replace(tbl_md, anchor)
                            except Exception:
                                pass

                    pages.append({
                        "page_number": int(page_no),
                        "text": page_text,
                        "char_count": len(page_text),
                        "format": ext.lstrip("."),
                    })
            else:
                full_md = doc.export_to_markdown().strip()
                # Structured deduplication: Replace table grids with [TABLE_ANCHOR]
                if hasattr(doc, "tables") and doc.tables:
                    for idx, table_obj in enumerate(doc.tables):
                        try:
                            tbl_md = table_obj.export_to_markdown().strip() if hasattr(table_obj, "export_to_markdown") else str(table_obj).strip()
                            if tbl_md and tbl_md in full_md:
                                anchor = f"\n\n[TABLE_ANCHOR: Table {idx} on Page 1 — See dedicated Table Summary for structured metrics]\n\n"
                                full_md = full_md.replace(tbl_md, anchor)
                        except Exception:
                            pass

                pages.append({
                    "page_number": 1,
                    "text": full_md,
                    "char_count": len(full_md),
                    "format": ext.lstrip("."),
                })

            return pages
        except Exception as exc:
            logger.warning("Docling conversion failed for '%s' (%s). Attempting fallback...", path.name, exc)
            if ext != ".pdf":
                raise RuntimeError(f"Failed to parse non-PDF document '{path.name}' with Docling: {exc}") from exc

    # Fallback to PyMuPDF for PDFs if Docling is unavailable or failed on PDF
    if ext == ".pdf":
        return parse_pdf(path)
    else:
        raise RuntimeError(f"Cannot parse format '{ext}' without IBM Docling (`pip install docling`).")


def parse_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file page by page using PyMuPDF (fitz).

    Retained for backward compatibility and as a fast fallback for standard PDFs.

    Args:
        file_path: Path to the PDF file (str or Path).

    Returns:
        List of dicts, one per page, with keys:
            - page_number: 1-based page index
            - text: Extracted text for the page
            - char_count: Number of characters in the extracted text
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages: list[dict[str, Any]] = []

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ValueError(f"Could not open PDF '{path}': {e}") from e

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            try:
                text = page.get_text()
            except Exception:
                text = ""

            text = (text or "").strip()
            pages.append({
                "page_number": page_index + 1,
                "text": text,
                "char_count": len(text),
            })
    finally:
        doc.close()

    return pages

