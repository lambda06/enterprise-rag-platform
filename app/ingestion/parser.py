"""
PDF text extraction for the Enterprise RAG Platform.

Uses PyMuPDF (fitz) to extract text page by page, preserving page numbers
as metadata. Handles empty pages and extraction errors gracefully.

Requires: pip install pymupdf
"""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def parse_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file page by page using PyMuPDF (fitz).

    Each page yields a dictionary with page_number, text, and char_count.
    Pages with no extractable text (e.g., image-only pages) return empty
    text and char_count 0. Extraction errors on individual pages are
    handled gracefully without failing the entire parse.

    Args:
        file_path: Path to the PDF file (str or Path).

    Returns:
        List of dicts, one per page, with keys:
            - page_number: 1-based page index (human-readable)
            - text: Extracted text for the page (empty string if none)
            - char_count: Number of characters in the extracted text

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the path is not a valid PDF or cannot be opened.

    Example:
        >>> pages = parse_pdf("document.pdf")
        >>> pages[0]
        {'page_number': 1, 'text': 'Hello world...', 'char_count': 11}
    """
    # Normalize to Path for consistent handling
    path = Path(file_path)

    # Step 1: Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # Step 2: Validate file extension (basic sanity check)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages: list[dict[str, Any]] = []

    # Step 3: Open the PDF document
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ValueError(f"Could not open PDF '{path}': {e}") from e

    try:
        # Step 4: Iterate over each page (fitz uses 0-based indices)
        for page_index in range(len(doc)):
            page = doc[page_index]

            # Step 5: Extract text; handle errors (e.g., corrupted or image-only pages)
            try:
                text = page.get_text()
            except Exception:
                # Graceful fallback: treat failed extraction as empty page
                text = ""

            # Step 6: Normalize text (strip whitespace, handle None)
            text = (text or "").strip()

            # Step 7: Build page record with metadata (1-based page_number for readability)
            page_record = {
                "page_number": page_index + 1,
                "text": text,
                "char_count": len(text),
            }
            pages.append(page_record)

    finally:
        # Step 8: Always close the document to free file handles and memory
        doc.close()

    return pages
