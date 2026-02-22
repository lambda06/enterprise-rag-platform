"""
Text chunking for the Enterprise RAG Platform.

Splits parsed page content into chunks suitable for embedding and retrieval,
preserving metadata for citations and source attribution.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict[str, Any]],
    source_filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Split parsed page content into chunks with metadata.

    Uses RecursiveCharacterTextSplitter to split on natural boundaries
    (paragraphs, sentences, words) while respecting chunk_size and overlap.

    Args:
        pages: List of page dicts from parser.parse_pdf() with keys
               page_number, text, char_count.
        source_filename: Original file name for source attribution.
        chunk_size: Maximum characters per chunk (default 1000).
        chunk_overlap: Overlap between adjacent chunks (default 200).

    Returns:
        List of dicts with keys:
            - text: Chunk content
            - metadata: Dict with page_number, chunk_index, source_filename
    """
    # Build LangChain Documents from pages (one per page preserves page boundaries)
    documents: list[Document] = []
    for page in pages:
        text = page.get("text", "") or ""
        page_number = page.get("page_number", 0)
        # Skip empty pages; they produce no useful chunks
        if not text.strip():
            continue
        doc = Document(
            page_content=text,
            metadata={
                "page_number": page_number,
                "source_filename": source_filename,
            },
        )
        documents.append(doc)

    # RecursiveCharacterTextSplitter: splits on "\n\n", "\n", " ", "" in order
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split documents; metadata (page_number, source_filename) is preserved per chunk
    chunks = splitter.split_documents(documents)

    # Add chunk_index and convert to output format
    result: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        result.append({
            "text": chunk.page_content,
            "metadata": {
                "page_number": chunk.metadata.get("page_number", 0),
                "chunk_index": idx,
                "source_filename": source_filename,
            },
        })

    return result
