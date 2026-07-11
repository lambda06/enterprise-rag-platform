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


def chunk_pages_hierarchical(
    pages: list[dict[str, Any]],
    source_filename: str,
    document_id: str | None = None,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 350,
    child_chunk_overlap: int = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split parsed pages hierarchically into Parent (~2,000 chars) and Child (~350 chars) chunks.

    In our "Rerank-Then-Expand" architecture:
    - Stage 1 (Parent Splitter): Breaks pages into large, comprehensive ~2,000-character chunks
      destined for relational storage (`ParentChunk` table in PostgreSQL).
    - Stage 2 (Child Splitter): Subdivides each parent chunk into focused ~350-character chunks
      destined for dense vector embedding and indexing in Qdrant. Each child's metadata contains
      `parent_id`, allowing Stage 2 retrieval/reranking to expand back to the exact parent text.

    Args:
        pages: List of page dicts from parser (`page_number`, `text`, etc.).
        source_filename: Original file name for source attribution.
        document_id: Optional unique document ID; if None, derived deterministically
                     from `source_filename`.
        parent_chunk_size: Maximum characters per parent chunk (default 2,000).
        parent_chunk_overlap: Overlap between adjacent parent chunks (default 200).
        child_chunk_size: Maximum characters per child chunk (default 350).
        child_chunk_overlap: Overlap between adjacent child chunks within a parent (default 50).

    Returns:
        Tuple `(parent_records, child_records)` where:
        - `parent_records`: List of dicts matching `ParentChunk` ORM fields (`id`, `document_id`,
          `source_filename`, `page_number`, `section_title`, `content`, `metadata_json`).
        - `child_records`: List of dicts matching Qdrant chunk structure (`text`, `metadata` with
          `parent_id`, `document_id`, `content_type: "text"`).
    """
    import hashlib

    if not document_id:
        doc_hash = hashlib.sha256(source_filename.encode("utf-8")).hexdigest()[:12]
        document_id = f"doc_{doc_hash}"

    # Build LangChain Documents from pages
    documents: list[Document] = []
    for page in pages:
        text = page.get("text", "") or ""
        page_number = page.get("page_number", 1) or 1
        section_title = page.get("section_title") or page.get("title") or None
        if not text.strip():
            continue
        doc = Document(
            page_content=text,
            metadata={
                "page_number": page_number,
                "section_title": section_title,
                "source_filename": source_filename,
            },
        )
        documents.append(doc)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_docs = parent_splitter.split_documents(documents)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_records: list[dict[str, Any]] = []
    child_records: list[dict[str, Any]] = []
    global_child_idx = 0

    for parent_idx, p_doc in enumerate(parent_docs):
        parent_id = f"{document_id}_parent_{parent_idx}"
        page_num = p_doc.metadata.get("page_number", 1) or 1
        sec_title = p_doc.metadata.get("section_title")

        parent_record = {
            "id": parent_id,
            "document_id": document_id,
            "source_filename": source_filename,
            "page_number": page_num,
            "section_title": sec_title,
            "content": p_doc.page_content,
            "metadata_json": {
                "source_filename": source_filename,
                "page_number": page_num,
                "parent_index": parent_idx,
            },
        }
        parent_records.append(parent_record)

        # Stage 2: Split this parent chunk's text into child chunks
        child_doc_input = Document(
            page_content=p_doc.page_content,
            metadata=p_doc.metadata,
        )
        c_docs = child_splitter.split_documents([child_doc_input])

        for child_idx_in_parent, c_doc in enumerate(c_docs):
            child_record = {
                "text": c_doc.page_content,
                "metadata": {
                    "page_number": page_num,
                    "chunk_index": global_child_idx,
                    "child_index": child_idx_in_parent,
                    "source_filename": source_filename,
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "content_type": "text",
                },
            }
            child_records.append(child_record)
            global_child_idx += 1

    return parent_records, child_records
