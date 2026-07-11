"""
Unit tests for hierarchical chunking (`chunk_pages_hierarchical` in `chunker.py`).
"""

import pytest
from app.ingestion.chunker import chunk_pages, chunk_pages_hierarchical


def test_chunk_pages_legacy_compatibility():
    """Ensure `chunk_pages` (single-tier splitter) still works exactly as expected."""
    pages = [
        {"page_number": 1, "text": "This is page one text that is reasonably long for testing."},
        {"page_number": 2, "text": "This is page two text."},
    ]
    chunks = chunk_pages(pages, "sample.pdf", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["source_filename"] == "sample.pdf"
    assert chunks[0]["metadata"]["page_number"] == 1


def test_chunk_pages_hierarchical_returns_parent_and_child():
    """Verify that hierarchical chunking returns both parent (~2000 chars) and child (~350 chars) records."""
    long_text = (
        "Enterprise RAG systems require two tiers of context for maximum accuracy. "
        "First, small child chunks with dense semantic embeddings are indexed in Qdrant. "
        "When a user query arrives, dense and sparse retrieval find these child chunks. "
        "Then, Stage 2 cross-encoder reranking scores the candidate child chunks against the query. "
        "Once the winning child chunks are selected, the pipeline reads their parent_id metadata. "
        "It then queries PostgreSQL to retrieve the complete 2,000 character parent chunk. "
        "This parent chunk provides the LLM with all surrounding paragraphs, headers, and tables. "
        "Without parent chunks, the LLM often suffers from context fragmentation and hallucination. "
        "Let us repeat this text so it spans across multiple child chunks during our test. "
        "Enterprise RAG systems require two tiers of context for maximum accuracy. "
        "First, small child chunks with dense semantic embeddings are indexed in Qdrant. "
        "When a user query arrives, dense and sparse retrieval find these child chunks. "
        "Then, Stage 2 cross-encoder reranking scores the candidate child chunks against the query."
    )
    pages = [{"page_number": 1, "text": long_text, "section_title": "Architecture Overview"}]

    parent_records, child_records = chunk_pages_hierarchical(
        pages=pages,
        source_filename="architecture_guide.docx",
        document_id="doc_12345",
        parent_chunk_size=600,
        parent_chunk_overlap=100,
        child_chunk_size=150,
        child_chunk_overlap=30,
    )

    # We should get parents and more children than parents
    assert len(parent_records) >= 2
    assert len(child_records) > len(parent_records)

    # Verify Parent record schema
    p0 = parent_records[0]
    assert p0["id"] == "doc_12345_parent_0"
    assert p0["document_id"] == "doc_12345"
    assert p0["source_filename"] == "architecture_guide.docx"
    assert p0["page_number"] == 1
    assert p0["section_title"] == "Architecture Overview"
    assert "Enterprise RAG" in p0["content"]
    assert p0["metadata_json"]["parent_index"] == 0

    # Verify Child record schema & parent linking
    c0 = child_records[0]
    assert "text" in c0
    assert c0["metadata"]["parent_id"] == "doc_12345_parent_0"
    assert c0["metadata"]["document_id"] == "doc_12345"
    assert c0["metadata"]["content_type"] == "text"
    assert c0["metadata"]["chunk_index"] == 0


def test_chunk_pages_hierarchical_generates_doc_id_if_none():
    """If `document_id` is None, a deterministic ID derived from source_filename is assigned."""
    pages = [{"page_number": 1, "text": "Quick test for auto-generated document ID."}]
    parents, children = chunk_pages_hierarchical(
        pages=pages,
        source_filename="auto_test.pdf",
        document_id=None,
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0]["document_id"].startswith("doc_")
    assert children[0]["metadata"]["document_id"] == parents[0]["document_id"]
    assert children[0]["metadata"]["parent_id"] == parents[0]["id"]
