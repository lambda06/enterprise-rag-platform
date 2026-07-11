"""
Unit tests for `ParentChunk` SQLAlchemy ORM model in `app/models/document.py`.
"""

from datetime import datetime, timezone
from app.models.document import ParentChunk


def test_parent_chunk_instantiation_and_properties():
    """Verify `ParentChunk` can be instantiated and that the `.text` property returns `.content`."""
    chunk = ParentChunk(
        id="doc_abc_parent_0",
        document_id="doc_abc",
        source_filename="report.pdf",
        page_number=2,
        section_title="Financial Highlights",
        content="Full parent chunk content ~2,000 chars going here...",
        metadata_json={"author": "Finance Team", "file_type": ".pdf"},
        created_at=datetime.now(timezone.utc),
    )

    assert chunk.id == "doc_abc_parent_0"
    assert chunk.document_id == "doc_abc"
    assert chunk.source_filename == "report.pdf"
    assert chunk.page_number == 2
    assert chunk.section_title == "Financial Highlights"
    assert chunk.content == "Full parent chunk content ~2,000 chars going here..."
    assert chunk.text == chunk.content
    assert chunk.metadata_json["author"] == "Finance Team"
    assert chunk.__tablename__ == "parent_chunks"
