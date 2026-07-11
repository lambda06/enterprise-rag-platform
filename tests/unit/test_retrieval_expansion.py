"""
Unit tests for Stage 3 relational context expansion in RetrievalService (`app.rag.retrieval`).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.rag.retrieval import RetrievalService
from app.models.document import ParentChunk


@pytest.mark.asyncio
async def test_expand_parents_empty_list():
    service = RetrievalService()
    result = await service._expand_parents([])
    assert result == []


@pytest.mark.asyncio
async def test_expand_parents_no_parent_ids():
    """Verify chunks without parent_id (e.g. atomic table/image summaries) pass through unchanged."""
    service = RetrievalService()
    chunks = [
        {"text": "Table Summary: Q3 Revenue", "metadata": {"content_type": "table"}},
        {"text": "Image Caption: Architecture diagram", "metadata": {"content_type": "image"}},
    ]
    result = await service._expand_parents(chunks)
    assert len(result) == 2
    assert result[0]["text"] == "Table Summary: Q3 Revenue"
    assert result[0]["metadata"].get("expanded_from_child") is None
    assert result[1]["text"] == "Image Caption: Architecture diagram"


@pytest.mark.asyncio
async def test_expand_parents_with_postgres_lookup():
    """Verify child chunks with parent_id expand to their full ParentChunk context from PostgreSQL."""
    service = RetrievalService()
    
    child_chunks = [
        {
            "text": "Child snippet about cloud enterprise adoption in Q3.",
            "metadata": {
                "parent_id": "doc_123_parent_0",
                "content_type": "text",
                "source_filename": "report.pdf",
            },
        },
        {
            "text": "Table Summary: Q3 metrics",
            "metadata": {
                "content_type": "table",
                "source_filename": "report.pdf",
            },
        },
    ]

    # Mock ParentChunk row returned from DB
    mock_parent = ParentChunk(
        id="doc_123_parent_0",
        document_id="doc_123",
        source_filename="report.pdf",
        page_number=4,
        section_title="Section 2: Revenue",
        content="Section 2: Revenue\n\nIn Q3, North American operations saw a significant surge driven by cloud enterprise adoption. European markets remained steady despite currency headwinds.",
        metadata_json={},
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value = [mock_parent]
    mock_db.execute.return_value = mock_result

    # Mock async_session context manager
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_db
    mock_session_cm.__aexit__.return_value = None

    with patch("app.db.session.async_session", return_value=mock_session_cm):
        expanded = await service._expand_parents(child_chunks)

    assert len(expanded) == 2
    
    # Check that child chunk was expanded
    assert expanded[0]["text"] == mock_parent.content
    assert expanded[0]["metadata"]["expanded_from_child"] is True
    assert expanded[0]["metadata"]["child_text"] == "Child snippet about cloud enterprise adoption in Q3."
    assert expanded[0]["metadata"]["parent_id"] == "doc_123_parent_0"

    # Check that table summary without parent_id remained unexpanded
    assert expanded[1]["text"] == "Table Summary: Q3 metrics"
    assert expanded[1]["metadata"].get("expanded_from_child") is None
