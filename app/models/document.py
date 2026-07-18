"""
SQLAlchemy async ORM model for persisting parent document chunks.

In our "Rerank-Then-Expand" hierarchical RAG architecture:
- Small, dense child chunks (~350 chars) live in Qdrant (`DENSE_VECTOR_NAME`,
  `SPARSE_VECTOR_NAME`) where their specific, focused semantic signal ensures
  high recall and accurate cross-encoder reranking.
- Large, comprehensive parent chunks (~2,000 chars) live here in PostgreSQL
  (`parent_chunks` table).

When child chunks win the Jina Cross-Encoder reranker step during query retrieval,
we execute a fast primary-key lookup in this table by `id` (`child_metadata.parent_id`)
to expand the context window before passing the retrieved text to the LLM.

Using PostgreSQL as the relational Document Store ensures ACID reliability and
keeps our Qdrant vector memory footprint lightweight.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.conversation import Base


class ParentChunk(Base):
    """
    Relational storage model for a parent document chunk (~2,000 chars).

    Column layout
    -------------
    id              : String PK — unique parent identifier across the system,
                      typically formatted as `{document_id}_parent_{index}`.
    document_id     : Unique identifier of the source document (indexed).
    source_filename : Human-readable name of the source file (indexed).
    page_number     : 1-based page number where this parent chunk begins.
    section_title   : Optional heading or section title for context headers.
    content         : The full text of the parent chunk (~2,000 chars).
    metadata_json   : Optional JSON dict for extra metadata (file_type, etc.).
    created_at      : Server-side timestamp set at insert time.
    """

    __tablename__ = "parent_chunks"

    # ---------------------------------------------------------------------- #
    # Primary key                                                             #
    # ---------------------------------------------------------------------- #

    id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        comment="Unique parent chunk identifier, e.g. 'doc_abc_parent_0'.",
    )

    # ---------------------------------------------------------------------- #
    # Core metadata & indexing                                                #
    # ---------------------------------------------------------------------- #

    document_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Unique identifier for the source document.",
    )

    source_filename: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True,
        comment="Display name of the ingested source document.",
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="1-based page number where this parent chunk starts.",
    )

    section_title: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        default=None,
        comment="Optional heading or section title.",
    )

    # ---------------------------------------------------------------------- #
    # Content & JSON metadata                                                 #
    # ---------------------------------------------------------------------- #

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full text of the parent chunk (~2,000 chars).",
    )

    metadata_json: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        comment="Optional dictionary of additional metadata (file_type, author, etc.).",
    )

    # ---------------------------------------------------------------------- #
    # Timestamps                                                              #
    # ---------------------------------------------------------------------- #

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Timestamp when this parent chunk was persisted into the document store.",
    )

    @property
    def text(self) -> str:
        """Alias for `content` to maintain structural parity with dict records."""
        return self.content

    # Composite index for document cleanup and fast lookups
    __table_args__ = (
        Index("ix_parent_chunks_doc_id_page", "document_id", "page_number"),
    )
