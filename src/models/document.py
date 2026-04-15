"""
DocumentRecord SQLModel table — one row per ingested document.

RAG Pipeline Position:
  [DOCUMENT_RECORD] is written when a file is ingested.
  Document -> Chunks -> Embeddings -> [DOCUMENT_RECORD] written here
                                   -> ChromaDB (vector chunks stored there)

What concept it teaches:
  Using a content-hash as the primary key (instead of auto-increment) makes
  deduplication trivial: a second upload of the same file produces the same
  id and a harmless "already exists" conflict rather than a duplicate row.

Why this approach over alternatives:
  Auto-increment or UUID PKs would silently create duplicate document rows.
  A content-hash PK (SHA-256 of the raw bytes) is idempotent — re-ingesting
  the same file is a no-op at the DB level.

Where it fits in the RAG pipeline:
  This is the METADATA STORE for ingested documents. ChromaDB stores the
  actual chunk vectors; DocumentRecord stores the file-level metadata
  (name, type, size, chunk count) for the Documents dashboard page.
"""

# NOTE: from __future__ import annotations is intentionally OMITTED.
# See conversation.py for the full explanation.

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class DocumentRecord(SQLModel, table=True):
    """
    Metadata record for a document that has been ingested into the RAG system.

    Attributes:
        id: Content-hash (SHA-256) of the raw file bytes. Acts as a
            deduplication key — re-uploading the same file is idempotent.
        filename: Original filename as uploaded by the user.
        file_type: Extension without dot (e.g. "pdf", "docx", "txt").
        file_size_bytes: Raw file size in bytes for display in the UI.
        chunks_count: Number of chunks the document was split into.
        upload_date: UTC timestamp of the first successful ingest.

    TRADE-OFF: We do NOT store a FK to Conversation/Message here because a
               document can be cited across many conversations. The link lives
               in MessageSource.doc_id (a string reference, not a FK) to keep
               schema evolution simple.
    """

    __tablename__ = "documents"

    # WHY: str primary key (content-hash) rather than auto-increment so that
    #      inserting the same document twice is a PK conflict, not a duplicate.
    #      The caller uses INSERT OR IGNORE / on_conflict_do_nothing to handle
    #      re-uploads gracefully.
    id: str = Field(primary_key=True)  # content-hash doc_id

    filename: str

    file_type: str = Field(default="")

    file_size_bytes: int = Field(default=0)

    chunks_count: int = Field(default=0)

    upload_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
