"""
Conversation SQLModel table — one row per chat session.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                          |
                                    [CONVERSATION] <-> [MESSAGE]
                                          |
                                     (persisted in SQLite via SQLModel)

What concept it teaches:
  SQLModel unifies Pydantic validation and SQLAlchemy ORM table definitions in
  one class. The same Conversation class validates API input AND maps to the
  'conversations' DB table — no duplication.

Why this approach over alternatives:
  Separate SQLAlchemy model + Pydantic schema would require keeping two class
  hierarchies in sync. SQLModel collapses that to one class, which is ideal for
  a project this size.

Where it fits in the RAG pipeline:
  This is the ROOT of the chat-history tree. Each Conversation has many Messages
  (children), which have many MessageSources (grandchildren). Deleting a
  Conversation cascades through both child levels.
"""

# NOTE: from __future__ import annotations is intentionally OMITTED from this
# file. SQLModel evaluates field types at class-definition time to build
# SQLAlchemy Column objects. Deferring annotations (PEP 563) causes those
# evaluations to see strings instead of types, breaking Relationship() resolution
# in SQLModel 0.0.24. The rest of the project uses the import, but model files
# must not.

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

# TYPE_CHECKING guard: these imports only happen during static analysis (mypy,
# pyright) — not at runtime. This prevents circular imports between models
# while still giving type checkers full knowledge of the relationship types.
if TYPE_CHECKING:
    from src.models.message import Message


class Conversation(SQLModel, table=True):
    """
    A single chat session — maps to the 'conversations' SQLite table.

    Attributes:
        id: UUID primary key generated in Python (not the DB) so we know the
            ID before the first commit.
        title: Short human-readable label, defaults to "New Chat".
        pinned: Whether the user has starred/pinned this conversation.
        created_at: UTC timestamp of creation.
        updated_at: UTC timestamp of last modification (set on message add).
        share_token: Optional opaque token for read-only public sharing.
        messages: ORM relationship to Message rows (lazy-loaded list).

    TRADE-OFF: We store created_at / updated_at in Python (default_factory)
               rather than DB-side DEFAULT CURRENT_TIMESTAMP so that:
               1. The values are timezone-aware Python datetimes, not naïve
                  SQLite TEXT strings.
               2. Tests can inspect them without parsing raw DB values.
    """

    __tablename__ = "conversations"

    # WHY: str UUID (not int) primary key — globally unique across shards/exports
    #      and readable in logs without joining to another table.
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    title: str = Field(default="New Chat", max_length=200)

    # WHY: bool column with explicit default=False. SQLModel maps this to
    #      INTEGER(0/1) in SQLite, which is the standard SQLite boolean encoding.
    pinned: bool = Field(default=False)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # WHY: Nullable + indexed — most conversations are private (NULL), but
    #      shared ones need fast lookup by token without a full table scan.
    share_token: Optional[str] = Field(default=None, index=True)

    # PATTERN: cascade_delete=True tells SQLModel to include
    #          ON DELETE CASCADE on the Message.conversation_id FK column.
    #          The actual enforcement happens in SQLite only when
    #          PRAGMA foreign_keys=ON — see database.py for the event listener.
    messages: list["Message"] = Relationship(
        back_populates="conversation",
        cascade_delete=True,
    )
