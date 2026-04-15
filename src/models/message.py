"""
Message and MessageSource SQLModel tables — one row per chat turn / citation.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                          |
                             [CONVERSATION] <-> [MESSAGE] <-> [MESSAGE_SOURCE]
                                                                    |
                                             Points back to a chunk in ChromaDB

What concept it teaches:
  A three-level parent/child/grandchild relationship in SQLModel.
  Conversation -> Message -> MessageSource forms a 1:N:N hierarchy.
  ON DELETE CASCADE flows down both levels automatically when foreign_keys
  are enabled in SQLite.

Why this approach over alternatives:
  Storing sources as a separate table (rather than a JSON column on Message)
  keeps the schema normalized and allows per-source queries (e.g. "which
  documents were cited most?") without deserializing blobs.

Where it fits in the RAG pipeline:
  Messages record the user/assistant dialogue. MessageSources record which
  document chunks the assistant cited — linking the answer back to the
  RETRIEVAL step.
"""

# NOTE: from __future__ import annotations is intentionally OMITTED.
# See conversation.py for the full explanation.

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from src.models.conversation import Conversation


class Message(SQLModel, table=True):
    """
    One turn in a Conversation — either a user question or an assistant reply.

    Attributes:
        id: UUID primary key.
        conversation_id: FK -> conversations.id with ON DELETE CASCADE.
        role: "user" or "assistant" — mirrors the OpenAI chat message role.
        content: Full message text.
        model: LLM model used to generate this message (null for user messages).
        created_at: UTC timestamp.
        token_count: Approximate token count for budget tracking (nullable).
        conversation: Back-reference to the parent Conversation.
        sources: List of MessageSource citations attached to this message.

    TRADE-OFF: role is a plain str rather than an Enum so that future roles
               ("system", "tool") don't require a migration. Validation is
               handled at the API layer (Pydantic model).
    """

    __tablename__ = "messages"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    # WHY: ondelete="CASCADE" generates the ON DELETE CASCADE DDL clause on this
    #      FK. index=True speeds up "give me all messages for conversation X"
    #      queries which the chat page fires on every load.
    conversation_id: str = Field(
        foreign_key="conversations.id",
        ondelete="CASCADE",
        index=True,
    )

    role: str  # "user" | "assistant"

    content: str = Field(default="")

    # WHY: Nullable — user messages don't have a model; only assistant messages do.
    model: Optional[str] = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # WHY: token_count is approximate and may be unavailable for some providers,
    #      so it's Optional rather than raising at insert time.
    token_count: Optional[int] = Field(default=None)

    # PATTERN: back_populates="messages" must match the attribute name on
    #          Conversation.messages. SQLModel uses these strings to wire
    #          the bidirectional relationship at class-registration time.
    conversation: Optional["Conversation"] = Relationship(
        back_populates="messages",
    )

    sources: list["MessageSource"] = Relationship(
        back_populates="message",
        cascade_delete=True,
    )


class MessageSource(SQLModel, table=True):
    """
    A single document chunk cited by an assistant Message.

    Attributes:
        id: Auto-increment integer PK (no UUID needed — sources have no
            independent identity outside their message).
        message_id: FK -> messages.id with ON DELETE CASCADE.
        doc_id: Content-hash document ID matching DocumentRecord.id.
        chunk_id: Chunk identifier within the document (ChromaDB chunk_id).
        filename: Human-readable filename for display in the UI.
        score: Cosine similarity score from the retriever (0.0–1.0).
        excerpt: Short text snippet for display without re-fetching ChromaDB.
        message: Back-reference to the parent Message.

    WHY: Storing excerpt here avoids a round-trip to ChromaDB when rendering
         source citations in the UI. The score lets the UI sort/filter sources
         by relevance.
    """

    __tablename__ = "message_sources"

    # WHY: Optional[int] + default=None tells SQLModel this is an auto-increment
    #      PK. SQLite assigns the value on INSERT, so Python starts with None.
    id: Optional[int] = Field(default=None, primary_key=True)

    message_id: str = Field(
        foreign_key="messages.id",
        ondelete="CASCADE",
        index=True,
    )

    doc_id: str
    chunk_id: str

    # WHY: filename is Optional — a source might reference a chunk whose document
    #      was deleted from DocumentRecord (soft-delete scenario).
    filename: Optional[str] = None

    score: float

    excerpt: str = Field(default="")

    message: Optional["Message"] = Relationship(back_populates="sources")
