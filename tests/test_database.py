"""
Tests for the database layer: engine setup, model tables, and cascade deletes.

What concept it teaches:
  TDD for a SQLModel/SQLite persistence layer — write tests first, then make
  them green. Each test class targets a distinct concern (setup, conversation
  model, document model) so failures pinpoint the broken layer.

Why this approach over alternatives:
  Using an in-memory SQLite URL ("sqlite://") per test class keeps tests
  hermetic and fast. We never touch the real data/rag.db file here.

Where it fits in the RAG pipeline:
  This validates the DATABASE layer that underpins chat-history persistence.
  Document -> Chunks -> Embeddings -> Vector Store -> [DATABASE] <- Conversations / Messages
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from sqlmodel import Session, select, text

from src.database import create_db_and_tables, get_engine, get_session
from src.models.conversation import Conversation
from src.models.message import Message, MessageSource
from src.models.document import DocumentRecord


# ---------------------------------------------------------------------------
# Shared fixture: isolated in-memory engine for each test class
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """
    Provide a fresh in-memory SQLite engine with all tables created.

    WHY: Using "sqlite://" (two slashes = in-memory, no file) means every test
         run starts from a clean slate without touching data/rag.db.

    PATTERN: get_engine() is called here -- not create_engine() directly --
             so the PRAGMA foreign_keys=ON event listener is always attached.
             Without it, ON DELETE CASCADE silently does nothing in SQLite.
    """
    eng = get_engine("sqlite://")
    create_db_and_tables(eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# TestDatabaseSetup
# ---------------------------------------------------------------------------

class TestDatabaseSetup:
    """Verify the engine and table scaffolding work correctly."""

    def test_tables_created(self, engine):
        """
        All four tables exist after create_db_and_tables().

        WHY: SQLModel.metadata.create_all() only works if every model class
             has been imported before the call. This test would fail with a
             missing table if any model import was forgotten.
        """
        with Session(engine) as session:
            # SQLite stores table names in sqlite_master
            result = session.exec(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            ).all()
            table_names = {row[0] for row in result}

        expected = {"conversations", "messages", "message_sources", "documents"}
        assert expected.issubset(table_names), (
            f"Missing tables. Found: {table_names}. Expected at least: {expected}"
        )

    def test_foreign_keys_enabled(self, engine):
        """
        PRAGMA foreign_keys returns 1 (enabled) on every connection.

        WHY: SQLite disables foreign-key enforcement by default. The event
             listener in get_engine() must fire on every new connection --
             not just on the first one -- because SQLAlchemy pools connections.
        """
        with Session(engine) as session:
            result = session.exec(text("PRAGMA foreign_keys")).all()
            fk_value = result[0][0]

        assert fk_value == 1, (
            "Foreign keys are OFF. The PRAGMA foreign_keys=ON listener in "
            "get_engine() did not fire for this connection."
        )


# ---------------------------------------------------------------------------
# TestConversationModel
# ---------------------------------------------------------------------------

class TestConversationModel:
    """Verify Conversation CRUD and cascade behaviour."""

    def test_create_conversation(self, engine):
        """
        Creating a Conversation and committing it persists all default fields.

        WHY: Defaults like pinned=False and created_at are set by
             default_factory at the Python level, not by the DB. This test
             confirms they survive a full commit + re-fetch cycle.
        """
        with Session(engine) as session:
            conv = Conversation(title="Test Chat")
            session.add(conv)
            session.commit()
            session.refresh(conv)

            # Verify defaults
            assert conv.pinned is False, "pinned should default to False"
            assert conv.created_at is not None, "created_at must be set"
            assert conv.title == "Test Chat"
            conv_id = conv.id

        # Re-fetch in a new session to confirm persistence
        with Session(engine) as session:
            fetched = session.get(Conversation, conv_id)
            assert fetched is not None
            assert fetched.title == "Test Chat"
            assert fetched.pinned is False

    def test_cascade_delete_messages(self, engine):
        """
        Deleting a Conversation removes its Messages and their MessageSources.

        WHY: ON DELETE CASCADE only works when PRAGMA foreign_keys=ON.
             This test would silently pass (leaving orphans) without the
             PRAGMA listener -- which is exactly why test_foreign_keys_enabled
             must pass first.

        PATTERN: Create parent -> child -> grandchild, delete parent, verify
                 child and grandchild rows are gone.
        """
        with Session(engine) as session:
            conv = Conversation(title="Cascade Test")
            session.add(conv)
            session.commit()
            session.refresh(conv)
            conv_id = conv.id

            msg = Message(
                conversation_id=conv_id,
                role="user",
                content="Hello, world!",
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            msg_id = msg.id

            source = MessageSource(
                message_id=msg_id,
                doc_id="doc-001",
                chunk_id="chunk-001",
                filename="test.pdf",
                score=0.92,
                excerpt="Relevant excerpt text.",
            )
            session.add(source)
            session.commit()

        # Delete the conversation and verify cascade
        with Session(engine) as session:
            conv = session.get(Conversation, conv_id)
            session.delete(conv)
            session.commit()

        with Session(engine) as session:
            assert session.get(Message, msg_id) is None, (
                "Message was not deleted when its Conversation was deleted"
            )
            remaining_sources = session.exec(
                select(MessageSource).where(MessageSource.message_id == msg_id)
            ).all()
            assert len(remaining_sources) == 0, (
                "MessageSources were not deleted when their Message was deleted"
            )


# ---------------------------------------------------------------------------
# TestDocumentRecordModel
# ---------------------------------------------------------------------------

class TestDocumentRecordModel:
    """Verify DocumentRecord persistence."""

    def test_create_document_record(self, engine):
        """
        All fields on DocumentRecord are stored and retrieved correctly.

        WHY: DocumentRecord uses a content-hash id (not auto-generated uuid).
             This test confirms we can store and retrieve that custom primary key.
        """
        doc_id = "sha256-abc123"

        with Session(engine) as session:
            doc = DocumentRecord(
                id=doc_id,
                filename="report.pdf",
                file_type="pdf",
                file_size_bytes=204_800,
                chunks_count=42,
            )
            session.add(doc)
            session.commit()

        with Session(engine) as session:
            fetched = session.get(DocumentRecord, doc_id)
            assert fetched is not None
            assert fetched.filename == "report.pdf"
            assert fetched.file_type == "pdf"
            assert fetched.file_size_bytes == 204_800
            assert fetched.chunks_count == 42
            assert fetched.upload_date is not None


# ---------------------------------------------------------------------------
# TestGetSession
# ---------------------------------------------------------------------------

class TestGetSession:
    """Verify the get_session dependency-injection helper."""

    def test_get_session_yields_session(self, engine):
        """
        get_session is a generator that yields a usable Session.

        WHY: FastAPI dependency injection calls next() on the generator once
             to get the session, then calls it again at request end to run the
             finally block. This test mimics that lifecycle.
        """
        gen = get_session(engine)
        session = next(gen)
        assert isinstance(session, Session)
        # Exhaust the generator (runs the finally / cleanup block)
        try:
            next(gen)
        except StopIteration:
            pass  # Expected -- generator is exhausted after yielding once
