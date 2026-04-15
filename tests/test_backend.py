"""
Tests for RAGBackend — the stateful facade that wires ChromaDB + SQLite persistence.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> [VECTOR STORE] -> Retrieval -> [GENERATOR] -> Answer
                                           |                              |
                                      ChromaDB                       LLMHandler
                                           |                              |
                                      [BACKEND] <-> SQLite (Conversations, Messages, Documents)

What concept it teaches:
  Testing a facade that coordinates two data stores (ChromaDB for vectors,
  SQLite for relational data) requires careful fixture design:
    1. EphemeralClient for ChromaDB — no disk I/O, isolated per test.
    2. In-memory SQLite via get_engine("sqlite://") — no database files.
    3. Both stores are torn down automatically when fixtures go out of scope.

WHY TDD:
  Writing tests first defines the public contract of RAGBackend from the
  caller's perspective.  The implementation is free to change internals
  (chunking strategy, embedding model) as long as the contract holds.

PATTERN: Each test class groups related behaviour (ingestion, conversation
  CRUD, sliding window) so failures point to the broken subsystem immediately.
"""

import uuid
from pathlib import Path

import chromadb
import pytest
from sqlmodel import Session, select

from src.backend import RAGBackend
from src.database import create_db_and_tables, get_engine
from src.models.conversation import Conversation
from src.models.document import DocumentRecord
from src.models.message import Message, MessageSource


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tmp_sqlite_engine():
    """
    In-memory SQLite engine with all tables created.

    WHY in-memory: Each test gets a fresh database that vanishes when the
    fixture goes out of scope.  No teardown needed, no file cleanup, no
    shared state between tests.
    """
    engine = get_engine("sqlite://")
    create_db_and_tables(engine)
    return engine


@pytest.fixture
def chroma_backend_collection():
    """
    Ephemeral ChromaDB collection with auto-embedding enabled.

    WHY no embedding_function=None: Unlike the vector_store tests (which
    supply explicit embeddings), the backend relies on ChromaDB's built-in
    embedding function to auto-embed chunks at upsert time and queries at
    query time.  Omitting embedding_function lets the default all-MiniLM-L6-v2
    model handle both operations transparently.

    WHY unique name: ChromaDB's EphemeralClient shares an in-process store.
    A UUID suffix ensures complete isolation between test runs.
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=f"test_backend_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def backend(tmp_sqlite_engine, chroma_backend_collection):
    """
    A fully-wired RAGBackend with ephemeral ChromaDB + in-memory SQLite.

    This is the system-under-test for every backend test.  Both stores
    are isolated and disposable.
    """
    return RAGBackend(engine=tmp_sqlite_engine, collection=chroma_backend_collection)


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    """
    A temporary .txt file with enough content to produce multiple chunks.

    WHY 3+ sentences: The TextChunker's MIN_CHUNK_LENGTH (20 chars) filters
    out very short chunks.  Three substantial sentences ensure at least one
    chunk survives filtering after the recursive splitter runs.
    """
    content = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances "
        "large language models by retrieving relevant documents from an external "
        "knowledge base before generating a response. "
        "This allows the model to access up-to-date information and produce "
        "more accurate, grounded answers. "
        "The retrieval step typically uses dense vector search, where both the "
        "query and documents are embedded into a shared vector space."
    )
    file = tmp_path / "rag_overview.txt"
    file.write_text(content, encoding="utf-8")
    return file


# --------------------------------------------------------------------------- #
# Document operation tests                                                     #
# --------------------------------------------------------------------------- #

class TestDocumentOperations:
    """Tests for ingest, list, query, delete, and idempotent re-ingest."""

    def test_ingest_file(self, backend: RAGBackend, txt_file: Path):
        """ingest_file returns success with a positive chunk count."""
        result = backend.ingest_file(txt_file)

        assert result["status"] == "success"
        assert result["chunks_count"] > 0
        assert "doc_id" in result
        assert "filename" in result

    def test_list_documents_after_ingest(self, backend: RAGBackend, txt_file: Path):
        """After ingestion, list_documents returns the ingested document."""
        result = backend.ingest_file(txt_file)
        docs = backend.list_documents()

        assert len(docs) == 1
        assert docs[0]["id"] == result["doc_id"]
        assert docs[0]["filename"] == txt_file.name

    def test_query_after_ingest(self, backend: RAGBackend, txt_file: Path):
        """After ingestion, querying returns an answer and sources."""
        backend.ingest_file(txt_file)
        result = backend.query("What is RAG?")

        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_delete_document(self, backend: RAGBackend, txt_file: Path):
        """Deleting a document removes it from both stores."""
        ingest_result = backend.ingest_file(txt_file)
        doc_id = ingest_result["doc_id"]

        deleted = backend.delete_document(doc_id)

        assert deleted is True
        assert backend.list_documents() == []

    def test_reingest_same_file_is_idempotent(
        self, backend: RAGBackend, txt_file: Path
    ):
        """Ingesting the same file twice results in only 1 document in list_documents."""
        backend.ingest_file(txt_file)
        backend.ingest_file(txt_file)

        docs = backend.list_documents()
        assert len(docs) == 1


# --------------------------------------------------------------------------- #
# Conversation CRUD tests                                                      #
# --------------------------------------------------------------------------- #

class TestConversationCRUD:
    """Tests for conversation create, list, get, update, delete, search, export, share."""

    def test_create_conversation(self, backend: RAGBackend):
        """create_conversation returns a dict with id and title."""
        conv = backend.create_conversation(title="Test Chat")

        assert "id" in conv
        assert conv["title"] == "Test Chat"

    def test_list_conversations(self, backend: RAGBackend):
        """list_conversations returns all created conversations."""
        backend.create_conversation(title="Chat 1")
        backend.create_conversation(title="Chat 2")

        convs = backend.list_conversations()

        assert len(convs) == 2
        titles = {c["title"] for c in convs}
        assert titles == {"Chat 1", "Chat 2"}

    def test_get_conversation_with_messages(self, backend: RAGBackend):
        """After _save_message, get_conversation returns the messages."""
        conv = backend.create_conversation(title="Msg Test")
        backend._save_message(conv["id"], "user", "Hello!")
        backend._save_message(conv["id"], "assistant", "Hi there!")

        result = backend.get_conversation(conv["id"])

        assert result is not None
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "Hi there!"

    def test_update_conversation(self, backend: RAGBackend):
        """update_conversation can rename and pin a conversation."""
        conv = backend.create_conversation(title="Old Title")

        updated = backend.update_conversation(
            conv["id"], title="New Title", pinned=True
        )

        assert updated is not None
        assert updated["title"] == "New Title"
        assert updated["pinned"] is True

    def test_delete_conversation_cascades(self, backend: RAGBackend):
        """Deleting a conversation removes its messages and sources."""
        conv = backend.create_conversation()
        msg_id = backend._save_message(
            conv["id"], "assistant", "Answer",
            sources=[{
                "doc_id": "d1",
                "chunk_id": "c1",
                "filename": "f.txt",
                "score": 0.9,
                "excerpt": "some text",
            }],
        )

        deleted = backend.delete_conversation(conv["id"])
        assert deleted is True

        # Verify messages and sources are gone from the database
        with Session(backend.engine) as session:
            messages = session.exec(select(Message)).all()
            sources = session.exec(select(MessageSource)).all()
            assert len(messages) == 0
            assert len(sources) == 0

    def test_search_conversations(self, backend: RAGBackend):
        """search_conversations finds by title and message content."""
        conv = backend.create_conversation(title="Machine Learning Chat")
        backend._save_message(conv["id"], "user", "Tell me about neural networks")

        # Search by title
        results_title = backend.search_conversations("Machine Learning")
        assert len(results_title) >= 1

        # Search by message content
        results_msg = backend.search_conversations("neural networks")
        assert len(results_msg) >= 1

    def test_export_conversation(self, backend: RAGBackend):
        """export_conversation returns a markdown string with messages."""
        conv = backend.create_conversation(title="Export Test")
        backend._save_message(conv["id"], "user", "What is RAG?")
        backend._save_message(conv["id"], "assistant", "RAG is a technique.")

        md = backend.export_conversation(conv["id"])

        assert md is not None
        assert "Export Test" in md
        assert "What is RAG?" in md
        assert "RAG is a technique." in md

    def test_share_token(self, backend: RAGBackend):
        """create_share_token returns a token; get_shared_conversation returns data."""
        conv = backend.create_conversation(title="Shared Chat")
        backend._save_message(conv["id"], "user", "Hello")
        backend._save_message(conv["id"], "assistant", "Hi")

        token = backend.create_share_token(conv["id"])
        assert token is not None

        shared = backend.get_shared_conversation(token)
        assert shared is not None
        assert shared["title"] == "Shared Chat"
        assert len(shared["messages"]) == 2


# --------------------------------------------------------------------------- #
# Sliding window tests                                                         #
# --------------------------------------------------------------------------- #

class TestSlidingWindow:
    """Tests for _get_sliding_window — the chat-history truncation logic."""

    def test_sliding_window_empty(self, backend: RAGBackend):
        """A new conversation with no messages returns an empty window."""
        conv = backend.create_conversation()
        window = backend._get_sliding_window(conv["id"])
        assert window == []

    def test_sliding_window_excludes_unpaired(self, backend: RAGBackend):
        """
        A user message with no assistant reply is NOT a completed pair.

        WHY: The current user question is saved BEFORE streaming starts.
        If the window included unpaired user messages, the LLM prompt would
        contain the current question twice — once in the window and once
        as the explicit user turn.
        """
        conv = backend.create_conversation()
        backend._save_message(conv["id"], "user", "Hello?")

        window = backend._get_sliding_window(conv["id"])
        assert window == []

    def test_sliding_window_respects_limit(self, backend: RAGBackend):
        """
        With 5 completed pairs and max_pairs=2, only the last 4 messages
        (2 pairs) are returned.
        """
        conv = backend.create_conversation()

        # Create 5 completed pairs (10 messages total)
        for i in range(5):
            backend._save_message(conv["id"], "user", f"Question {i}")
            backend._save_message(conv["id"], "assistant", f"Answer {i}")

        window = backend._get_sliding_window(conv["id"], max_pairs=2)

        assert len(window) == 4
        assert window[0]["role"] == "user"
        assert window[0]["content"] == "Question 3"
        assert window[1]["role"] == "assistant"
        assert window[1]["content"] == "Answer 3"
        assert window[2]["role"] == "user"
        assert window[2]["content"] == "Question 4"
        assert window[3]["role"] == "assistant"
        assert window[3]["content"] == "Answer 4"
