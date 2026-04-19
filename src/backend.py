"""
Unified RAG backend — stateful facade that wires ChromaDB + SQLite persistence.

RAG Pipeline Position:
  INDEXING:  File -> DocumentLoader -> TextChunker -> ChromaDB (auto-embed + store)
                                                   -> SQLite (DocumentRecord metadata)

  QUERYING:  Question -> ChromaDB (auto-embed query, cosine search) -> top-K chunks
                      -> LLMHandler (build context, generate answer) -> Response

  CHAT:      Conversation -> Messages -> sliding window -> LLM (multi-turn context)
             All persisted in SQLite via SQLModel.

What concept it teaches:
  The Facade pattern — RAGBackend is a single entry point that coordinates four
  independent subsystems (document loading, vector storage, relational persistence,
  LLM generation) so that callers never need to know how they interact.

Why this approach over alternatives:
  The old backend created in-memory data structures that were lost on restart.
  This rewrite persists documents in ChromaDB (vectors) and SQLite (metadata +
  conversations), so data survives process restarts. The constructor accepts
  pre-built engine and collection objects (dependency injection) so tests can
  pass ephemeral/in-memory instances.

Where it fits in the RAG pipeline:
  This is the ORCHESTRATION layer. It does not implement any algorithm itself —
  it delegates to DocumentLoader (parsing), TextChunker (splitting),
  ChromaVectorStore (embedding + search), and LLMHandler (generation).
"""

import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, select

from .config import (
    DEFAULT_MODEL,
    MAX_TITLE_LENGTH,
    REASONING_MODEL,
    SLIDING_WINDOW_SIZE,
    TOP_K_RESULTS,
)
from .document_loader import DocumentLoader, TextChunker
from .llm_handler import LLMHandler
from .models.conversation import Conversation
from .models.document import DocumentRecord
from .models.message import Message, MessageSource
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGBackend:
    """Stateful RAG facade that persists data across requests and restarts.

    Coordinates two data stores:
      - ChromaDB: chunk text + embeddings (vector search)
      - SQLite: document metadata, conversations, messages, sources

    Lifecycle:
      1. ingest_file() / ingest_bytes() — parse, chunk, embed, persist
      2. query() / stream_query() — retrieve, generate, optionally persist chat
      3. delete_document() — remove from both stores
      4. Conversation CRUD — create, list, get, update, delete, search, export, share

    PATTERN: Dependency injection — the constructor takes a pre-built SQLAlchemy
    engine and ChromaDB collection so tests can supply ephemeral instances while
    production code supplies persistent ones.
    """

    def __init__(self, engine: Engine, collection: Any) -> None:
        """
        Args:
            engine: SQLAlchemy engine for SQLite persistence. Created via
                    database.get_engine() — use "sqlite://" for tests,
                    "sqlite:///path/to/file.db" for production.
            collection: A ChromaDB Collection instance. Use EphemeralClient for
                        tests, PersistentClient for production.
        """
        self.engine = engine

        # WHY: ChromaVectorStore is a thin wrapper that provides a clean API
        #      (upsert, query, delete_by_doc_id, get_stats) over the raw
        #      ChromaDB collection.
        self.vector_store = ChromaVectorStore(collection=collection)

        self.loader = DocumentLoader()

        # TRADE-OFF: Recursive chunking gives better retrieval quality than
        #            fixed-size because it respects paragraph/sentence boundaries.
        #            512-char chunks with 64-char overlap is a balanced default.
        self.chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=64,
            strategy="recursive",
        )

        # WHY: DEFAULT_MODEL from config ensures the UI and backend share the
        #      same fallback model without hard-coding the name.
        self.llm = LLMHandler(model=DEFAULT_MODEL)

        # PATTERN: Separate handler for the CoT reasoning pass. Cached once
        #          in the backend so we don't re-construct an LLMHandler on
        #          every query — the reasoning model is fixed by config, so
        #          caching is safe regardless of which answer model a user picks.
        # WHY max_tokens=2048: The reasoning pass is the "Step 1" of the UI's
        #      two-step visible flow (think → answer). Making it beefy enough
        #      to produce 6-10 sentences of genuine analysis (~400-700 tokens)
        #      gives users a clear thinking phase to watch, instead of a blink
        #      that ends before they notice. Headroom also protects against a
        #      future swap to a reasoning-style model whose hidden tokens
        #      would otherwise consume a tight budget.
        self.reasoning_llm = LLMHandler(model=REASONING_MODEL, max_tokens=2048)

        logger.info(
            "RAGBackend initialised (engine=%s, answer_model=%s, reasoning_model=%s)",
            engine.url, DEFAULT_MODEL, REASONING_MODEL,
        )

    # ------------------------------------------------------------------ #
    # Session helper                                                       #
    # ------------------------------------------------------------------ #

    def _session(self) -> Session:
        """Create a new SQLModel session bound to the backend's engine.

        Returns:
            A Session instance. Use as a context manager:
                with self._session() as session:
                    session.add(obj)
                    session.commit()

        WHY: Each operation gets its own short-lived session rather than
        sharing one across the backend's lifetime. This prevents stale reads,
        thread-safety issues, and keeps transaction scope tight.
        """
        return Session(self.engine)

    # ------------------------------------------------------------------ #
    # Document ingestion                                                   #
    # ------------------------------------------------------------------ #

    def ingest_file(
        self,
        file_path: str | Path,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """Load a file from disk, chunk it, and persist to both stores.

        Cross-store order: ChromaDB first, then SQLite.

        WHY ChromaDB first: If ChromaDB fails, SQLite is untouched and we can
        retry cleanly. The reverse (SQLite first, ChromaDB fails) leaves a
        phantom metadata record with no backing vectors — harder to detect
        and recover from.

        Args:
            file_path: Path to the file to ingest.
            original_filename: Override the filename stored in metadata.
                              Useful when the file comes from a temp directory.

        Returns:
            Dict with doc_id, filename, chunks_count, status.
        """
        path = Path(file_path)
        document = self.loader.load(path)
        if original_filename:
            document.metadata["filename"] = original_filename

        chunks = self.chunker.chunk(document)

        if not chunks:
            logger.warning("No chunks produced from '%s'", path.name)
            return {
                "doc_id": document.doc_id,
                "filename": document.metadata.get("filename", path.name),
                "chunks_count": 0,
                "status": "empty",
            }

        # STEP 1: Upsert chunks into ChromaDB (auto-embeds via collection's
        #         default embedding function — no explicit embeddings needed).
        self.vector_store.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.metadata.get("filename", ""),
                    "chunk_index": c.metadata.get("chunk_index", 0),
                }
                for c in chunks
            ],
            # WHY: No embeddings= argument. ChromaDB auto-embeds using the
            #      collection's embedding function (all-MiniLM-L6-v2 by default).
            #      This eliminates the need for a separate TF-IDF or neural
            #      embedder in the backend.
        )

        # STEP 2: Save document metadata to SQLite.
        filename = document.metadata.get("filename", path.name)
        file_type = document.metadata.get("file_type", path.suffix.lstrip("."))
        file_size = document.metadata.get("file_size_bytes", 0)

        record = DocumentRecord(
            id=document.doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size,
            chunks_count=len(chunks),
        )

        # WHY session.merge: DocumentRecord.id is a content-hash. If the same
        #      file is uploaded twice, the hash is identical. merge() does an
        #      upsert (INSERT or UPDATE) based on primary key, making
        #      re-ingestion idempotent rather than raising a PK conflict.
        with self._session() as session:
            session.merge(record)
            session.commit()

        logger.info(
            "Ingested '%s' -> doc_id=%s (%d chunks)",
            filename, document.doc_id, len(chunks),
        )
        return {
            "doc_id": document.doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "status": "success",
        }

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        """Ingest raw file bytes (e.g. from an upload endpoint).

        Writes data to a temp file so DocumentLoader can detect the format
        from the file extension, then delegates to ingest_file().

        Args:
            filename: Original filename (used for extension detection + metadata).
            data: Raw file contents.

        Returns:
            Same dict as ingest_file().
        """
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return self.ingest_file(tmp_path, original_filename=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Querying                                                             #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run a full RAG query: retrieve chunks -> build context -> generate answer.

        Args:
            question: Natural language question from the user.
            top_k: Number of chunks to retrieve (default from config).
            model: LLM model override (creates a new handler if different).

        Returns:
            Dict with answer (str), sources (list[dict]), confidence (float).
        """
        k = top_k or TOP_K_RESULTS

        # WHY query_text: ChromaDB auto-embeds the question using the same
        #      embedding function that was used to embed the chunks. This
        #      ensures query and document embeddings live in the same space.
        results = self.vector_store.query(query_text=question, top_k=k)

        if not results:
            return {
                "answer": "No documents indexed yet. Please upload documents first.",
                "sources": [],
                "confidence": 0.0,
            }

        # Build context string from retrieved chunks
        context = "\n\n".join(
            f"[{r.metadata.get('filename', 'unknown')}] {r.content}"
            for r in results
        )

        # Generate answer (create a per-query handler if model differs)
        handler = self.llm
        if model and model != self.llm.model:
            handler = LLMHandler(model=model)

        answer = handler.generate_with_context(question, context)

        sources = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "filename": r.metadata.get("filename"),
                "score": round(r.score, 4),
                "excerpt": r.content[:300],
                "chunk_index": r.metadata.get("chunk_index"),
            }
            for r in results
        ]

        # PATTERN: Confidence = clamped average of top-3 similarity scores.
        #          This gives a rough signal of retrieval quality without
        #          requiring a separate calibration model.
        top_scores = [r.score for r in results[: min(3, len(results))]]
        confidence = max(0.0, min(1.0, sum(top_scores) / len(top_scores)))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 4),
        }

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
        conversation_id: str | None = None,
    ):
        """Retrieve context and stream reasoning + answer with chain-of-thought events.

        Event stream shape (in order):
          ("status", str)       — programmatic retrieval milestones
          ("reasoning", str)    — LLM chain-of-thought tokens (brief pre-answer pass)
          ("token", str)        — final answer tokens
          ("done", dict)        — sources + persistence metadata

        WHY two LLM calls: The reasoning pass uses a focused prompt that asks
        the model to think out loud about the retrieved context BEFORE giving
        an answer. This makes the agent's reasoning visible to users (like
        ChatGPT's "thinking" mode) at the cost of a short extra call.

        WHY only the answer is persisted: Reasoning is ephemeral scaffolding
        for UX, not a durable artifact of the conversation. Persisting it
        would double storage and confuse the sliding-window context.

        Yields:
            Tuples as described above.
        """
        k = top_k or TOP_K_RESULTS

        # ---- PHASE 0: Retrieval ------------------------------------------------
        yield ("status", "Searching indexed documents...")
        results = self.vector_store.query(query_text=question, top_k=k)

        if not results:
            yield ("status", "No indexed documents — nothing to retrieve.")
            yield ("token", "No documents indexed yet. Please upload documents first.")
            yield ("done", {"sources": []})
            return

        # WHY: Summarise retrieval in one status line so the user can see which
        #      files contributed without inspecting the sources panel yet.
        filenames = [r.metadata.get("filename", "unknown") for r in results]
        unique_files = sorted({f for f in filenames if f})
        file_summary = ", ".join(unique_files[:3])
        if len(unique_files) > 3:
            file_summary += f" (+{len(unique_files) - 3} more)"
        yield (
            "status",
            f"Retrieved {len(results)} chunk(s) across {len(unique_files)} file(s): {file_summary}",
        )

        # Build context from retrieved chunks (shared by reasoning + answer)
        context = "\n\n".join(
            f"[{r.metadata.get('filename', 'unknown')}] {r.content}"
            for r in results
        )

        handler = self.llm
        if model and model != self.llm.model:
            handler = LLMHandler(model=model)

        sources = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "filename": r.metadata.get("filename"),
                "score": round(r.score, 4),
                "excerpt": r.content[:300],
            }
            for r in results
        ]

        # ---- PHASE 1: Reasoning pass (chain-of-thought) ------------------------
        # WHY a dedicated reasoning prompt: Asking the model to "think first,
        # answer later" in a single call is brittle — formatting drifts between
        # providers. A separate short call with a focused system prompt gives
        # deterministic reasoning tokens we can stream as their own event type.
        #
        # WHY a dedicated reasoning model: The CoT output is short, throwaway
        # scaffolding. Running it through the user's (potentially premium)
        # answer model doubles cost for no quality gain. self.reasoning_llm is
        # cached to REASONING_MODEL (default: gpt-5-nano) independent of the
        # answer model — so premium answers stay cheap to "think" about.
        yield ("status", f"Analyzing retrieved context ({self.reasoning_llm.model})...")

        # WHY a detailed prompt: This is Step 1 of the UI's two-step visible
        #      flow. A rich thinking stream makes the reasoning phase feel
        #      substantive — the user watches the model deliberate before
        #      Step 2 (the actual answer) begins. Prompt is tuned to produce
        #      6-10 sentences of plain-text analysis at a relaxed pace.
        reasoning_system = (
            "You are a reasoning assistant for a retrieval-augmented Q&A system. "
            "Given the user's question and the retrieved document excerpts, think "
            "out loud in 6-10 plain sentences about how you will construct the "
            "answer. Walk through, in order:\n"
            "1) What the user is really asking and any ambiguity to resolve.\n"
            "2) Which retrieved excerpts look most relevant and why.\n"
            "3) Any gaps, conflicts, or caveats in the retrieved context.\n"
            "4) The outline or structure of the answer you will give next.\n"
            "Write in the first person ('I'll start by...', 'I notice that...'). "
            "No bullet lists. No markdown headings. No preamble. Do NOT give the "
            "final answer — a later step handles that. Keep a natural, deliberate "
            "pace — the user is watching this stream in real time."
        )
        reasoning_user = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Reasoning (think out loud, do not answer):"
        )

        try:
            for token in self.reasoning_llm.stream_response(
                reasoning_user, system_prompt=reasoning_system
            ):
                yield ("reasoning", token)
        except Exception as exc:
            # PATTERN: Reasoning is best-effort — a failure here must not block
            # the final answer. Log, emit a terse status, continue to the answer.
            logger.warning("Reasoning pass failed: %s", exc)
            yield ("status", "Reasoning unavailable — skipping to answer.")

        # ---- PHASE 2: Answer pass ----------------------------------------------
        yield ("status", "Composing answer...")

        system_prompt = (
            "You are a helpful assistant. Answer the user's question based solely on the "
            "provided context. If the context does not contain enough information, say so.\n\n"
            "Format your response using Markdown for readability:\n"
            "- Use ## for main sections and ### for sub-sections (max 3 levels)\n"
            "- Use **bold** for key terms and important concepts\n"
            "- Use bullet points (-) for lists of related items\n"
            "- Use numbered lists (1.) for sequential steps\n"
            "- Use `inline code` for technical terms, parameters, or commands\n"
            "- Use fenced code blocks (```language) for code snippets\n"
            "- Use > blockquotes for notable quotes from the context\n"
            "- Keep paragraphs short (2-3 sentences max)\n"
            "- Add blank lines between sections for visual breathing room\n"
            "Do NOT use # (h1) headings. Start directly with content or ## sections."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Accumulate full response for persistence
        full_response = []

        if conversation_id:
            # PHASE 1: Save user message BEFORE streaming
            self._save_message(conversation_id, "user", question)

            # PHASE 2: Load sliding window (completed pairs only — excludes
            #          the just-saved user message because it has no assistant
            #          reply yet).
            window = self._get_sliding_window(conversation_id)

            # PHASE 3: Build messages list for multi-turn generation
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(window)
            messages.append({"role": "user", "content": user_prompt})

            # PHASE 4: Stream via messages API (multi-turn aware)
            for token in handler.stream_messages(messages):
                full_response.append(token)
                yield ("token", token)

            # PHASE 5: Save assistant message + sources
            assistant_content = "".join(full_response)
            assistant_msg_id = self._save_message(
                conversation_id, "assistant", assistant_content,
                model=handler.model, sources=sources,
            )

            # PHASE 6: Auto-title on first message
            self._auto_title(conversation_id, question)

            # WHY: Include message_id and conversation_id in the done event
            #      so the frontend can update its local state (add the new
            #      message to the conversation without re-fetching).
            yield ("done", {
                "sources": sources,
                "message_id": assistant_msg_id,
                "conversation_id": conversation_id,
            })

        else:
            # No conversation — simple single-turn streaming
            for token in handler.stream_response(user_prompt, system_prompt=system_prompt):
                full_response.append(token)
                yield ("token", token)

            yield ("done", {"sources": sources})

    # ------------------------------------------------------------------ #
    # Document management                                                  #
    # ------------------------------------------------------------------ #

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from both ChromaDB and SQLite.

        Cross-store order: ChromaDB first, then SQLite.

        WHY: Same reasoning as ingest — ChromaDB failure leaves SQLite clean.
        If ChromaDB succeeds but SQLite fails, we have orphan-free vectors
        (the SQLite record still references them, so a retry can clean up).

        Args:
            doc_id: The document's content-hash ID.

        Returns:
            True if the document was found and deleted, False otherwise.
        """
        # STEP 1: Delete chunks from ChromaDB
        self.vector_store.delete_by_doc_id(doc_id)

        # STEP 2: Delete metadata from SQLite
        with self._session() as session:
            record = session.get(DocumentRecord, doc_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()

        logger.info("Deleted document doc_id=%s", doc_id)
        return True

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for all ingested documents.

        Returns:
            List of dicts with id, filename, file_type, file_size_bytes,
            chunks_count, upload_date — sorted by upload_date descending.
        """
        with self._session() as session:
            records = session.exec(
                select(DocumentRecord).order_by(DocumentRecord.upload_date.desc())
            ).all()
            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "file_type": r.file_type,
                    "file_size_bytes": r.file_size_bytes,
                    "chunks_count": r.chunks_count,
                    "upload_date": r.upload_date.isoformat(),
                }
                for r in records
            ]

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """Return all chunks for a specific document from ChromaDB.

        Args:
            doc_id: The document's content-hash ID.

        Returns:
            List of chunk dicts with chunk_id, content, and metadata.
        """
        # WHY: ChromaDB's get() with where filter retrieves all chunks for a
        #      document without needing to know individual chunk IDs.
        raw = self.vector_store._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, chunk_id in enumerate(raw["ids"]):
            chunks.append({
                "chunk_id": chunk_id,
                "content": raw["documents"][i] if raw["documents"] else "",
                "metadata": raw["metadatas"][i] if raw["metadatas"] else {},
            })
        return chunks

    def get_stats(self) -> dict[str, Any]:
        """Return combined statistics from both stores.

        Returns:
            Dict with total_docs, total_chunks, backend, collection.
        """
        store_stats = self.vector_store.get_stats()
        with self._session() as session:
            doc_count = len(session.exec(select(DocumentRecord)).all())

        return {
            "total_docs": doc_count,
            **store_stats,
        }

    # ------------------------------------------------------------------ #
    # Conversation CRUD                                                    #
    # ------------------------------------------------------------------ #

    def create_conversation(self, title: str = "New Chat") -> dict[str, Any]:
        """Create a new conversation in SQLite.

        Args:
            title: Human-readable conversation title.

        Returns:
            Dict with id, title, created_at.
        """
        conv = Conversation(title=title)
        with self._session() as session:
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return {
                "id": conv.id,
                "title": conv.title,
                "pinned": conv.pinned,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return all conversations, pinned first, then by updated_at descending.

        WHY pinned first: Users pin important conversations so they stay at the
        top of the sidebar regardless of when they were last updated.

        Returns:
            List of conversation summary dicts.
        """
        with self._session() as session:
            convs = session.exec(
                select(Conversation)
                .order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
            ).all()
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "pinned": c.pinned,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
                for c in convs
            ]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Return a conversation with its messages and their sources.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Dict with id, title, messages (each with sources), or None if not found.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None

            # WHY: Eagerly load messages ordered by creation time so the
            #      frontend can render them in chronological order.
            messages = session.exec(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()

            msg_dicts = []
            for msg in messages:
                # Load sources for each message
                sources = session.exec(
                    select(MessageSource)
                    .where(MessageSource.message_id == msg.id)
                ).all()

                msg_dicts.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "model": msg.model,
                    "created_at": msg.created_at.isoformat(),
                    "sources": [
                        {
                            "doc_id": s.doc_id,
                            "chunk_id": s.chunk_id,
                            "filename": s.filename,
                            "score": s.score,
                            "excerpt": s.excerpt,
                        }
                        for s in sources
                    ],
                })

            return {
                "id": conv.id,
                "title": conv.title,
                "pinned": conv.pinned,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "messages": msg_dicts,
            }

    def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any] | None:
        """Update a conversation's title and/or pinned status.

        Args:
            conversation_id: UUID of the conversation.
            title: New title (if provided).
            pinned: New pinned status (if provided).

        Returns:
            Updated conversation dict, or None if not found.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None

            if title is not None:
                conv.title = title
            if pinned is not None:
                conv.pinned = pinned

            conv.updated_at = datetime.now(timezone.utc)
            session.add(conv)
            session.commit()
            session.refresh(conv)

            return {
                "id": conv.id,
                "title": conv.title,
                "pinned": conv.pinned,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages and sources (cascade).

        WHY cascade: ON DELETE CASCADE in the FK definitions means deleting
        the Conversation row automatically removes all child Messages and
        grandchild MessageSources. The PRAGMA foreign_keys=ON listener in
        database.py ensures this works in SQLite.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            True if deleted, False if not found.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return False
            session.delete(conv)
            session.commit()
        return True

    def search_conversations(self, query: str) -> list[dict[str, Any]]:
        """Search conversations by title or message content.

        Uses SQL LIKE for simple substring matching. For a portfolio project
        this is adequate; production would use full-text search (FTS5).

        Args:
            query: Search string.

        Returns:
            List of matching conversation summary dicts.
        """
        with self._session() as session:
            # WHY: Two separate queries then union the IDs. This avoids a
            #      complex JOIN that could return duplicate rows.
            matching_by_title = session.exec(
                select(Conversation.id).where(Conversation.title.contains(query))
            ).all()

            matching_by_message = session.exec(
                select(Message.conversation_id)
                .where(Message.content.contains(query))
            ).all()

            # Combine and deduplicate
            matching_ids = set(matching_by_title) | set(matching_by_message)

            if not matching_ids:
                return []

            convs = session.exec(
                select(Conversation)
                .where(Conversation.id.in_(matching_ids))
                .order_by(Conversation.updated_at.desc())
            ).all()

            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "pinned": c.pinned,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
                for c in convs
            ]

    def export_conversation(self, conversation_id: str) -> str | None:
        """Export a conversation as a Markdown string.

        Format:
          # {title}
          ---
          **User:** {message}
          **Assistant:** {message}

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Markdown string, or None if conversation not found.
        """
        data = self.get_conversation(conversation_id)
        if data is None:
            return None

        lines = [f"# {data['title']}", "---", ""]
        for msg in data["messages"]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"**{role_label}:** {msg['content']}")
            lines.append("")

        return "\n".join(lines)

    def create_share_token(self, conversation_id: str) -> str | None:
        """Generate a share token for read-only public access.

        WHY UUID4: Opaque, unguessable tokens. Anyone with the token can
        view the conversation, so it must not be sequential or predictable.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            UUID4 token string, or None if conversation not found.
        """
        token = str(uuid.uuid4())
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None
            conv.share_token = token
            session.add(conv)
            session.commit()
        return token

    def get_shared_conversation(self, token: str) -> dict[str, Any] | None:
        """Retrieve a conversation by its share token.

        Args:
            token: The share token string.

        Returns:
            Conversation dict with messages, or None if token is invalid.
        """
        with self._session() as session:
            conv = session.exec(
                select(Conversation).where(Conversation.share_token == token)
            ).first()
            if conv is None:
                return None
            # WHY: Capture the ID inside the session scope to avoid detached
            #      instance errors when get_conversation opens a new session.
            conv_id = conv.id

        # WHY: Reuse get_conversation to build the full response dict with
        #      messages and sources, avoiding code duplication.
        return self.get_conversation(conv_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist a message (and optional sources) to SQLite.

        Also updates the parent conversation's updated_at timestamp so that
        list_conversations() sorts by most-recently-active.

        Args:
            conversation_id: UUID of the parent conversation.
            role: "user" or "assistant".
            content: Message text.
            model: LLM model name (only for assistant messages).
            sources: List of source dicts from retrieval (only for assistant).

        Returns:
            The UUID string of the newly created Message.
        """
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
        )

        # WHY: Capture the ID before entering the session scope. Message.id is
        #      set by default_factory at construction time (uuid4), so it's
        #      available immediately. After session.commit(), SQLAlchemy expires
        #      all attributes — accessing msg.id outside the session would
        #      trigger a DetachedInstanceError.
        msg_id = msg.id

        with self._session() as session:
            session.add(msg)

            # WHY: Save sources as separate MessageSource rows rather than
            #      embedding them in a JSON column. This keeps the schema
            #      normalized and enables per-source queries.
            if sources:
                for src in sources:
                    source = MessageSource(
                        message_id=msg_id,
                        doc_id=src.get("doc_id", ""),
                        chunk_id=src.get("chunk_id", ""),
                        filename=src.get("filename"),
                        score=src.get("score", 0.0),
                        excerpt=src.get("excerpt", ""),
                    )
                    session.add(source)

            # Update conversation's updated_at timestamp
            conv = session.get(Conversation, conversation_id)
            if conv:
                conv.updated_at = datetime.now(timezone.utc)
                session.add(conv)

            session.commit()

        return msg_id

    def _get_sliding_window(
        self,
        conversation_id: str,
        max_pairs: int = SLIDING_WINDOW_SIZE,
    ) -> list[dict[str, str]]:
        """Return the last N completed exchange pairs for LLM context.

        CRITICAL: Only return COMPLETED pairs — a user message followed by an
        assistant message. A user message with no assistant reply is NOT a
        complete pair and must be excluded.

        WHY exclude unpaired: The current user question is saved BEFORE streaming
        starts (phase 1 of stream_query). If we included it in the window, the
        LLM would see the question twice — once in the window and once as the
        explicit user turn appended by stream_query. This causes the model to
        repeat itself or get confused.

        Args:
            conversation_id: UUID of the conversation.
            max_pairs: Maximum number of user/assistant pairs to return.

        Returns:
            List of {"role": str, "content": str} dicts representing the
            last max_pairs completed exchanges, in chronological order.
        """
        with self._session() as session:
            messages = session.exec(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()

            # Build completed pairs only (inside session scope to prevent
            # DetachedInstanceError if a future change adds a commit above).
            # WHY: We walk the message list looking for consecutive user->assistant
            #      pairs. Any other pattern (user->user, assistant->assistant,
            #      standalone messages) is skipped.
            pairs: list[Message] = []
            i = 0
            while i < len(messages) - 1:
                if messages[i].role == "user" and messages[i + 1].role == "assistant":
                    pairs.append(messages[i])
                    pairs.append(messages[i + 1])
                    i += 2
                else:
                    i += 1

            # Take the last max_pairs * 2 messages (each pair = 2 messages)
            window = pairs[-(max_pairs * 2):]
            return [{"role": m.role, "content": m.content} for m in window]

    def _auto_title(
        self,
        conversation_id: str,
        first_query: str,
    ) -> None:
        """Set the conversation title from the first user query if still "New Chat".

        Truncates at a word boundary to avoid cutting mid-word, with a maximum
        length of MAX_TITLE_LENGTH characters from config.

        Args:
            conversation_id: UUID of the conversation.
            first_query: The user's first question text.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None or conv.title != "New Chat":
                return

            # Truncate at word boundary
            title = first_query.strip()
            if len(title) > MAX_TITLE_LENGTH:
                # Find the last space before the limit
                truncated = title[:MAX_TITLE_LENGTH]
                last_space = truncated.rfind(" ")
                if last_space > 0:
                    title = truncated[:last_space] + "..."
                else:
                    # Single long word — hard truncate
                    title = truncated + "..."

            conv.title = title
            conv.updated_at = datetime.now(timezone.utc)
            session.add(conv)
            session.commit()
