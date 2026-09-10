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

import hashlib
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from .api.schemas.telemetry import StageTelemetry
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MODEL,
    EVAL_MODEL,
    REASONING_MODEL,
    RERANK_OVER_FETCH_N,
    RETRIEVER_STRATEGY,
    SLIDING_WINDOW_SIZE,
    TOP_K_RESULTS,
)
from .conversations import ConversationHistory, ConversationStore
from .domain import SearchResult
from .evaluation import MessageEvaluator
from .ingestion import DocumentLoader, TextChunker
from .llm_handler import LLMHandler
from .models.document import DocumentRecord
from .query_engine import QueryEngine, StreamResult
from .retrieval import build_retrieval_plan
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def _source_dict(result: SearchResult) -> dict[str, Any]:
    """Shape one retrieved chunk into the source-citation dict the API returns.

    This is the single definition of a source citation. Both query paths use it,
    so a citation carries the same fields whichever endpoint produced it.

    Args:
        result: One chunk returned by the Retriever.

    Returns:
        The citation dict the API serialises and the frontend renders.

    BEFORE: the synchronous path spread this dict and added ``chunk_index``;
            the streaming path used the dict as-is, so the two endpoints
            returned different field sets for the same concept.
    AFTER:  ``chunk_index`` lives here, so the shapes cannot drift again.
    WHY:    one component renders citations from both paths.
    """
    return {
        "doc_id": result.doc_id,
        "chunk_id": result.chunk_id,
        "filename": result.metadata.get("filename"),
        "score": round(result.score, 4),
        "excerpt": result.content[:300],
        "chunk_index": result.metadata.get("chunk_index"),
    }


# One event streamed out of RAGBackend.stream_query: a label plus either a
# display string (status/reasoning/token) or a payload dict (done/telemetry).
# WHY named here and not reused from query_engine: the engine's terminal event
#      carries a StreamResult, which this facade consumes rather than forwards.
#      The two shapes are deliberately different, so they get different names.
BackendStreamEvent = tuple[str, "str | dict"]


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
        # SINGLE SOURCE: chunk size/overlap come from config (CHUNK_SIZE/
        #            CHUNK_OVERLAP) so the eval harness benchmarks production's
        #            actual chunking, not a drifted copy (issue #16, step 4c).
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
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

        # WHY a dedicated evaluation model: Using the same model that generated the
        #      answer to judge itself creates self-evaluation bias. A separate mid-tier
        #      model (gpt-4.1-mini by default) is cheap enough for real-time checks
        #      while strong enough to catch factual errors.
        self.eval_llm = LLMHandler(model=EVAL_MODEL, max_tokens=4096)

        # PATTERN: Conversation persistence is its own module, depending only
        #          on the session factory. The facade keeps its public methods
        #          so routes are unaffected, but conversation bugs and their
        #          tests now concentrate in one place.
        self.conversations = ConversationStore(session_factory=self._session)
        self.history = ConversationHistory(session_factory=self._session)

        # PATTERN: The evaluation cluster is its own module. The facade keeps
        #          the three public methods so routes are unaffected, but the
        #          skip/dedup decisions and their tests now live in one place.
        self.evaluator = MessageEvaluator(session_factory=self._session, judge_llm=self.eval_llm)

        # PATTERN: The QueryEngine owns retrieve->generate for both the sync and
        #          streaming paths. The Retriever is selected from config
        #          (RETRIEVER_STRATEGY) behind the seam, so a validated eval
        #          chain is promoted to production by configuration, not a
        #          rewrite. Default "dense" preserves current behaviour.
        # WHY the plan carries top_k: reranking changes how many chunks the
        #      engine ends up with, so the count is part of the composition
        #      rather than a constant the caller supplies alongside it. This
        #      rule used to exist only on the eval side.
        # WHY the answer handler is passed in: the multi_query strategy expands
        #      the user's query with an LLM call, and the facade already owns a
        #      configured handler. Every other strategy ignores the argument.
        plan = build_retrieval_plan(
            RETRIEVER_STRATEGY,
            self.vector_store,
            top_k=TOP_K_RESULTS,
            rerank_over_fetch_n=RERANK_OVER_FETCH_N,
            llm=self.llm,
        )
        self.query_engine = QueryEngine(
            retriever=plan.retriever,
            llm=self.llm,
            reasoning_llm=self.reasoning_llm,
            top_k=plan.top_k,
        )

        logger.info(
            "RAGBackend initialised (engine=%s, answer_model=%s, reasoning_model=%s, "
            "eval_model=%s, retriever=%s)",
            engine.url,
            DEFAULT_MODEL,
            REASONING_MODEL,
            EVAL_MODEL,
            RETRIEVER_STRATEGY,
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

        # BUG FIX: DocumentRecord.id is documented as SHA-256 of the raw file
        #          bytes, giving idempotent re-uploads. The loader used to
        #          derive a truncated 16-char hash of the *extracted text*,
        #          which (a) collides two different files with identical
        #          extracted text into one document and (b) truncates to
        #          64 bits — raising real collision risk at scale.
        # WHY compute here (not in the loader): the loader does not carry
        #          raw bytes around (it parses PDFs/DOCX into text), so the
        #          hash must be computed from `path` before chunking.
        document.doc_id = hashlib.sha256(path.read_bytes()).hexdigest()

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
            filename,
            document.doc_id,
            len(chunks),
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

        Thin wrapper over query_with_telemetry() that discards the StageTelemetry
        so existing callers (tests, route layer) see no behaviour change.

        Args:
            question: Natural language question from the user.
            top_k: Number of chunks to retrieve (default from config).
            model: LLM model override (creates a new handler if different).

        Returns:
            Dict with answer (str), sources (list[dict]), confidence (float).
        """
        result, _ = self.query_with_telemetry(question, top_k=top_k, model=model)
        return result

    def query_with_telemetry(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
    ) -> tuple[dict[str, Any], StageTelemetry]:
        """Run a full RAG query and return per-stage observability data.

        Delegates retrieve->generate to the shared QueryEngine, then shapes the
        {answer, sources, confidence} dict the API layer expects. Identical
        output to query(), plus a StageTelemetry (retrieve/generate timing and
        provider-reported token cost).

        RAG Pipeline Position:
          Question -> [QueryEngine: retrieve -> generate -> telemetry] -> (answer + telemetry)

        Args:
            question: Natural language question from the user.
            top_k: Number of chunks to retrieve (default from config).
            model: LLM model override.

        Returns:
            Tuple of (result_dict, StageTelemetry). result_dict has the shape
            {answer, sources, confidence}.
        """
        results, answer, telemetry = self.query_engine.ask(question, top_k=top_k, model=model)

        sources = [_source_dict(r) for r in results]

        # PATTERN: Confidence = clamped average of top-3 similarity scores; 0.0
        #          when there are no results (empty index or a refusal).
        top_scores = [r.score for r in results[: min(3, len(results))]]
        confidence = (
            round(max(0.0, min(1.0, sum(top_scores) / len(top_scores))), 4) if top_scores else 0.0
        )

        return (
            {"answer": answer, "sources": sources, "confidence": confidence},
            telemetry,
        )

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[BackendStreamEvent]:
        """Retrieve context and stream reasoning + answer with chain-of-thought events.

        Event stream shape (in order):
          ("status", str)         — programmatic retrieval milestones
          ("reasoning", str)      — LLM chain-of-thought tokens (brief pre-answer pass)
          ("token", str)          — final answer tokens
          ("done", dict)          — sources + persistence metadata  [UNCHANGED SHAPE]
          ("telemetry", dict)     — StageTelemetry dict (NEW — additive, after done)

        WHY telemetry is last: Old consumers that handle status/reasoning/token/done
        and ignore unknown event types continue to work. The ("telemetry", ...) event
        is purely additive — no existing event shape is modified.

        WHY two LLM calls: The reasoning pass uses a focused prompt that asks
        the model to think out loud about the retrieved context BEFORE giving
        an answer. This makes the agent's reasoning visible to users (like
        ChatGPT's "thinking" mode) at the cost of a short extra call.

        WHY only the answer is persisted: Reasoning is ephemeral scaffolding
        for UX, not a durable artifact of the conversation. Persisting it
        would double storage and confuse the sliding-window context.

        WHY telemetry covers the answer pass only (not reasoning):
        The reasoning pass uses a separate model (REASONING_MODEL) with its own
        cost. Telemetry here tracks the user-visible answer generation; the
        engine drops the reasoning pass's usage so the "per-query cost" display
        reflects one model's spend.

        WHY persistence is deferred to the terminal event: the QueryEngine owns
        retrieve->generate and yields a terminal ("result", ...) carrying the
        chunks + telemetry. This facade owns only conversation persistence — it
        saves the user and assistant messages together, and only when retrieval
        produced results, so an empty index leaves nothing persisted (as before)
        and all conversation writes concentrate at one point.

        Yields:
            Tuples as described above.
        """
        # The sliding window is prior *completed* pairs; the current question is
        # unpaired, so computing the window before persisting the user message is
        # behaviour-identical to computing it after (the old ordering).
        history = self._get_sliding_window(conversation_id) if conversation_id else []

        answer_parts: list[str] = []
        result: StreamResult | None = None
        for event_type, data in self.query_engine.ask_stream(
            question, top_k=top_k, model=model, history=history
        ):
            if isinstance(data, StreamResult):
                result = data  # internal terminal event — consumed, not forwarded
                continue
            if event_type == "token":
                answer_parts.append(data)
            yield (event_type, data)

        # The engine always emits exactly one terminal StreamResult (both the
        # normal and the no-results/refusal paths), so this binds every time.
        assert result is not None, "QueryEngine.ask_stream emitted no terminal result"
        results = result.results
        sources = [_source_dict(r) for r in results]

        if conversation_id and results:
            self._save_message(conversation_id, "user", question)
            assistant_msg_id = self._save_message(
                conversation_id,
                "assistant",
                "".join(answer_parts),
                model=result.model,
                sources=sources,
            )
            # WHY: Auto-title on the first turn so the sidebar shows something
            #      meaningful; message_id + conversation_id let the frontend
            #      update local state without re-fetching.
            self._auto_title(conversation_id, question)
            yield (
                "done",
                {
                    "sources": sources,
                    "message_id": assistant_msg_id,
                    "conversation_id": conversation_id,
                },
            )
        else:
            yield ("done", {"sources": sources})

        # Telemetry last (additive): the done event is what the client waits on
        # for sources; telemetry is a secondary signal.
        yield ("telemetry", result.telemetry.model_dump())

    # ------------------------------------------------------------------ #
    # Document management                                                  #
    # ------------------------------------------------------------------ #

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from both ChromaDB and SQLite.

        Cross-store order: ChromaDB first, then SQLite.

        WHY: Same reasoning as ingest — ChromaDB failure leaves SQLite clean.
        If ChromaDB succeeds but SQLite fails, we have orphan-free vectors
        (the SQLite record still references them, so a retry can clean up).

        Args:
            doc_id: The document's content-hash ID.

        Returns:
            Number of chunks removed (0 if the document was not found). The
            REST layer's response advertises `chunks_deleted` as a count —
            this is the value that populates it.
        """
        # STEP 1: Delete chunks from ChromaDB
        chunks_deleted = self.vector_store.delete_by_doc_id(doc_id)

        # STEP 2: Delete metadata from SQLite
        with self._session() as session:
            record = session.get(DocumentRecord, doc_id)
            if record is None:
                return 0
            session.delete(record)
            session.commit()

        logger.info("Deleted document doc_id=%s (%d chunks)", doc_id, chunks_deleted)
        return chunks_deleted

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for all ingested documents.

        Returns:
            List of dicts with id, filename, file_type, file_size_bytes,
            chunks_count, upload_date — sorted by upload_date descending.
        """
        with self._session() as session:
            records = session.exec(
                select(DocumentRecord).order_by(col(DocumentRecord.upload_date).desc())
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
        return self.vector_store.get_by_doc_id(doc_id)

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
    # Conversation CRUD — delegated to ConversationStore                   #
    # ------------------------------------------------------------------ #

    def create_conversation(self, title: str = "New Chat") -> dict[str, Any]:
        """Create a conversation.

        Args:
            title: Human-readable title.

        Returns:
            The conversation summary.
        """
        return self.conversations.create(title)

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return every conversation in sidebar order (pinned first).

        Returns:
            Conversation summaries.
        """
        return self.conversations.list_all()

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Return a conversation with its messages and their sources.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            The detail shape, or None when not found.
        """
        return self.conversations.get(conversation_id)

    def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any] | None:
        """Change a conversation's title and/or pinned state.

        Args:
            conversation_id: UUID of the conversation.
            title: New title, when supplied.
            pinned: New pinned state, when supplied.

        Returns:
            The updated summary, or None when not found.
        """
        return self.conversations.update(conversation_id, title=title, pinned=pinned)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation with its messages and sources.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            True when deleted, False when not found.
        """
        return self.conversations.delete(conversation_id)

    def search_conversations(self, query: str) -> list[dict[str, Any]]:
        """Find conversations by title or message content.

        Args:
            query: Search string.

        Returns:
            Matching conversation summaries.
        """
        return self.conversations.search(query)

    def export_conversation(self, conversation_id: str) -> str | None:
        """Render a conversation as Markdown.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            A Markdown transcript, or None when not found.
        """
        return self.conversations.export_markdown(conversation_id)

    def create_share_token(self, conversation_id: str) -> str | None:
        """Mint a read-only share token for a conversation.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            The token, or None when not found.
        """
        return self.conversations.create_share_token(conversation_id)

    def get_shared_conversation(self, token: str) -> dict[str, Any] | None:
        """Return the conversation a share token points at.

        Args:
            token: The share token.

        Returns:
            The detail shape, or None when the token matches nothing.
        """
        return self.conversations.get_by_share_token(token)

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_faithfulness_realtime(
        self,
        message_id: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """Score a freshly-generated answer for faithfulness and persist it.

        Args:
            message_id: UUID of the assistant Message to attach the score to.
            answer: The full generated answer text.
            contexts: Retrieved excerpts the answer should be grounded in.

        Returns:
            Dict with metric, score and reasoning; a zero-score sentinel on
            failure so callers can always read ``["score"]``.
        """
        return self.evaluator.score_realtime(message_id, answer, contexts)

    def evaluate_message(self, message_id: str) -> list[dict]:
        """Run every not-yet-recorded evaluation metric for a persisted message.

        Args:
            message_id: UUID of the assistant Message to evaluate.

        Returns:
            One dict per metric; empty when the message is not found.
        """
        return self.evaluator.score_message(message_id)

    def get_evaluation(self, message_id: str) -> list[dict]:
        """Return all stored evaluation scores for a message, calling no judge.

        Args:
            message_id: UUID of the assistant Message.

        Returns:
            One dict per stored metric; empty when nothing has been scored.
        """
        return self.evaluator.scores_for(message_id)

    # ------------------------------------------------------------------ #
    # Internal helpers — delegated to ConversationHistory                   #
    # ------------------------------------------------------------------ #

    def _save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist one message and its sources. See ConversationHistory."""
        return self.history.save_message(
            conversation_id, role, content, model=model, sources=sources
        )

    def _get_sliding_window(
        self,
        conversation_id: str,
        max_pairs: int = SLIDING_WINDOW_SIZE,
    ) -> list[dict[str, str]]:
        """Return the last N completed exchanges. See ConversationHistory."""
        return self.history.sliding_window(conversation_id, max_pairs=max_pairs)

    def _auto_title(self, conversation_id: str, first_query: str) -> None:
        """Name an untitled thread after its first question. See ConversationHistory."""
        self.history.auto_title(conversation_id, first_query)
