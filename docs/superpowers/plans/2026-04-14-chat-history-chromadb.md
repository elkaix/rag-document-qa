# Chat History & ChromaDB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent chat history (SQLite via SQLModel) and replace the in-memory TF-IDF/numpy vector store with ChromaDB for document embeddings.

**Architecture:** Two storage engines — SQLite for relational data (conversations, messages, documents) and ChromaDB for vector search (document chunks + embeddings). RAGBackend is refactored to use both. Frontend gets a sidebar conversation list, dynamic chat routes, and share page.

**Tech Stack:** SQLModel 0.0.24, ChromaDB 1.0+, FastAPI lifespan, React Query mutations, react-router dynamic segments.

**Spec:** `docs/superpowers/specs/2026-04-14-chat-history-chromadb-design.md`

---

## Phase 1: Foundation (Tasks 1-5)

### Task 1: Dependencies and Configuration

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config.py`
- Modify: `.gitignore`

- [ ] Add `chromadb>=1.0.0` and `sqlmodel>=0.0.24` to `requirements.txt`
- [ ] Add `data/rag.db` and `data/chroma/` to `.gitignore`
- [ ] Expand `src/config.py` with database and ChromaDB settings:
  - `SQLITE_PATH = DATA_DIR / "rag.db"`, `SQLITE_URL = f"sqlite:///{SQLITE_PATH}"`
  - `CHROMA_PATH = str(DATA_DIR / "chroma")`, `CHROMA_COLLECTION = "documents"`
  - `DEFAULT_MODEL = "glm-5.1"`, `SLIDING_WINDOW_SIZE = 5`, `MAX_TITLE_LENGTH = 60`
- [ ] Run: `pip install chromadb>=1.0.0 sqlmodel>=0.0.24`
- [ ] Verify: `python3 -c "import chromadb; import sqlmodel; print('OK')"` -> `OK`
- [ ] Commit: `chore: add chromadb + sqlmodel deps, expand config`

---

### Task 2: Database Layer (Engine + SQLModel Tables)

**Files:**
- Create: `src/database.py`
- Create: `src/models/__init__.py`
- Create: `src/models/conversation.py`
- Create: `src/models/message.py`
- Create: `src/models/document.py`
- Test: `tests/test_database.py`

- [ ] Write `tests/test_database.py` with tests for:
  - `TestDatabaseSetup.test_tables_created` — verify all 4 tables exist after `create_db_and_tables()`
  - `TestDatabaseSetup.test_foreign_keys_enabled` — `PRAGMA foreign_keys` returns 1
  - `TestConversationModel.test_create_conversation` — create, commit, verify defaults (pinned=False, created_at set)
  - `TestConversationModel.test_cascade_delete_messages` — delete conversation, verify messages and sources also deleted
  - `TestDocumentRecordModel.test_create_document_record` — create, verify fields persisted

- [ ] Run: `python3 -m pytest tests/test_database.py -v` -> FAIL (modules not found)

- [ ] Create `src/models/__init__.py` — imports and re-exports all models

- [ ] Create `src/models/conversation.py`:

```python
class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str = Field(default="New Chat", max_length=200)
    pinned: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    share_token: Optional[str] = Field(default=None, index=True)
    messages: list["Message"] = Relationship(back_populates="conversation", cascade_delete=True)
```

- [ ] Create `src/models/message.py`:

```python
class Message(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    conversation_id: str = Field(foreign_key="conversation.id", ondelete="CASCADE", index=True)
    role: str  # "user" | "assistant"
    content: str = Field(default="")
    model: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: Optional[int] = Field(default=None)
    conversation: Optional["Conversation"] = Relationship(back_populates="messages")
    sources: list["MessageSource"] = Relationship(back_populates="message", cascade_delete=True)

class MessageSource(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: str = Field(foreign_key="message.id", ondelete="CASCADE", index=True)
    doc_id: str
    chunk_id: str
    filename: Optional[str] = None
    score: float
    excerpt: str = Field(default="")
    message: Optional["Message"] = Relationship(back_populates="sources")
```

- [ ] Create `src/models/document.py`:

```python
class DocumentRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)  # content-hash doc_id
    filename: str
    file_type: str = Field(default="")
    file_size_bytes: int = Field(default=0)
    chunks_count: int = Field(default=0)
    upload_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] Create `src/database.py`:
  - `get_engine(url)` — creates engine with `check_same_thread=False`, attaches `PRAGMA foreign_keys=ON` listener via `@event.listens_for(engine, "connect")`
  - `create_db_and_tables(engine)` — imports models, calls `SQLModel.metadata.create_all(engine)`
  - `get_session(engine)` — yields `Session(engine)` for FastAPI dependency injection

- [ ] Run: `python3 -m pytest tests/test_database.py -v` -> All PASS
- [ ] Commit: `feat: add SQLModel database layer with conversation/message/document tables`

---

### Task 3: ChromaDB Vector Store

**Files:**
- Rewrite: `src/vector_store.py`
- Test: `tests/test_vector_store_chroma.py`

- [ ] Write `tests/test_vector_store_chroma.py` with tests for:
  - `test_upsert_and_query` — upsert 3 chunks, query for RAG, top result is relevant
  - `test_delete_by_doc_id` — upsert chunks from 2 docs, delete one, verify only one remains
  - `test_upsert_is_idempotent` — upsert same ID 3 times, count is still 1
  - `test_get_stats` — verify `total_chunks` and `backend` fields
  - `test_query_empty_store` — returns empty list
  - Use `chromadb.EphemeralClient()` for test fixtures

- [ ] Run: `python3 -m pytest tests/test_vector_store_chroma.py -v` -> FAIL

- [ ] Rewrite `src/vector_store.py` with:
  - `SearchResult` dataclass (content, metadata, score, doc_id, chunk_id)
  - `ChromaVectorStore` class wrapping a ChromaDB Collection
  - Methods: `upsert(ids, documents, metadatas)`, `query(query_text, top_k, where)`, `delete_by_doc_id(doc_id)`, `get_stats()`
  - Convert ChromaDB cosine distance (0-2) to similarity score (0-1) via `score = 1 - distance`

- [ ] Run: `python3 -m pytest tests/test_vector_store_chroma.py -v` -> All PASS
- [ ] Commit: `feat: replace in-memory vector store with ChromaDB implementation`

---

### Task 4: LLMHandler Messages-List API

**Files:**
- Modify: `src/llm_handler.py`
- Test: `tests/test_llm_handler.py`

- [ ] Write `tests/test_llm_handler.py` with tests for:
  - `test_stream_messages_falls_back_to_dummy` — nonexistent model yields dummy tokens
  - `test_generate_messages_falls_back_to_dummy` — nonexistent model returns dummy string
  - `test_stream_messages_accepts_sliding_window` — messages list with history is accepted

- [ ] Run: `python3 -m pytest tests/test_llm_handler.py -v` -> FAIL

- [ ] Add to `LLMHandler`:
  - `generate_messages(messages: List[dict]) -> str` — dispatches to provider-specific methods
  - `stream_messages(messages: List[dict]) -> Generator[str, None, None]` — same, streaming
  - Provider methods: `_openai_generate_messages`, `_openai_stream_messages`, `_glm_generate_messages`, `_glm_stream_messages`, `_anthropic_generate_messages`, `_anthropic_stream_messages`, `_ollama_generate_messages`, `_ollama_stream_messages`
  - Anthropic methods must separate system messages from user/assistant messages
  - Ollama methods use `/api/chat` (not `/api/generate`) with `{"messages": [...]}` payload

- [ ] Run: `python3 -m pytest tests/test_llm_handler.py -v` -> All PASS
- [ ] Commit: `feat: add messages-list API to LLMHandler, switch Ollama to /api/chat`

---

### Task 5: Remove Legacy TF-IDF Code

**Files:**
- Remove: `src/pipeline.py`, `src/retriever.py`, `src/embeddings.py`, `src/chunker.py`
- Modify: `tests/conftest.py`

- [ ] Verify no production code imports the legacy modules (grep `src/api/` and `src/backend.py`)
- [ ] `git rm src/pipeline.py src/retriever.py src/embeddings.py src/chunker.py`
- [ ] Update `tests/conftest.py`:
  - Remove `InMemoryVectorStore` import, `_make_deterministic_embedding`, `EMBEDDING_DIM`, `mock_embeddings`, `mock_query_embedding` fixtures
  - Replace `populated_vector_store` fixture with ChromaDB-based version using `EphemeralClient()`
- [ ] Run: `python3 -m pytest tests/test_document_loader.py -v` -> PASS (existing tests unbroken)
- [ ] Commit: `chore: remove legacy TF-IDF modules (pipeline, retriever, embeddings, chunker)`

---

## Phase 2: Backend Refactor (Task 6)

### Task 6: RAGBackend with ChromaDB + SQLite + Conversations

**Files:**
- Rewrite: `src/backend.py`
- Test: `tests/test_backend.py`

- [ ] Write `tests/test_backend.py` with fixtures:
  - `backend` fixture: `RAGBackend(engine=tmp_sqlite_engine, collection=ephemeral_chroma_collection)`
  - `txt_file` fixture: temp .txt file with RAG-related content

- [ ] Tests for document operations:
  - `test_ingest_file` — ingest returns success with chunks_count > 0
  - `test_list_documents_after_ingest` — lists the ingested document
  - `test_query_after_ingest` — returns answer and sources
  - `test_delete_document` — removes from both stores
  - `test_reingest_same_file_is_idempotent` — same file twice, still 1 document

- [ ] Tests for conversation CRUD:
  - `test_create_conversation` — returns id and title
  - `test_list_conversations` — returns created conversations
  - `test_get_conversation_with_messages` — after saving messages, returns them
  - `test_update_conversation` — rename and pin
  - `test_delete_conversation_cascades` — messages and sources removed
  - `test_search_conversations` — finds by title and message content
  - `test_export_conversation` — returns markdown string
  - `test_share_token` — create token, retrieve by token

- [ ] Tests for sliding window:
  - `test_sliding_window_empty` — new conversation returns empty list
  - `test_sliding_window_excludes_unpaired` — save a user message with no assistant reply, window excludes it (only completed pairs)
  - `test_sliding_window_respects_limit` — 10 messages (5 pairs), window=2 pairs returns last 4

- [ ] Run: `python3 -m pytest tests/test_backend.py -v` -> FAIL

- [ ] Rewrite `src/backend.py`:
  - Constructor takes `engine` and `collection` (no more in-memory state)
  - `ingest_file()`: chunks -> `store.upsert()` -> save `DocumentRecord` to SQLite
  - `query()`: `store.query()` -> build context -> LLM generate
  - `stream_query(conversation_id=)`: save user msg -> load window (completed pairs BEFORE current turn, excludes just-saved user msg) -> build prompt [system, ...window, user_with_context] -> stream -> save assistant msg
  - `delete_document()`: ChromaDB first, then SQLite
  - Conversation CRUD: `create_conversation`, `list_conversations`, `get_conversation`, `update_conversation`, `delete_conversation`, `search_conversations`, `export_conversation`, `create_share_token`, `get_shared_conversation`
  - `_save_message()`, `_get_sliding_window()`, `_auto_title()`

- [ ] Run: `python3 -m pytest tests/test_backend.py -v` -> All PASS
- [ ] Commit: `feat: rewrite RAGBackend with ChromaDB + SQLite persistence and conversation CRUD`

---

## Phase 3: API Layer (Task 7)

### Task 7: API Lifespan, Dependencies, and Routes

**Files:**
- Modify: `src/api/main.py`
- Create: `src/api/dependencies.py`
- Modify: `src/api/models.py`
- Create: `src/api/routes/conversations.py`
- Modify: `src/api/routes/query.py`
- Modify: `src/api/routes/__init__.py`

- [ ] Rewrite `src/api/main.py`:
  - Lifespan creates `get_engine()`, `create_db_and_tables()`, `chromadb.PersistentClient(path=CHROMA_PATH)`, `get_or_create_collection()`, `RAGBackend(engine, collection)`
  - Include `conversations_router`

- [ ] Create `src/api/dependencies.py` with `get_backend(request)` dependency

- [ ] Add Pydantic models to `src/api/models.py`:
  - `ConversationCreate`, `ConversationUpdate`, `ConversationSummary`, `MessageInfo`, `ConversationDetail`

- [ ] Create `src/api/routes/conversations.py` with all endpoints:
  - `GET /api/conversations` -> list
  - `POST /api/conversations` -> create
  - `GET /api/conversations/search?q=` -> search (must be before `/:id` route)
  - `GET /api/conversations/:id` -> detail
  - `PATCH /api/conversations/:id` -> update
  - `DELETE /api/conversations/:id` -> delete
  - `GET /api/conversations/:id/export` -> markdown export
  - `POST /api/conversations/:id/share` -> generate share token
  - `GET /api/shared/:token` -> public read-only

- [ ] Update `src/api/routes/__init__.py` to export `conversations_router`

- [ ] Update `src/api/routes/query.py`:
  - WebSocket handler extracts `conversation_id` from payload
  - Passes to `backend.stream_query(conversation_id=...)`
  - `done` message includes `message_id` and `conversation_id`

- [ ] Start server: `uvicorn src.api.main:app --port 8001` and test:
  - `curl http://localhost:8001/health` -> `{"status":"healthy"}`
  - `curl -X POST http://localhost:8001/api/conversations -H "Content-Type: application/json" -d '{"title":"Test"}'` -> returns id
  - `curl http://localhost:8001/api/conversations` -> returns list

- [ ] Commit: `feat: add conversation CRUD routes, update lifespan for ChromaDB + SQLite`

---

## Phase 4: Frontend (Tasks 8-11)

### Task 8: Frontend Types and API Client

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] Add to `types.ts`: `ConversationSummary`, `MessageInfo`, `ConversationDetail` interfaces
- [ ] Update `WsDoneMessage` to include `message_id` and `conversation_id`
- [ ] Add to `client.ts` api object: `listConversations`, `createConversation`, `getConversation`, `updateConversation`, `deleteConversation`, `searchConversations`, `exportConversation`, `shareConversation`, `getShared`
- [ ] Commit: `feat: add conversation types and API client methods`

---

### Task 9: Frontend Hooks

**Files:**
- Create: `frontend/src/hooks/use-conversations.ts`
- Modify: `frontend/src/hooks/use-chat.ts`

- [ ] Create `use-conversations.ts` with React Query:
  - `useQuery(["conversations"])` for list
  - `useMutation` for create, delete, update
  - `search` function for filtering

- [ ] Update `use-chat.ts`:
  - `sendMessage` accepts optional `conversationId`
  - WebSocket payload includes `conversation_id`
  - On `done`, invalidates `["conversations"]` query
  - Add `loadMessages(msgs)` for loading existing conversation

- [ ] Commit: `feat: add useConversations hook, update useChat with conversation support`

---

### Task 10: Frontend Sidebar and Chat Page

**Files:**
- Modify: `frontend/src/components/layout/sidebar.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/pages/chat.tsx`
- Create: `frontend/src/pages/shared.tsx`

- [ ] Update `sidebar.tsx`:
  - Add "New Chat" button at top
  - Add search input (debounced 300ms)
  - Add scrollable conversation list grouped by: Pinned, Today, Yesterday, This Week, Older
  - Each item: truncated title, relative time, `...` menu (Rename, Pin/Unpin, Export, Share, Delete)
  - Active conversation highlighted based on URL param

- [ ] Update `App.tsx`:
  - Chat route: `path: "chat/:conversationId?"` (optional param)
  - Add sibling route: `{ path: "shared/:token", Component: SharedPage }`

- [ ] Update `pages/chat.tsx`:
  - Read `conversationId` from `useParams()`
  - If present: load conversation via `api.getConversation(id)`, populate ChatThread
  - Pass `conversationId` to `sendMessage`

- [ ] Create `pages/shared.tsx`:
  - Read `token` from `useParams()`
  - Fetch via `api.getShared(token)`
  - Render read-only ChatThread (no input, no sidebar)

- [ ] Verify in browser:
  1. New Chat -> creates conversation, URL updates
  2. Type question -> response streams, sidebar shows conversation
  3. Click different conversation -> loads its messages
  4. Share -> copy link, open in incognito -> read-only view

- [ ] Commit: `feat: add conversation list sidebar, chat routing, shared page`

---

### Task 11: End-to-End Smoke Test

- [ ] Run all backend tests: `python3 -m pytest tests/ -v` -> All PASS
- [ ] Start both servers and manually verify:
  1. Upload a document
  2. New Chat -> ask question about document -> streaming works
  3. Follow-up question -> sliding window context works
  4. Sidebar shows conversation with auto-generated title
  5. Rename, pin, export, share all work
  6. Delete conversation -> removed from sidebar
  7. Restart backend -> documents and conversations persist
- [ ] Fix any issues found, commit

---

## Follow-up Items (Not in This Plan)

- Cross-store consistency health check endpoint
- Token-aware sliding window (use `token_count` field)
- Full-text search index for message content (SQLite FTS5)
- Frontend keyboard shortcuts (Cmd+N for new chat)
- ChromaDB HttpClient mode for multi-worker deployment
