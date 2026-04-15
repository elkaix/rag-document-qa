# Chat History Persistence & ChromaDB Integration

**Date:** 2026-04-14
**Status:** Approved

## Overview

Add persistent chat history and upgrade the vector store from in-memory TF-IDF/numpy to ChromaDB with sentence-transformer embeddings. Conversations survive server restarts, users can search/export/share chat history, and document retrieval quality improves significantly with dense semantic embeddings.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conversation scope | Global + sources tagged per-message | Simplest to build; sources metadata gives traceability |
| Conversation list UI | Inside existing sidebar | Standard ChatGPT/Claude pattern; no layout changes |
| Embeddings | ChromaDB default (all-MiniLM-L6-v2) | Zero config, massive quality jump over TF-IDF |
| LLM context | Sliding window (last N exchanges) | Handles follow-ups without unbounded token growth |
| Features | Full: CRUD, search, export, pin, share | Complete UX users expect |
| Architecture | SQLModel + ChromaDB (two storage engines) | Right tool for each job: SQL for relational, vector DB for embeddings |

---

## Section 1: Database Schema (SQLModel)

Four tables in SQLite, all managed by SQLModel:

### conversations

| Column | Type | Notes |
|--------|------|-------|
| id | str (UUID) | Primary key |
| title | str | Auto-generated from first message, editable |
| pinned | bool | Default false |
| created_at | datetime | |
| updated_at | datetime | |
| share_token | str \| None | Nullable, for sharing |

### messages

| Column | Type | Notes |
|--------|------|-------|
| id | str (UUID) | Primary key |
| conversation_id | str | FK -> conversations.id, indexed |
| role | str | "user" \| "assistant" |
| content | str | Full message text |
| model | str \| None | Which LLM generated this |
| created_at | datetime | |
| token_count | int \| None | Future hook for token-aware windowing; stays NULL for now. Sliding window counts message pairs, not tokens. |

### message_sources

| Column | Type | Notes |
|--------|------|-------|
| id | int | Autoincrement PK |
| message_id | str | FK -> messages.id, indexed |
| doc_id | str | |
| chunk_id | str | |
| filename | str \| None | |
| score | float | |
| excerpt | str | |

### documents

| Column | Type | Notes |
|--------|------|-------|
| id | str (doc_id) | Primary key |
| filename | str | |
| file_type | str | |
| file_size_bytes | int | |
| chunks_count | int | |
| upload_date | datetime | |

### Relationships

- `Conversation` -> has many `Message` (cascade delete)
- `Message` -> has many `MessageSource` (cascade delete)
- Deleting a conversation removes all its messages and their sources automatically

SQLModel cascade pattern (from official docs):
- Parent side: `Relationship(back_populates="conversation", cascade_delete=True)`
- Child side: `Field(foreign_key="conversation.id", ondelete="CASCADE")`
- Both `cascade_delete` and `ondelete` must be set for proper cascade behavior

**Critical: SQLite foreign key enforcement.** SQLite does not enforce foreign keys by default. The engine must emit `PRAGMA foreign_keys=ON` on every connection. Use an event listener on the engine:
```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```
Without this, `ondelete="CASCADE"` silently does nothing.

### Design notes

- `message_sources` is a separate table (not JSON in messages) so you can query "which conversations referenced this document"
- `documents` table mirrors what's currently in `RAGBackend._documents` dict, making document metadata persistent
- `share_token` enables sharing via a unique URL without auth complexity
- `token_count` on messages enables the sliding window to count tokens, not just message count

### ID strategy and re-ingest dedup

Current IDs are deterministic SHA-256 content hashes (`src/document_loader.py:25-27`). `doc_id = hash(content)`, `chunk_id = hash(content + doc_id)`. This means re-uploading the same file produces identical IDs.

**Dedup behavior:** Use `collection.upsert()` instead of `collection.add()` for ChromaDB operations. Upsert is idempotent — re-uploading the same document overwrites chunks with identical content, no duplicates. For SQLite, use `INSERT OR REPLACE` (or SQLModel equivalent) on the documents table, keyed by doc_id.

**Different content, same filename:** Produces different doc_id (hash is content-based), so both versions coexist. This is the correct behavior — different content is a different document.

**Versioning:** Not in v1 scope. If needed later, add a `version` column to documents and make PK composite `(doc_id, version)`.

---

## Section 2: ChromaDB Integration

Replaces the current dual system (TF-IDF embedder + InMemoryVectorStore) with a single ChromaDB client.

### Storage layout

```
data/
├── rag.db              <- SQLite (conversations, messages, documents)
└── chroma/             <- ChromaDB persistent storage
    └── (managed internally by ChromaDB)
```

### Single collection approach

- One ChromaDB collection called `"documents"` holds all chunks
- Each chunk stored with metadata: `{doc_id, filename, chunk_index, file_type}`
- ChromaDB handles embedding via its default `all-MiniLM-L6-v2` sentence-transformer (ONNX runtime)
- At query time: `collection.query(query_texts=[question], n_results=top_k)`
- Collection created with `metadata={"hnsw:space": "cosine"}` for cosine similarity (best for semantic search)

### Deployment constraint: single-writer process

ChromaDB's embedded `PersistentClient` is **not process-safe** for multiple writers. Only one process may write to `data/chroma/` at a time. This is fine for this app because all writes go through the single FastAPI server process. Do NOT run multiple uvicorn workers writing to the same Chroma path. If horizontal scaling is needed later, switch to ChromaDB's client-server mode (`HttpClient` + standalone Chroma server).

### ChromaDB v1.0+ notes (released March 2025, rewritten in Rust)

- Use `chromadb.PersistentClient(path=...)` for embedded persistent storage
- Use `chromadb.EphemeralClient()` for tests (in-memory, no disk)
- `collection.query()` and `collection.get()` return embeddings as 2D NumPy arrays (not Python lists)
- `get_or_create_collection` ignores metadata if collection already exists — set metadata only on first creation
- Built-in auth removed in v1.0 — not needed for this single-user app

### What gets replaced

| Current | New |
|---------|-----|
| `TfidfEmbedder.fit_transform()` | ChromaDB auto-embeds on `collection.add()` |
| `TfidfEmbedder.transform()` (query) | ChromaDB auto-embeds on `collection.query()` |
| `InMemoryVectorStore.add_documents()` | `collection.add(documents, metadatas, ids)` |
| `InMemoryVectorStore.search()` | `collection.query(query_texts, n_results)` |
| `_rebuild_index()` on every add/delete | Not needed -- ChromaDB indexes incrementally |

### Key wins

- **No full re-index on every document add/delete.** TF-IDF IDF weights are corpus-dependent so `_rebuild_index()` re-embeds everything. Sentence-transformer embeddings are document-independent -- add/delete is O(1).
- **Persistence.** Restart the server, documents are still indexed.
- **Better semantic search.** Dense embeddings understand synonyms and paraphrase; TF-IDF only matches exact terms.

### Document deletion

`collection.delete(where={"doc_id": target_id})` -- ChromaDB supports metadata-filtered deletes natively.

### Cross-store consistency

SQLite and ChromaDB are independent stores — there is no distributed transaction. Partial failures can leave inconsistent state. Mitigate with operation ordering and compensating actions:

**Ingest order:** Write to ChromaDB first (via `collection.upsert()`), then save document metadata to SQLite. If SQLite fails after ChromaDB succeeds, the chunks are searchable but undiscoverable via the documents list — recoverable by retrying the SQLite insert. The reverse (SQLite first, ChromaDB fails) would list a document with no searchable content, which is worse UX.

**Deletion order:** Delete from ChromaDB first, then SQLite. If ChromaDB succeeds but SQLite fails, the document metadata remains in SQLite (recoverable -- retry the SQLite delete). The reverse order would leave orphaned embeddings in ChromaDB with no metadata record, which is harder to clean up.

**Health check:** Add a periodic or on-demand consistency check that compares doc_ids in SQLite vs distinct doc_id values in ChromaDB metadata, logging any orphans. Not required for v1 but the implementation plan should note it as a follow-up.

---

## Section 3: Backend Architecture Changes

`RAGBackend` refactored to use SQLModel sessions + ChromaDB instead of in-memory dicts/lists.

### Initialization (lifespan)

```
Startup:
  1. Create SQLite engine + tables (SQLModel.metadata.create_all)
  2. Connect ChromaDB persistent client (path="data/chroma")
  3. Get or create "documents" collection
  4. Store engine + chroma collection on app.state
  5. Instantiate RAGBackend(engine, collection, llm_model)

Shutdown:
  (nothing -- SQLite and ChromaDB handle their own cleanup)
```

### Method changes in RAGBackend

| Method | Current (in-memory) | New (persistent) |
|--------|---------------------|-------------------|
| `ingest_file()` | Chunks -> TF-IDF -> numpy array -> `_all_chunks` list | Chunks -> `collection.add()` + save doc metadata to SQLite |
| `query()` | `embedder.transform()` -> numpy cosine sim | `collection.query()` -> build context -> LLM |
| `stream_query()` | Same as query but streams | Same + prepend sliding window from SQLite |
| `delete_document()` | Filter list + `_rebuild_index()` | `collection.delete(where={doc_id})` + delete from SQLite |
| `list_documents()` | Return `_documents` dict | `SELECT * FROM documents ORDER BY upload_date DESC` |

### New methods on RAGBackend

**Conversation CRUD:**
- `create_conversation(title?) -> Conversation`
- `list_conversations(pinned_first=True) -> List[Conversation]`
- `get_conversation(id) -> Conversation + messages`
- `update_conversation(id, title?, pinned?) -> Conversation`
- `delete_conversation(id) -> None` (cascades to messages + sources)
- `search_conversations(query) -> List[Conversation]` (LIKE on title + message content)
- `export_conversation(id, format="md") -> str`
- `create_share_token(id) -> str`
- `get_shared_conversation(token) -> Conversation + messages`

**Message persistence (called internally during query/stream_query):**
- `_save_message(conversation_id, role, content, model, sources?) -> Message`
- `_get_sliding_window(conversation_id, max_pairs=5) -> List[dict]`

### Sliding window integration into stream_query()

```
1. Retrieve sliding window: last N exchanges from SQLite
2. Save user message to SQLite immediately (the question is always valid)
3. Build messages list: [system_prompt, ...window_messages, user_prompt_with_context]
4. Stream LLM response
5. After streaming completes successfully:
   - Save assistant message + sources to SQLite
6. On stream error:
   - Save assistant message with error content (e.g., "[Generation failed]")
   - User message is already persisted — not lost on disconnect
```

**Two-phase persistence:** The user message is saved *before* streaming begins. The assistant message is saved *after* streaming completes. This ensures:
- A disconnect/error never loses the user's question from history
- Partial assistant responses are not persisted (error state is recorded instead)
- The sliding window always includes the user turn even if the assistant response failed

### LLMHandler adaptation for sliding window

The current `LLMHandler.stream_response(prompt, system_prompt)` accepts a single string prompt. The sliding window requires a **messages list** (system + history + user). The handler needs a new method (or overload) that accepts `List[dict]` messages instead of a flat prompt string. This applies to all providers:
- OpenAI/GLM: already use `messages` list internally -- just expose it
- Anthropic: already uses `messages` list internally -- just expose it
- Ollama: currently uses `/api/generate` (prompt-based). Switch to `/api/chat` endpoint which accepts a messages list natively. This is a full payload structure change -- `/api/chat` expects `{"model": "...", "messages": [{"role": "user", "content": "..."}]}` instead of `{"model": "...", "prompt": "..."}`. This should be its own implementation task, not a side effect of sliding window work.

---

## Section 4: API Endpoints

### New REST endpoints

```
Conversations:
  GET    /api/conversations              -> list all (pinned first, then by updated_at desc)
  POST   /api/conversations              -> create new conversation
  GET    /api/conversations/:id          -> get conversation with messages + sources
  PATCH  /api/conversations/:id          -> update title, pinned status
  DELETE /api/conversations/:id          -> delete conversation + all messages
  GET    /api/conversations/search?q=    -> full-text search across titles and message content
  GET    /api/conversations/:id/export   -> export as markdown
  POST   /api/conversations/:id/share    -> generate share token, return share URL
  GET    /api/shared/:token              -> public read-only conversation view

Settings remain in browser localStorage (via existing useSettings hook). No backend settings endpoints -- this avoids scope creep and the frontend already handles persistence fine for single-user use.
```

### WebSocket changes (/api/chat)

Current payload:
```json
{"query": "...", "top_k": 5, "model": "glm-5.1"}
```

New payload adds `conversation_id`:
```json
{"query": "...", "top_k": 5, "model": "glm-5.1", "conversation_id": "uuid-here"}
```

New response messages:
```json
{"type": "token", "content": "..."}
{"type": "done", "sources": [...], "message_id": "uuid", "conversation_id": "uuid"}
```

The `done` message returns IDs so the frontend can associate the response with the right conversation without a refetch.

### Auto-titling

When the first message in a conversation is saved, the backend generates a title by taking the first 60 characters of the user's query, truncated at a word boundary. No LLM call for titling.

---

## Section 5: Frontend Changes

### New hooks

**useConversations():**
- `useQuery(["conversations"])` -> list conversations
- `createMutation` -> POST /api/conversations
- `deleteMutation` -> DELETE + invalidateQueries
- `updateMutation` -> PATCH (rename, pin/unpin)
- `searchQuery(q)` -> GET /api/conversations/search?q=

**useConversation(id):**
- `useQuery(["conversation", id])` -> get conversation with messages
- `exportMutation` -> GET /api/conversations/:id/export
- `shareMutation` -> POST /api/conversations/:id/share

### useChat() changes

- Accepts `conversationId` parameter
- On mount with existing ID: loads messages from `useConversation(id)` instead of empty state
- On `done` WebSocket message: invalidates `["conversations"]` query to refresh sidebar list
- New conversation flow: calls `createMutation`, gets ID, then opens WebSocket with that ID

### Sidebar changes (sidebar.tsx)

```
Current:                          New:
+-------------------+             +-------------------+
| Logo              |             | Logo              |
|                   |             |                   |
| Chat              |             | + New Chat        |
| Upload            |             | ----------------- |
| Documents         |             | Search...         |
|                   |             | ----------------- |
|                   |             | Pinned            |
|                   |             |  Conversation A   |
|                   |             | ----------------- |
|                   |             | Today             |
|                   |             |  Conversation B   |
|                   |             |  Conversation C   |
|                   |             | Yesterday         |
|                   |             |  Conversation D   |
|                   |             | ----------------- |
| ----------------- |             | Upload            |
| Settings          |             | Documents         |
| Model: GLM 5.1   |             | ----------------- |
| Top-K: [===]      |             | Settings          |
| Stats             |             | Model / Top-K     |
+-------------------+             +-------------------+
```

### Conversation list features

- Grouped by: Pinned, Today, Yesterday, This Week, Older
- Each item shows: truncated title, relative timestamp
- Right-click or `...` menu: Rename, Pin/Unpin, Export, Share, Delete
- Click to switch conversations
- "New Chat" button always visible at top
- Search input filters conversations in real-time (debounced, hits backend)

### Chat page changes

- URL becomes `/chat/:conversationId` (react-router dynamic segment)
- `/chat` with no ID redirects to new conversation or shows empty state
- `ChatThread` loads initial messages from DB, then appends streamed tokens
- Export button in chat header -> downloads markdown file
- Share button -> copies share link to clipboard

### Share page

- New route: `/shared/:token`
- Read-only conversation view, no sidebar, minimal layout
- Shows all messages + sources, no input box

**Router restructure required:** The current `App.tsx` nests all pages under `AppLayout` which always renders the sidebar. The `/shared/:token` route must be a **sibling** of the `AppLayout` route, not a child:
```tsx
const router = createBrowserRouter([
  {
    Component: AppLayout,          // has sidebar
    children: [
      { path: "chat/:conversationId?", Component: ChatPage },
      { path: "upload", Component: UploadPage },
      { path: "documents", Component: DocumentsPage },
    ],
  },
  { path: "shared/:token", Component: SharedPage },  // no sidebar
]);
```

---

## Section 6: Data Flow (End to End)

### New conversation flow

```
User clicks "+ New Chat"
  -> Frontend: POST /api/conversations -> gets {id, title: "New Chat"}
  -> Frontend: navigates to /chat/:id
  -> User types question
  -> Frontend: opens WebSocket, sends {query, conversation_id, top_k, model}
  -> Backend:
      1. Load sliding window (empty for new conversation)
      2. ChromaDB collection.query() -> top-K chunks with sources
      3. Build prompt: [system, ...window, user_context_question]
      4. Stream tokens -> WebSocket -> Frontend renders incrementally
      5. After stream completes:
         - Save user message to SQLite
         - Save assistant message + sources to SQLite
         - Auto-title: UPDATE conversation SET title = first_query[:60]
      6. Send {"type": "done", "sources": [...], "message_id", "conversation_id"}
  -> Frontend: invalidates ["conversations"] -> sidebar refreshes with new title
```

### Resume existing conversation

```
User clicks conversation in sidebar
  -> Frontend: navigates to /chat/:id
  -> Frontend: GET /api/conversations/:id -> loads messages + sources
  -> ChatThread renders full history
  -> User types follow-up question
  -> Frontend: opens WebSocket, sends {query, conversation_id, ...}
  -> Backend:
      1. Load sliding window: last 5 exchanges from this conversation
      2. ChromaDB query for relevant chunks
      3. Build prompt with window context -> LLM understands follow-ups
      4. Stream -> save -> done (same as above, but no auto-title)
```

### Document ingestion flow

```
User uploads file on /upload page
  -> Frontend: POST /api/upload (multipart)
  -> Backend:
      1. DocumentLoader.load() -> Document
      2. TextChunker.chunk() -> List[Chunk]
      3. ChromaDB collection.add(
           documents=[c.content for c in chunks],
           metadatas=[{doc_id, filename, chunk_index} for c in chunks],
           ids=[c.chunk_id for c in chunks]
         )
         # ChromaDB auto-embeds with all-MiniLM-L6-v2
      4. Save document metadata to SQLite documents table
  -> Frontend: invalidates ["documents"] -> doc table refreshes
```

### Export flow

```
User clicks Export on conversation
  -> GET /api/conversations/:id/export
  -> Backend builds markdown:
      # {title}
      *Exported {date}*

      ## User
      {message}

      ## Assistant
      {message}

      **Sources:** file.pdf (score: 0.87), ...
  -> Frontend triggers download as .md file
```

### Share flow

```
User clicks Share on conversation
  -> POST /api/conversations/:id/share
  -> Backend: generates UUID token, saves to conversation.share_token
  -> Returns share URL: /shared/{token}
  -> Frontend: copies to clipboard, shows toast

Anyone with link visits /shared/{token}
  -> GET /api/shared/{token}
  -> Backend: looks up conversation by share_token, returns messages + sources
  -> Frontend: renders read-only view
```

---

## Section 7: Dependencies & Configuration

### New Python dependencies

```
chromadb >= 1.0.0        # Vector store + default sentence-transformer embeddings
sqlmodel >= 0.0.24       # ORM (Pydantic + SQLAlchemy) for SQLite
```

**Note on async:** SQLModel's `Session` is synchronous. FastAPI automatically runs sync `def` endpoints and `Depends(get_session)` callables in a threadpool, so there is no performance penalty. No async SQLite driver needed.

### Removed/demoted dependencies

- `scikit-learn` -- no longer needed for TF-IDF (keep in requirements for tests but not imported in production path)
- `numpy` -- still used indirectly by ChromaDB, but no longer directly imported in backend

### Legacy TF-IDF code to deprecate/remove

The repo has TF-IDF-centric modules that will be superseded by ChromaDB. These must be explicitly handled to avoid a half-ported state:

| File | Action |
|------|--------|
| `src/pipeline.py` | **Remove.** Replaced entirely by `RAGBackend`. Currently unused by the API layer. |
| `src/retriever.py` | **Remove.** Cosine similarity search replaced by `collection.query()`. |
| `src/embeddings.py` | **Remove or gut.** TF-IDF embedding logic replaced by ChromaDB's built-in embedder. |
| `src/vector_store.py` | **Rewrite.** Replace `InMemoryVectorStore` / `QdrantVectorStore` with `ChromaVectorStore`. |
| `src/chunker.py` | **Keep.** Still used by `document_loader.py: TextChunker` which is the active chunking system. |
| `src/evaluation.py` | **Keep but update imports.** If it references TF-IDF embedder, update to use ChromaDB. |
| `tests/conftest.py` | **Rewrite fixtures.** Current deterministic SHA-256-seeded embeddings won't work with ChromaDB. Replace with `EphemeralClient()` + real sentence-transformer embeddings, or mock ChromaDB's embedding function for deterministic tests. |

### Configuration (src/config.py)

```python
DATA_DIR = "data"
SQLITE_URL = f"sqlite:///{DATA_DIR}/rag.db"
CHROMA_PATH = f"{DATA_DIR}/chroma"
CHROMA_COLLECTION = "documents"

# Chat defaults
DEFAULT_MODEL = "glm-5.1"
DEFAULT_TOP_K = 5
SLIDING_WINDOW_SIZE = 5       # number of exchange pairs
MAX_TITLE_LENGTH = 60
SHARE_TOKEN_LENGTH = 16
```

### File structure changes

```
src/
├── config.py              <- expanded with DB/Chroma config
├── database.py            <- NEW: engine, get_session, create_tables
├── models/                <- NEW: SQLModel table definitions
│   ├── __init__.py
│   ├── conversation.py    <- Conversation model
│   ├── message.py         <- Message + MessageSource models
│   └── document.py        <- Document model (replaces _documents dict)
├── vector_store.py        <- refactored: ChromaDB implementation replaces numpy
├── embeddings.py          <- simplified or removed (ChromaDB handles embedding)
├── backend.py             <- refactored: uses Session + ChromaDB
├── llm_handler.py         <- unchanged (already fixed max_tokens)
├── document_loader.py     <- unchanged
├── chunker.py             <- unchanged
└── api/
    ├── main.py            <- lifespan creates engine + ChromaDB client
    ├── models.py          <- expanded with conversation/message schemas
    ├── dependencies.py    <- NEW: get_session, get_backend dependencies
    └── routes/
        ├── query.py       <- WebSocket accepts conversation_id
        ├── upload.py      <- unchanged
        ├── documents.py   <- unchanged
        └── conversations.py <- NEW: full CRUD + search + export + share

frontend/src/
├── hooks/
│   ├── use-chat.ts        <- accepts conversationId, loads history
│   ├── use-conversations.ts <- NEW: list, create, delete, update, search
│   └── use-settings.ts    <- unchanged (or migrate to backend-persisted)
├── api/
│   ├── client.ts          <- new endpoints added
│   └── types.ts           <- new types for conversations, messages
├── components/
│   ├── layout/
│   │   └── sidebar.tsx    <- conversation list, search, grouping
│   ├── chat/
│   │   ├── chat-thread.tsx    <- loads from DB + appends stream
│   │   ├── chat-message.tsx   <- unchanged
│   │   ├── chat-input.tsx     <- unchanged
│   │   └── sources-panel.tsx  <- unchanged
│   └── shared/
│       └── shared-view.tsx    <- NEW: read-only shared conversation
└── pages/
    ├── chat.tsx           <- /chat/:conversationId route
    └── shared.tsx         <- NEW: /shared/:token route
```

---

## Section 8: Testing Strategy

### Backend tests

```
tests/
├── test_database.py          <- SQLModel table creation, session lifecycle
├── test_models.py            <- Conversation/Message CRUD, cascade deletes
├── test_vector_store.py      <- ChromaDB add/query/delete, metadata filtering
├── test_backend.py           <- Full RAG flow: ingest -> query -> verify sources
├── test_conversations.py     <- CRUD, search, export, share token generation
├── test_sliding_window.py    <- Window retrieval, token counting, edge cases
├── test_api_conversations.py <- REST endpoint integration tests
└── test_api_websocket.py     <- WebSocket with conversation_id, message persistence
```

### Key test scenarios

| Area | Tests |
|------|-------|
| Conversation lifecycle | Create -> query -> verify messages saved -> delete -> verify cascade |
| Sliding window | Empty conversation (no window), exactly N pairs, more than N pairs, single message |
| ChromaDB | Add chunks -> query returns relevant results -> delete doc -> query excludes deleted |
| Auto-titling | First message sets title, second message doesn't change it |
| Search | Matches title, matches message content, no results returns empty |
| Export | Markdown format includes messages + sources, handles special characters |
| Share | Generate token -> fetch by token -> returns read-only data, invalid token -> 404 |
| Pin | Pin conversation -> list returns pinned first |
| Persistence | Ingest docs -> restart server (new engine/client) -> docs and conversations still there |

### Test fixtures (conftest.py additions)

- `db_session` -- in-memory SQLite session (`sqlite://`) for fast isolated tests
- `chroma_collection` -- ephemeral ChromaDB collection (ChromaDB supports in-memory mode for tests)
- `backend` -- RAGBackend wired to test session + test collection
- `sample_conversation` -- pre-populated conversation with 3 exchanges and sources

### Frontend tests (if applicable)

- `useConversations` hook: list, create, delete trigger correct API calls
- `useChat` with `conversationId`: loads existing messages before WebSocket connect
- Sidebar: conversation grouping (pinned, today, yesterday), search filtering
- Share view: renders read-only, no input box visible
