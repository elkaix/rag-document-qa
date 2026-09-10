# Architecture

## System Overview

RAG Document Q&A is a full-stack retrieval-augmented generation system. Users upload documents (PDF, DOCX, TXT, MD, HTML, CSV, JSON), the system chunks and embeds them into ChromaDB, and then answers natural-language questions by retrieving relevant chunks and generating responses through configurable LLM providers.

The system persists all state across restarts: document vectors in ChromaDB, metadata and chat history in SQLite.

```
┌──────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐   │
│  │  Upload   │  │   Chat   │  │ Documents │  │   Sidebar     │   │
│  │  Page     │  │   Page   │  │   Page    │  │ (convos/      │   │
│  │          │  │  (WS)    │  │           │  │  settings)    │   │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └───────────────┘   │
│       │              │              │                             │
│       └──────────────┼──────────────┘                             │
│                      │  REST + WebSocket                         │
└──────────────────────┼───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8001)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    RAGBackend (Facade)                    │    │
│  │                                                          │    │
│  │   ingest: src/ingestion/        ask: src/query_engine/    │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │    │
│  │  │  parsers +   │  │  chunking     │  │ QueryEngine  │  │    │
│  │  │  loader      │  │ (3 strategies)│  │ retrieve →   │  │    │
│  │  │  (7 formats) │  │               │  │ generate     │  │    │
│  │  └──────────────┘  └───────────────┘  └──────┬───────┘  │    │
│  │   conversations: src/conversations/          │          │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────┴───────┐  │    │
│  │  │ Conversation │  │ Conversation  │  │  Retriever   │  │    │
│  │  │ Store        │  │ History       │  │  seam +      │  │    │
│  │  │              │  │               │  │  LLMHandler  │  │    │
│  │  └──────────────┘  └───────────────┘  └──────────────┘  │    │
│  │                                                          │    │
│  └──────────────────────┬───────────────────────────────────┘    │
│                         │                                        │
│              ┌──────────┴──────────┐                             │
│              ▼                     ▼                              │
│  ┌───────────────────┐  ┌──────────────────┐                    │
│  │  ChromaDB         │  │  SQLite          │                    │
│  │  (Vector Store)   │  │  (SQLModel ORM)  │                    │
│  │                   │  │                  │                    │
│  │  - Chunk text     │  │  - Documents     │                    │
│  │  - Embeddings     │  │  - Conversations │                    │
│  │  - Metadata       │  │  - Messages      │                    │
│  │  - HNSW index     │  │  - Sources       │                    │
│  └───────────────────┘  └──────────────────┘                    │
│    data/chroma/           data/rag.db                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flows

### Indexing Pipeline

```
File Upload (multipart)
    │
    ▼
DocumentLoader.load()          ← src/ingestion/loader.py
    │                             format dispatch through the PARSERS registry
    │                             in src/ingestion/parsers.py:
    │                             PDF: pypdf (with line-break normalization)
    │                             DOCX: python-docx
    │                             HTML: BeautifulSoup
    │                             CSV/JSON/TXT/MD: stdlib
    ▼
Document { content, metadata, doc_id (SHA-256 hash) }   ← src/domain.py
    │
    ▼
TextChunker.chunk()            ← src/ingestion/chunking.py
    │                             recursive strategy by default (512 chars, 64 overlap)
    │                             separators: \n\n → \n → ". " → " " → ""
    │                             filters: MIN_CHUNK_LENGTH=20, dot-ratio < 15%
    ▼
List[Chunk] { content, metadata, chunk_id, doc_id }     ← src/domain.py
    │
    ├──► ChromaDB.upsert()     ← auto-embeds via all-MiniLM-L6-v2
    │                             cosine similarity, HNSW index
    │                             first-occurrence dedup: chunk ids are
    │                             content-addressed, so a document with
    │                             repeated text yields the same id twice
    │                             in one batch (Chroma rejects the batch)
    │
    └──► SQLite INSERT         ← DocumentRecord (filename, type, size, chunk count)
                                  idempotent via session.merge() on content-hash PK
```

### Query Pipeline (Streaming)

```
WebSocket message { query, model, top_k, conversation_id }
    │
    ▼
Retriever.retrieve(query, top_k)  ← src/retrieval/, selected by
    │                             RETRIEVER_STRATEGY through the single
    │                             composition rule in
    │                             src/retrieval/composition.py.
    │                             `dense` (default) goes straight to
    │                             ChromaDB.query(query_text): auto-embeds
    │                             the query, cosine nearest-neighbor,
    │                             returns top-K chunks with distances
    ▼
Status event: "Retrieved N chunks across M files"
    │
    ▼
PHASE 1: Reasoning Pass        ← separate LLMHandler (REASONING_MODEL: gpt-4.1-nano)
    │                             system prompt asks for 6-10 sentences of analysis
    │                             streams "reasoning" events token-by-token
    ▼
PHASE 2: Answer Pass            ← primary LLMHandler (user-selected model)
    │                             context = retrieved chunks + sliding window history
    │                             system prompt instructs markdown formatting
    │                             (##/### headings, **bold**, bullets, `code`)
    │                             streams "token" events
    ▼
"done" event { sources, message_id, conversation_id }
    │
    ├──► SQLite: save user Message + assistant Message + MessageSources
    └──► SQLite: auto-title conversation from first query
```

### Chat History (Sliding Window)

```
Conversation (SQLite)
    │
    ├── Message (user, Q1)
    ├── Message (assistant, A1)  ← with MessageSources
    ├── Message (user, Q2)
    ├── Message (assistant, A2)
    │   ...
    └── Message (user, Q_current) ← saved BEFORE streaming starts

_get_sliding_window(max_pairs=5):
    → returns last 5 completed user/assistant pairs
    → excludes the just-saved unpaired user message (prevents duplication)
    → passed as OpenAI-style messages list to LLM
```

---

## Backend Components

### `src/config.py` — Configuration

Centralized constants imported by every module. Key values:

| Constant | Default | Purpose |
|----------|---------|---------|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Chunks retrieved per query |
| `RETRIEVER_STRATEGY` | `dense` | Which retrieval composition to build — `dense`, `hybrid`, `reranked`, `multi_query`. Env-overridable |
| `RERANK_OVER_FETCH_N` | 20 | Candidates fetched before cross-encoder rerank |
| `HYBRID_BM25_TOP_K` / `HYBRID_DENSE_TOP_K` | 20 / 20 | Candidates each half of the hybrid retriever contributes |
| `HYBRID_RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `QUERY_REWRITER_MODEL` | unset | Model that expands queries under `multi_query`. Unset → the strategy refuses to build. Env-overridable |
| `MAX_QUERY_EXPANSIONS` | 3 | Alternative phrasings per query |
| `REFUSAL_SIMILARITY_THRESHOLD` | 0.35 | Below this, the refusal gate may decline |
| `DEFAULT_MODEL` | `gpt-5-mini` | Answer generation model |
| `REASONING_MODEL` | `gpt-4.1-nano` | Chain-of-thought model |
| `EVAL_MODEL` | `gpt-4.1-mini` | Judge model for message evaluation |
| `SLIDING_WINDOW_SIZE` | 5 | Max conversation pairs in context |
| `SQLITE_PATH` | `data/rag.db` | Database file |
| `CHROMA_PATH` | `data/chroma/` | Vector store directory |
| `API_HOST` / `API_PORT` | `0.0.0.0` / 8001 | Bind address for the local runner |

Two callables live here alongside the constants, both called only by
`src/api/main.py`:

- **`load_env()`** — reads `.env` into `os.environ`. Importing a library must
  never arm real credentials, so no module under `src/` calls `load_dotenv()`
  at import time; the application entry point is the one place allowed to.
- **`allowed_origins()`** — parses `ALLOWED_ORIGINS` into the CORS list.
  Unset, it falls back to `["*"]`, which is deliberate for local dev against a
  Vite server on another port. `docker-compose.prod.yml` pins it to the nginx
  origin so a deployed API is not wide open.

### `src/backend.py` — RAGBackend (Facade)

The central orchestrator, and a **facade only** — it implements no algorithm
itself and owns no persistence logic beyond wiring. Every cluster it once
contained now lives in a module it delegates to (see [ADR 0007](docs/adr/0007-backend-split.md)):

- **`ingest_file()`** / **`ingest_bytes()`** — parse (`src/ingestion/`) → chunk → ChromaDB upsert → SQLite metadata
- **`query()`** / **`query_with_telemetry()`** / **`stream_query()`** — delegated to `QueryEngine` (`src/query_engine/`), which owns retrieve → generate for both the sync and the streaming path
- **Conversation CRUD** — create, list, get, update, delete, search, export, share — delegated to `ConversationStore` (`src/conversations/store.py`)
- **`_get_sliding_window()`** / **`_auto_title()`** — delegated to `ConversationHistory` (`src/conversations/history.py`)
- **`evaluate_message()`** / **`get_evaluation()`** / **`evaluate_faithfulness_realtime()`** — delegated to `MessageEvaluator` (`src/evaluation/message_evaluator.py`)
- **Document CRUD** — `list_documents()`, `delete_document()`, `get_document_chunks()`, `get_stats()`

The facade's own interface is unchanged for callers; only its implementation moved.

Cross-store write order: ChromaDB first, then SQLite. If ChromaDB fails, SQLite is untouched; the reverse would leave phantom metadata records.

**Answer formatting:** The answer pass system prompt instructs the LLM to format responses with Markdown — `##`/`###` headings (max 3 levels), `**bold**` for key terms, bullet/numbered lists, `` `inline code` `` for technical terms, fenced code blocks, and `>` blockquotes for notable quotes. This ensures the frontend's `MarkdownRenderer` always has structured content to style.

### `src/domain.py` — Value Types

The leaf of the dependency graph: `Document`, `Chunk`, `SearchResult`, and
`content_hash()`. Frozen dataclasses with no vendor imports — importing this
module pulls in neither ChromaDB nor SQLModel, which a test pins by subprocess.
Every other module depends on these types; this module depends on nothing.
See [ADR 0005](docs/adr/0005-domain-value-types.md).

`content_hash()` returns the **full** SHA-256 hex digest. Document and chunk ids
derived from it are persisted in both SQLite and ChromaDB, so truncating it
would orphan every existing row.

### `src/ingestion/` — Parsing & Chunking

Three modules behind one seam (see [ADR 0008](docs/adr/0008-ingestion-parsing-seam.md)):

**`parsers.py`** — a `PARSERS` registry mapping extension → parse function.
`SUPPORTED_EXTENSIONS` is derived from the registry (`frozenset(PARSERS)`), so
adding a format is one entry, not two:
- PDF: `pypdf` with line-break normalization (`\n` → space, preserve `\n\n`) and hyphen-rejoin
- DOCX: `python-docx` paragraph extraction
- HTML: BeautifulSoup with script/style/nav/footer stripping
- CSV: header-value pair formatting per row
- JSON: pretty-printed text
- TXT/MD: direct read

**`loader.py`** — `DocumentLoader` resolves a path to a parser via
`parser_for()`, reads it, and builds a `Document` with a content-hash id. It
knows nothing about any individual format.

**`chunking.py`** — `TextChunker`, three strategies:
- **Fixed** — sliding window with character overlap
- **Recursive** — hierarchical splitting (`\n\n` → `\n` → `. ` → ` ` → `""`), overlap applied once at the top level via `_apply_word_overlap()` (word-boundary-safe)
- **Semantic** — sentence-aware accumulation with sentence-level overlap

Post-chunking filters discard chunks shorter than 20 characters and chunks with >15% dot characters (PDF table-of-contents artifacts).

### `src/vector_store.py` — ChromaVectorStore

Wrapper over a ChromaDB Collection that owns the cosine-space invariant:
- **`open()`** (classmethod) — get-or-create the collection with
  `SPACE_METADATA` applied. The distance → similarity conversion below is only
  correct in cosine space, so the store sets it rather than trusting each
  construction site to remember
- **`collection`** (property) — the underlying Chroma Collection, for the one
  production caller (`src/api/main.py`, wiring a `RAGBackend` that still takes a
  raw collection). Writing through it bypasses `revision`, so a derived index
  will not see the change; tests that do so are exercising the cold-start path
  on purpose
- **`upsert()`** — idempotent insert/update; auto-embeds via all-MiniLM-L6-v2 when no explicit embeddings provided. Chunk ids are content-addressed, so a document with repeated text (a boilerplate footer, a disclaimer page, a CSV with duplicate rows) produces the same id twice within one batch; the store keeps the first occurrence of each id rather than letting Chroma reject the whole upload
- **`all_chunks()`** — every stored chunk as a `Chunk` (text + metadata + doc_id), for BM25 corpus construction. Carrying metadata is what lets a sparse-only hit render a citation
- **`revision`** (property) — a counter bumped by every `upsert` and by every
  `delete_by_doc_id` that actually removed something (a delete matching nothing
  changes no chunk, so invalidating derived indexes for it only bought a wasted
  rebuild on the ordinary 404 and retry paths). Each mutation holds a write lock
  across the Chroma call *and* the increment, so the counter cannot be lost to a
  racing write and "this delete matched nothing" is still true when the counter
  decides not to move — the API serves retrieval from more than one thread. Derived indexes compare it to the one they were built at instead of every mutation site remembering to invalidate them — [ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md)
- **`query()`** — accepts `query_text` (production, auto-embedded) or `query_embedding` (tests, explicit); converts ChromaDB cosine distance `[0,2]` to similarity score `[0,1]`
- **`delete_by_doc_id()`** — removes all chunks for a document via metadata WHERE clause
- **`get_stats()`** — returns chunk count, backend name, collection name

### `src/retrieval/` — The Retriever Seam

A runtime-checkable `Retriever` Protocol — `retrieve(query, top_k) -> list[SearchResult]`
— with adapters that either conform directly or compose an inner Retriever
(`DenseRetriever`, `BM25HybridRetriever`, `RerankingRetriever`,
`MultiQueryRetriever`). See [ADR 0004](docs/adr/0004-retriever-seam-and-query-engine.md).

**`composition.py` owns the composition rule, once.** `compose_retrieval()`
takes a base Retriever plus optional rewriter and reranker and returns a frozen
`RetrievalPlan { retriever, top_k }`. It answers the two questions that used to
be answered independently in production and in eval — *what order do the levers
wrap in* and *what `top_k` does the caller ask for after a reranker has
over-fetched* — so the two paths agree by construction, not by coincidence.
`build_retrieval_plan(strategy, vector_store)` is the config-driven entry point.
See [ADR 0006](docs/adr/0006-one-retrieval-composition-rule.md).

**All four strategies are wired.** Each preset activates one lever over the
dense baseline; the eval harness stacks several at once by calling
`compose_retrieval` directly.

| `RETRIEVER_STRATEGY` | Composition | Needs |
|----------------------|-------------|-------|
| `dense` (default) | `DenseRetriever` | — |
| `hybrid` | `BM25HybridRetriever` — BM25 + dense fused by RRF | — |
| `reranked` | `RerankingRetriever` over dense | cross-encoder download on first use |
| `multi_query` | `MultiQueryRetriever` over dense | `QUERY_REWRITER_MODEL`; one LLM call per query |

**`corpus.py` keeps the sparse index honest.** `ChunkCorpus` mirrors the whole
collection and re-reads it whenever `ChromaVectorStore.revision` moves, so
documents ingested or deleted after startup are reflected on the next query.
This is what let the `hybrid` strategy ship — its corpus used to freeze at
construction time. Hybrid degrades to dense-only rather than raising on an
empty or untokenizable corpus, drops zero-score BM25 candidates, and returns
results in **fused-rank order**, so `RefusalHandler` reads the best score in
the list rather than position 0. `QueryRewriter` catches every provider
failure and falls back to the original query. See
[ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md).

**`fusion.py` holds Reciprocal Rank Fusion, used twice.** `BM25HybridRetriever`
fuses a sparse ranking with a dense one; `MultiQueryRetriever` fuses one
expansion's ranking with another's. Both combine lists whose *scores* live in
incomparable spaces but whose *ranks* always compare — which is why neither
adapter sorts by `score` any more. Two documented caveats: an id repeated
inside one ranking counts once at its best rank, and ties resolve toward the
list passed first (hybrid passes sparse first), because with no common scale
there is no principled tie-break.

**`sparse_index.py` owns the BM25 half.** Building the index and scoring a
query against it are pure — a `CorpusSnapshot` in, ranked ids out, no store and
no I/O — so they are testable without a retriever, an embedder or a vector
store. It also owns the degradation rule: `BM25Okapi` divides by the corpus
size and by the term count, so an empty collection (a fresh deployment's normal
state) and an all-whitespace one both raised `ZeroDivisionError`; the index
reports "not defined for this corpus" instead, and hybrid falls back to
dense-only.

**What the `Retriever` seam does and does not promise.** `retrieve` returns
results ordered by descending *relevance*, never by descending `score`. Each
strategy scores in its own space — dense reports cosine similarity in `[0, 1]`,
the reranker a raw cross-encoder logit, hybrid `0.0` for a sparse-only hit — so
a caller asking "how good is the best match?" must read `max(r.score for r in
results)`. Both `RefusalHandler.should_refuse` and `RAGBackend`'s `confidence`
had been reading position 0 and were corrected.

### `src/query_engine/` — QueryEngine

Owns retrieve → generate for **both** the sync and the streaming path behind a
two-method interface (`ask`, `ask_stream`). It owns the single answer prompt
(`prompt.py`), filename-prefixed context assembly, telemetry assembly
(`telemetry.py`), the streaming event protocol (`streaming.py`), and an optional
refusal gate checked before the no-documents branch. Only the streaming path
runs the reasoning pass, so the sync path keeps its single LLM call.

### `src/conversations/` — Conversation Persistence

Split out of the backend facade ([ADR 0007](docs/adr/0007-backend-split.md)):

- **`store.py`** — `ConversationStore`: create, list, get, update, delete,
  search, export-as-Markdown, share tokens. Takes a `session_factory`, opening
  one session per operation.
- **`history.py`** — `ConversationHistory`: `save_message()`,
  `sliding_window()`, `auto_title()`. Title truncation is word-boundary-safe.
- **`shaping.py`** — pure functions turning ORM rows into the JSON dicts the API
  returns (`conversation_summary`, `message_dict`, `source_dict`,
  `conversation_detail`). No session, no I/O; they are read directly by tests.

### `src/evaluation/` — Per-Message Judging

`judges.py` holds the faithfulness / answer-relevancy / context-precision LLM
judges. `message_evaluator.py` holds `MessageEvaluator`, which loads a stored
message and its sources, runs the judges (injected as a `Judges` dataclass, so
tests substitute stubs without monkeypatching), and persists
`MessageEvaluation` rows idempotently.

Distinct from `src/eval/`, which is the offline harness over labeled gold sets.

### `src/llm_handler/` — LLM Provider Routing

`LLMHandler` (in `src/llm_handler/__init__.py`) auto-detects the provider from the model-name prefix and selects **one adapter** at construction:

| Prefix | Provider | Adapter | API Key Env Var |
|--------|----------|---------|-----------------|
| `gpt*`, `o1*`, `o3*` | OpenAI | `OpenAICompatibleAdapter` | `OPENAI_API_KEY` |
| `claude*` | Anthropic | `AnthropicAdapter` | `ANTHROPIC_API_KEY` |
| `glm*` | Zhipu AI (OpenAI-compatible) | `OpenAICompatibleAdapter` | `GLM_API_KEY` |
| everything else | Ollama (localhost:11434) | `OllamaAdapter` | none |

Each provider lives behind a `ProviderAdapter` (`src/llm_handler/adapters/`) whose SDK client is **injected** via a zero-arg `client_factory`, so every provider path is unit-testable with a fake (`tests/test_llm_adapters.py`). Adapters return `GenerationResult(text, usage)`; streaming yields text chunks then a terminal `Usage`. Usage is provider-reported where the SDK supplies it, adapter-counted otherwise. See [ADR 0002](docs/adr/0002-provider-adapters.md).

`LLMHandler` owns provider selection, the single-prompt → messages translation, and the fallback: only `ProviderUnavailableError` (missing SDK, missing GLM key, Ollama connection refused) routes to the `DummyAdapter`; real API errors propagate.

Public API surfaces (unchanged for callers):
- `generate()` / `stream_response()` — single prompt string
- `generate_messages()` / `stream_messages()` — OpenAI-style messages list (for multi-turn chat)
- `generate_with_usage()` — returns `(text, prompt_tokens, completion_tokens)` from provider-reported usage

GPT-5 family and o-series models use `max_completion_tokens` instead of `max_tokens` and omit the `temperature` parameter (constrained to default) — handled inside `OpenAICompatibleAdapter`.

### `src/database.py` — SQLite/SQLModel

- **`get_engine()`** — creates engine with `check_same_thread=False` for FastAPI's threadpool; attaches `PRAGMA foreign_keys=ON` event listener per connection
- **`create_db_and_tables()`** — imports all model classes and runs `SQLModel.metadata.create_all()`
- **`get_session()`** — generator-based FastAPI dependency for session lifecycle

### `src/models/` — Data Models

Four SQLModel table classes forming a hierarchy:

```
DocumentRecord (documents)
    PK: id (content-hash SHA-256)
    filename, file_type, file_size_bytes, chunks_count, upload_date

Conversation (conversations)
    PK: id (UUID4)
    title, pinned, created_at, updated_at, share_token
    │
    └── Message (messages)                    [ON DELETE CASCADE]
            PK: id (UUID4)
            FK: conversation_id
            role, content, model, created_at, token_count
            │
            └── MessageSource (message_sources) [ON DELETE CASCADE]
                    PK: id (auto-increment)
                    FK: message_id
                    doc_id, chunk_id, filename, score, excerpt
```

`from __future__ import annotations` is intentionally omitted from model files because SQLModel evaluates field types at class-definition time.

---

## API Layer

### `src/api/main.py` — FastAPI Application

`load_env()` is called at module scope — this is the one place in the codebase
allowed to pull `.env` into the process, so that importing any library module
never arms real credentials.

Lifespan startup creates, in order:
1. SQLite engine + tables
2. ChromaDB PersistentClient + `ChromaVectorStore.open()` (cosine/HNSW)
3. `RAGBackend` on `app.state`
4. `RunRegistry` on `app.state` — the in-process eval run tracker, a singleton
   so `POST /api/eval/run` and `GET /api/eval/runs/{id}/status` share it
5. `init_observability()` — fail-quiet OpenTelemetry export

CORS middleware reads `allowed_origins()` rather than hardcoding a list, so
the security-relevant setting sits with the rest of configuration. Unset it is
`["*"]` (local dev); `docker-compose.prod.yml` sets it. Routes are mounted via
`include_router()`.

A `if __name__ == "__main__":` block runs uvicorn on `API_HOST:API_PORT`, so the
`python -m src.api.main` command documented in the README actually starts the
server. Docker invokes `uvicorn src.api.main:app` directly instead.

### Endpoints

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| `POST` | `/api/upload` | `upload_single` | Upload and index a single file |
| `POST` | `/api/upload/batch` | `upload_batch` | Upload multiple files |
| `POST` | `/api/query` | `query` | Synchronous RAG query |
| `WS` | `/api/chat` | `chat_websocket` | Streaming chat with chain-of-thought |
| `GET` | `/api/documents` | `list_documents` | List all indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | `delete_document` | Delete document and chunks |
| `GET` | `/api/documents/{doc_id}/chunks` | `get_document_chunks` | View document chunks |
| `GET` | `/api/conversations` | `list_conversations` | List all conversations |
| `POST` | `/api/conversations` | `create_conversation` | Create new conversation |
| `GET` | `/api/conversations/search` | `search_conversations` | Search by title/content |
| `GET` | `/api/conversations/{id}` | `get_conversation` | Get conversation with messages |
| `PATCH` | `/api/conversations/{id}` | `update_conversation` | Rename or pin |
| `DELETE` | `/api/conversations/{id}` | `delete_conversation` | Delete with cascade |
| `GET` | `/api/conversations/{id}/export` | `export_conversation` | Export as Markdown |
| `POST` | `/api/conversations/{id}/share` | `create_share_token` | Generate share token |
| `GET` | `/api/shared/{token}` | `get_shared_conversation` | View shared conversation |
| `POST` | `/api/messages/{message_id}/evaluate` | `evaluate_message` | Run the judges on one message |
| `GET` | `/api/messages/{message_id}/evaluation` | `get_evaluation` | Read stored judge scores |
| `GET` | `/health` | `health` | Health check |

The eval-harness routes (`/api/eval/*`) are tabled separately under
[Evaluation Harness](#api--ui). Together the two tables cover all 27 registered
operations: 26 HTTP method/path pairs across 23 paths, plus the WebSocket.

### WebSocket Protocol

Client sends:
```json
{"query": "...", "top_k": 5, "model": "gpt-5-mini", "conversation_id": "uuid"}
```

Server streams events in order:
```json
{"type": "status",    "content": "Searching indexed documents..."}
{"type": "status",    "content": "Retrieved 5 chunk(s) across 2 file(s): ..."}
{"type": "status",    "content": "Analyzing retrieved context (gpt-4.1-nano)..."}
{"type": "reasoning", "content": "I'll start by..."}
{"type": "status",    "content": "Composing answer..."}
{"type": "token",     "content": "The answer is..."}
{"type": "done",      "sources": [...], "message_id": "...", "conversation_id": "..."}
```

**Async/sync bridge:** `stream_query()` is a synchronous generator that makes blocking HTTP calls to LLM APIs. The WebSocket handler runs each `next(gen)` call via `asyncio.run_in_executor()` in the default thread pool — this keeps the event loop free so `send_json()` flushes each WebSocket frame immediately between tokens, enabling real-time streaming. The generator is closed in a `finally` block to prevent resource leaks on client disconnect.

### Dependency Injection

All six route modules take the backend through `BackendDep` —
`Annotated[RAGBackend, Depends(get_backend)]`, declared once in
`src/api/dependencies.py`. No route reaches into `request.app.state` directly,
so every route can be tested by overriding one dependency.

---

## Frontend

### Stack

- **React 19** with TypeScript
- **Vite** dev server with HMR
- **React Router v7** for client-side routing
- **TanStack Query** for server state (queries + mutations)
- **shadcn/ui** component library (Radix primitives + Tailwind)
- **Tailwind CSS** for styling
- **react-markdown** + **remark-gfm** — Markdown rendering with GFM extensions (tables, strikethrough, task lists, autolink literals)
- **react-syntax-highlighter** (PrismLight build) — Code block syntax highlighting with `oneLight` theme; registers only needed languages (Python, JS, TS, Bash, JSON, SQL, YAML, CSS, Markdown) for minimal bundle size

### Routes

| Path | Component | Description |
|------|-----------|-------------|
| `/chat/:conversationId?` | ChatPage | Main chat interface |
| `/upload` | UploadPage | Drag-and-drop file upload |
| `/documents` | DocumentsPage | Document library with stats |
| `/shared/:token` | SharedPage | Read-only shared conversation |

### Key Hooks

**`useChat()`** — WebSocket-based streaming chat:
- Opens a new WebSocket per query to `ws://host/api/chat`
- Tracks four event types: `status` → `reasoning` → `token` → `done`
- Measures CoT reasoning duration via `performance.now()` timestamps
- Uses a ref-based guard for the first-token stamp (immune to React StrictMode updater replay)
- Invalidates conversation list query on `done`

**`useConversations()`** — TanStack Query CRUD:
- `listQuery` with 30s refetch interval
- `createMutation`, `deleteMutation`, `updateMutation` — all invalidate on success

**`useSettings()`** — `useSyncExternalStore` backed by localStorage:
- Caches parsed settings to avoid Object.is() infinite re-render
- Default model: `gpt-5-mini`

**`useDocuments()`** / `useUploadFile()`** — document list query and upload mutation

### Component Architecture

```
App
└── AppLayout
    ├── Sidebar
    │   ├── New Chat button
    │   ├── Search input (debounced 300ms)
    │   ├── Conversation list (grouped: Pinned, Today, Yesterday, This Week, Older)
    │   │   └── ConversationItem (context menu: rename, pin, export, share, delete)
    │   ├── Nav links (Upload, Documents)
    │   ├── Settings (Model dropdown, Top-K slider)
    │   └── Collection stats (docs, chunks, size, types)
    │
    └── <Outlet>
        ├── ChatPage
        │   ├── ChatThread
        │   │   └── ChatMessage
        │   │       ├── ThinkingPanel (collapsible, status + reasoning)
        │   │       ├── MarkdownRenderer (GFM, syntax highlighting, copy buttons)
        │   │       └── CopyButton (hover-reveal, on both user and assistant bubbles)
        │   ├── ChatInput
        │   └── SourcesPanel (resizable via drag handle)
        │
        ├── UploadPage
        │   ├── Dropzone
        │   └── FileQueue
        │
        └── DocumentsPage
            ├── DocStats
            └── DocTable
                └── ChunkViewer
```

### ThinkingPanel Lifecycle

1. **Reasoning streaming** (`thinkingSeconds` undefined) — panel open, shimmering "Thinking" header, status bullets and italic reasoning text stream in a scrollable area (max ~4 lines, auto-scrolls to bottom)
2. **Answer starts** (`thinkingSeconds` set) — panel auto-collapses to compact "Thought for N.Ns" header; answer bubble begins streaming below
3. **Done** (`streamDone` true) — panel stays collapsed; user can click to re-expand and inspect full reasoning

### MarkdownRenderer

Custom component wrapping `react-markdown` with `remark-gfm` plugin and `PrismLight` syntax highlighter. Renders LLM output with:

- **Headings** (`##`, `###`) — border-bottom separator, bold, proper spacing
- **Bold/italic** — semibold key terms, italic emphasis
- **Lists** — bullet and numbered with gray markers, proper nesting
- **Code blocks** — language label header + copy button + Prism oneLight theme
- **Inline code** — purple monospace pill with gray background
- **Blockquotes** — blue left border + light blue background
- **Tables** — rounded borders, striped header, GFM alignment support
- **Strikethrough/task lists** — GFM extensions via remark-gfm
- **Max width** — 65ch for optimal line readability (60-75ch best practice)

---

## Infrastructure

### Docker

Two Dockerfiles:
- **`Dockerfile`** (production) — Python 3.12-slim, single uvicorn worker (ChromaDB is single-writer)
- **`Dockerfile.dev`** — development with hot reload

`docker-compose.yml` runs two services:
- **api** — FastAPI backend on port 8001, mounts `src/`, `tests/`, `data/`, `books/`; ChromaDB ONNX model cached in a named volume
- **frontend** — Vite dev server on port 3000, proxies `/api` to the api service

### Data Persistence

All runtime data lives in `data/`:
- `data/rag.db` — SQLite database (conversations, messages, sources, documents)
- `data/chroma/` — ChromaDB persistent storage (vectors, HNSW index)

Both are gitignored. The `data/` directory is created at import time by `config.py`.

### Environment Variables

| Variable | Required | Read by |
|----------|----------|---------|
| `OPENAI_API_KEY` | For OpenAI/GPT models | `src/llm_handler/providers.py` |
| `ANTHROPIC_API_KEY` | For Claude models | `src/llm_handler/providers.py` |
| `GLM_API_KEY` | For GLM/Zhipu models | `src/llm_handler/providers.py` |
| `GLM_BASE_URL` | Optional GLM endpoint override | `src/llm_handler/providers.py` |
| `ALLOWED_ORIGINS` | Optional; comma-separated CORS list. Unset → `*` (dev). Pinned in `docker-compose.prod.yml` | `src/config.py` (`allowed_origins()`) |
| `OTLP_ENDPOINT` | Optional; OpenTelemetry traces endpoint. Default `http://localhost:6006/v1/traces` | `src/observability.py` |
| `EVAL_RUNS_DIR` | Optional; where eval run directories are written. Default `eval_runs/`, resolved per call, not at import | `src/eval/storage.py` |
| `RETRIEVER_STRATEGY` | Optional; `dense` (default), `hybrid`, `reranked`, or `multi_query`. An unknown name fails at startup | `src/config.py` |
| `QUERY_REWRITER_MODEL` | Required *only* for `RETRIEVER_STRATEGY=multi_query`; the model that expands queries. Costs one extra LLM call per question | `src/config.py` |
| `EVAL_SQUAD_PATH` | Optional; path to the frozen SQuAD v2 JSONL | `src/eval/cli.py` |
| `EVAL_LLM_OVERRIDE_DUMMY` | Set to `1` to force the eval harness onto a deterministic dummy LLM | `src/eval/doubles.py` |
| `RAG_QA_LIVE_LLM` | **Tests only.** Set to `1` to let the suite make real, billable provider calls. Unset, `tests/conftest.py` stubs every provider — a clean checkout with a populated `.env` must never spend money | `tests/conftest.py` |

No env vars are required for basic operation — the system works with ChromaDB's built-in embeddings and dummy LLM responses.

`.env` is read **only** by `load_env()` in `src/config.py`, called from
`src/api/main.py` at module scope. No library module loads it at import time.

---

## Testing

Tests use isolated, in-memory instances of both stores:

`tests/conftest.py` stubs every LLM provider by default; a run only reaches a
real API when `RAG_QA_LIVE_LLM=1` is set deliberately.

| Test File | Scope | Fixtures |
|-----------|-------|----------|
| `test_domain.py` | Value types; pins `content_hash` to the full digest and pins the module vendor-free | subprocess import check |
| `test_ingestion_parsers.py` | PARSERS registry, per-format parse | tmp files |
| `test_ingestion_loader.py` | DocumentLoader dispatch + error paths | tmp files |
| `test_ingestion_chunking.py` | TextChunker, three strategies | in-memory strings |
| `test_vector_store_chroma.py` | ChromaVectorStore, incl. duplicate-id dedup | EphemeralClient, unit vectors |
| `test_database.py` | Engine, tables, cascade deletes | In-memory SQLite |
| `test_backend.py` | RAGBackend integration | EphemeralClient + in-memory SQLite |
| `test_conversations.py` | ConversationStore, History, shaping | In-memory SQLite |
| `test_evaluation.py`, `test_backend_evaluation.py` | Judges + MessageEvaluator | Injected stub judges |
| `test_query_engine.py` | QueryEngine sync + streaming parity | Fake Retriever, fake LLM |
| `test_retrieval_adapters.py`, `test_retrieval_composition.py` | Retriever contract across adapters; the single composition rule | Fake inner Retriever |
| `test_llm_adapters.py`, `test_llm_handler.py` | Per-provider adapters; fallback paths | Injected fake SDK clients |
| `test_eval_*.py` | The offline harness, end to end | Ephemeral Chroma, dummy eval LLM |
| `test_api_*.py` | Routes, schemas, run registry | `TestClient` + dependency overrides |

Run: `python -m pytest tests/ -v`

---

## Evaluation Harness

The `src/eval/` package provides a reproducible evaluation system over labeled gold sets, separate from the user-facing chat path.

### Layers

| Module | Responsibility |
|--------|----------------|
| `src/eval/schemas.py` | Pydantic contracts: `EvalQuestion`, `EvalResult`, `AggregatedMetric`, `RunMetadata`, `MetricDelta`, `CompareResult`. |
| `src/telemetry/pricing.py`, `src/telemetry/tokens.py` | **Core** (not eval): model price table + `cost_usd()`, and `count_tokens()`. Imported by both production telemetry and the eval harness (see [ADR 0003](docs/adr/0003-telemetry-ownership.md)). |
| `src/eval/statistics.py` | `bootstrap_ci()` and `paired_permutation_test()` for run-level confidence intervals and two-run significance testing. |
| `src/eval/metrics/retrieval.py` | Recall@k, MRR@k, nDCG@k over `(gold_chunk_ids, retrieved_chunk_ids)`. |
| `src/eval/metrics/operational.py` | Per-stage latency p50/p95/p99, cost, token aggregation. |
| `src/eval/metrics/refusal.py` | Regex + LLM-judge refusal correctness for unanswerable questions. |
| `src/eval/metrics/generation.py` | Adds `answer_correctness` (cosine + judge mean) and `context_recall`; the faithfulness/relevancy/context-precision judges from `src/evaluation/judges.py` are called directly by `src/eval/runner.py`. |
| `src/eval/datasets/squad_v2.py` | Seeded sample + frozen 200-row JSONL artifact from HuggingFace `squad_v2`. |
| `src/eval/datasets/ml_papers.py` | Hand-labeled dev set loader + manifest SHA-256 verification. |
| `src/eval/config.py` | YAML-loaded `EvalConfig`. |
| `src/eval/storage.py` | Run-directory CRUD over `eval_runs/<run_id>/`. |
| `src/eval/pipeline_factory.py` | Builds an isolated RAG pipeline per (config, dataset) using ephemeral Chroma. Token counting and pricing come from `src/telemetry/` (core), not the eval package. |
| `src/eval/aggregator.py` | Per-dataset + combined `AggregatedMetric` rows from per-question results. |
| `src/eval/runner.py` | Orchestrates `git_sha`, ingest, query+score loop, aggregation, persistence. |
| `src/eval/compare.py` | Two-run diff with paired permutation tests + per-question regressions/wins. |
| `src/eval/report.py` + `templates/eval/*.html.j2` | Standalone jinja2 HTML reports. |
| `src/eval/cli.py` | `run`/`list`/`show`/`compare` argparse subcommands. |
| `src/eval/submission.py` | The run-submission interface: `resolve_config()`, `reserve_run_id()`, `submit_run()`, and the `RunProgressSink` Protocol the API's `RunRegistry` satisfies. The route handler validates and dispatches; it owns no run logic. |
| `src/eval/doubles.py` | `DummyEvalLLM` and `resolve_llm_overrides()` — the deterministic LLM substitution the harness uses when `EVAL_LLM_OVERRIDE_DUMMY=1`. |
| `src/eval/embedders/bge_small.py` | Optional BGE-small embedder for retrieval experiments. |

### API + UI

`src/api/routes/eval.py` exposes:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/eval/configs` | List available eval configs |
| `POST` | `/api/eval/run` | Start a new eval run (dispatched via `BackgroundTasks`) |
| `GET` | `/api/eval/runs` | List all eval runs |
| `GET` | `/api/eval/runs/{id}` | Get run metadata |
| `GET` | `/api/eval/runs/{id}/results` | Per-question results |
| `GET` | `/api/eval/runs/{id}/results/{question_id}` | One question's result |
| `GET` | `/api/eval/runs/{id}/status` | Live status for in-progress runs |
| `GET` | `/api/eval/compare` | Two-run diff with significance tests |

Long-running runs dispatch via FastAPI `BackgroundTasks` and report progress
through an in-process `RunRegistry` (`src/api/services/eval_runs.py`). The
registry's `update_progress(run_id, n_completed, n_total=None)` learns the
total on the first callback, because the submitting route cannot know the
question count until the dataset is loaded — registering with a total of 0 and
never updating it froze every run's reported progress at 0.0 until it finished.
`progress_fraction()` is the one place that division lives.

React route `/eval/*` mounts three views:
- **`RunsList`** — sortable/filterable table with multi-select compare
- **`RunDetail`** — metric chart + per-question table with lazy expand
- **`CompareView`** — side-by-side bars + Top Wins / Top Regressions cards

Charts use `recharts` with CI whiskers.

### Eval Run Directory

Each run produces `eval_runs/<run_id>/` with:
- `metadata.json` — run ID, git SHA, config name, timestamps
- `questions.jsonl` — per-question scores and retrieved chunks
- `metrics.json` — aggregated metric values with bootstrap CIs
- `cost.json` — token counts and USD costs per model
- `config.yaml` — snapshot of the config used

The `eval_runs/` directory is gitignored; the labeled dev sets in `eval_data/` are checked in.

---

## Observability

The system exports per-stage spans for every chat query via OpenTelemetry to [Arize Phoenix](https://github.com/Arize-ai/phoenix) on `localhost:6006`.

### Spans

`RAGBackend.query_with_telemetry` and `RAGBackend.stream_query` open spans:

| Span | Attributes |
|------|------------|
| `rag.retrieve` | `top_k`, `chunk_count` |
| `rag.generate` | `model`, `prompt_tokens`, `completion_tokens`, `cost_usd` |

### Telemetry Payload

The same numbers are returned to the client as a `StageTelemetry` Pydantic model (`src/api/schemas/telemetry.py`). Token counts are the **provider-reported usage** for the answer pass (from the LLM adapter's `GenerationResult` / streaming terminal `Usage`), with the adapter's local count as fallback — never a reconstructed prompt. Telemetry covers the answer pass only, not the chain-of-thought reasoning pass.

- REST `POST /api/query` — includes a `telemetry` field in the response JSON.
- WebSocket `/api/chat` — emits a final `{"type": "telemetry", "content": {...}}` event after the existing `done` event.

The frontend renders these as a muted footer line under each assistant chat bubble:

> *Retrieve 142ms · Generate 2.1s · 4,217 tok · $0.0083*

with a hover tooltip showing the prompt/completion token split.

### Running with Traces

Phoenix is profile-gated in `docker-compose.yml`; bare `docker compose up` does not start it.

```bash
docker compose --profile observability up
```

`init_observability()` (`src/observability.py`) is called during the FastAPI lifespan startup. It is idempotent and fail-quiet — if Phoenix is unreachable, spans become no-ops and the chat continues to work normally.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | ChromaDB | Pure Python, no external service, built-in embeddings |
| Embedding model | all-MiniLM-L6-v2 (via ChromaDB) | Zero-config, runs locally, 384-dim |
| Relational store | SQLite via SQLModel | Zero-ops, file-based, ORM convenience |
| Frontend | React + Vite | SPA with component reuse, fast HMR |
| Streaming | WebSocket | Bi-directional, low latency for token streaming |
| Reasoning | Separate cheap model | Visible CoT without doubling cost on the answer model |
| Chunking | Recursive (default) | Respects paragraph/sentence boundaries |
| Document ID | Content-hash (SHA-256, full digest) | Idempotent re-ingestion |
| Value types | Vendor-free leaf module (`src/domain.py`) | Nothing depends upward on ChromaDB or SQLModel — [ADR 0005](docs/adr/0005-domain-value-types.md) |
| Retrieval composition | One rule, one owner (`src/retrieval/composition.py`) | Production and eval compose levers identically by construction — [ADR 0006](docs/adr/0006-one-retrieval-composition-rule.md) |
| Backend shape | Facade that delegates, never implements | Conversation, evaluation and query clusters are testable without the facade — [ADR 0007](docs/adr/0007-backend-split.md) |
| Format support | Registry keyed by extension | Adding a parser is one entry; `SUPPORTED_EXTENSIONS` derives from it — [ADR 0008](docs/adr/0008-ingestion-parsing-seam.md) |
| Configuration | Injected, never globally mutated | `.env` is loaded once, at the entry point; libraries stay credential-free on import |
| Retrieval score semantics | The seam orders by *relevance*; `score` is strategy-specific and not cross-comparable | A caller asking "how good is the best match?" reads `max(...)`, never position 0 — [ADR 0010](docs/adr/0010-score-contract-and-retrieval-hardening.md) |
| Sparse-index freshness | Store publishes a revision; indexes ask | A new mutation site cannot forget to invalidate, because the counter lives with the write — [ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md) |
