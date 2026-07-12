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
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │    │
│  │  │ DocumentLoader│  │ TextChunker   │  │  LLMHandler  │  │    │
│  │  │ (7 formats)  │  │ (3 strategies)│  │ (4 providers)│  │    │
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
DocumentLoader.load()          ← format detection by extension
    │                             PDF: pypdf (with line-break normalization)
    │                             DOCX: python-docx
    │                             HTML: BeautifulSoup
    │                             CSV/JSON/TXT/MD: stdlib
    ▼
Document { content, metadata, doc_id (SHA-256 hash) }
    │
    ▼
TextChunker.chunk()            ← recursive strategy by default (512 chars, 64 overlap)
    │                             separators: \n\n → \n → ". " → " " → ""
    │                             filters: MIN_CHUNK_LENGTH=20, dot-ratio < 15%
    ▼
List[Chunk] { content, metadata, chunk_id, doc_id }
    │
    ├──► ChromaDB.upsert()     ← auto-embeds via all-MiniLM-L6-v2
    │                             cosine similarity, HNSW index
    │
    └──► SQLite INSERT         ← DocumentRecord (filename, type, size, chunk count)
                                  idempotent via session.merge() on content-hash PK
```

### Query Pipeline (Streaming)

```
WebSocket message { query, model, top_k, conversation_id }
    │
    ▼
ChromaDB.query(query_text)     ← auto-embeds query, cosine nearest-neighbor
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
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Chunks retrieved per query |
| `DEFAULT_MODEL` | `glm-5.1` | Answer generation model |
| `REASONING_MODEL` | `gpt-4.1-nano` | Chain-of-thought model |
| `SLIDING_WINDOW_SIZE` | 5 | Max conversation pairs in context |
| `SQLITE_PATH` | `data/rag.db` | Database file |
| `CHROMA_PATH` | `data/chroma/` | Vector store directory |

### `src/backend.py` — RAGBackend (Facade)

The central orchestrator. Coordinates four subsystems without implementing any algorithm itself:

- **`ingest_file()`** / **`ingest_bytes()`** — parse → chunk → ChromaDB upsert → SQLite metadata
- **`query()`** — ChromaDB search → context assembly → LLM generation (non-streaming)
- **`stream_query()`** — same flow but yields `(event_type, data)` tuples for WebSocket streaming with chain-of-thought reasoning
- **Conversation CRUD** — create, list, get, update, delete, search, export, share
- **`_get_sliding_window()`** — extracts completed message pairs for multi-turn context
- **`_auto_title()`** — sets conversation title from the first user query

Cross-store write order: ChromaDB first, then SQLite. If ChromaDB fails, SQLite is untouched; the reverse would leave phantom metadata records.

**Answer formatting:** The answer pass system prompt instructs the LLM to format responses with Markdown — `##`/`###` headings (max 3 levels), `**bold**` for key terms, bullet/numbered lists, `` `inline code` `` for technical terms, fenced code blocks, and `>` blockquotes for notable quotes. This ensures the frontend's `MarkdownRenderer` always has structured content to style.

### `src/document_loader.py` — Document Loading & Chunking

**DocumentLoader** — format-agnostic file parser:
- PDF: `pypdf` with line-break normalization (`\n` → space, preserve `\n\n`) and hyphen-rejoin
- DOCX: `python-docx` paragraph extraction
- HTML: BeautifulSoup with script/style/nav/footer stripping
- CSV: header-value pair formatting per row
- JSON: pretty-printed text
- TXT/MD: direct read

**TextChunker** — three strategies:
- **Fixed** — sliding window with character overlap
- **Recursive** — hierarchical splitting (`\n\n` → `\n` → `. ` → ` ` → `""`), overlap applied once at the top level via `_apply_word_overlap()` (word-boundary-safe)
- **Semantic** — sentence-aware accumulation with sentence-level overlap

Post-chunking filters discard chunks shorter than 20 characters and chunks with >15% dot characters (PDF table-of-contents artifacts).

### `src/vector_store.py` — ChromaVectorStore

Thin wrapper over a ChromaDB Collection:
- **`upsert()`** — idempotent insert/update; auto-embeds via all-MiniLM-L6-v2 when no explicit embeddings provided
- **`query()`** — accepts `query_text` (production, auto-embedded) or `query_embedding` (tests, explicit); converts ChromaDB cosine distance `[0,2]` to similarity score `[0,1]`
- **`delete_by_doc_id()`** — removes all chunks for a document via metadata WHERE clause
- **`get_stats()`** — returns chunk count, backend name, collection name

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

Lifespan startup creates:
1. SQLite engine + tables
2. ChromaDB PersistentClient + collection (cosine/HNSW)
3. RAGBackend instance on `app.state`

CORS middleware allows all origins (development). Routes are mounted via `include_router()`.

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
| `GET` | `/health` | `health` | Health check |

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

Conversation routes use the modern `Annotated[RAGBackend, Depends(get_backend)]` pattern. Upload, query, and document routes access `request.app.state.backend` directly.

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

| Variable | Required | Used By |
|----------|----------|---------|
| `OPENAI_API_KEY` | For OpenAI/GPT models | LLMHandler |
| `ANTHROPIC_API_KEY` | For Claude models | LLMHandler |
| `GLM_API_KEY` | For GLM/Zhipu models | LLMHandler |
| `GLM_BASE_URL` | Optional GLM endpoint override | LLMHandler |

No env vars are required for basic operation — the system works with ChromaDB's built-in embeddings and dummy LLM responses.

---

## Testing

Tests use isolated, in-memory instances of both stores:

| Test File | Scope | Fixtures |
|-----------|-------|----------|
| `test_document_loader.py` | DocumentLoader + TextChunker | tmp files |
| `test_vector_store_chroma.py` | ChromaVectorStore | EphemeralClient, 3D unit vectors |
| `test_database.py` | Engine, tables, cascade deletes | In-memory SQLite |
| `test_backend.py` | RAGBackend integration | EphemeralClient + in-memory SQLite |
| `test_llm_handler.py` | LLMHandler fallback paths | Dummy model (no live provider) |

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
| `src/eval/metrics/generation.py` | Adds `answer_correctness` (cosine + judge mean) and `context_recall`; the faithfulness/relevancy/context-precision judges from `src/evaluation.py` are called directly by `src/eval/runner.py`. |
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

### API + UI

`src/api/routes/eval.py` exposes:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/eval/configs` | List available eval configs |
| `POST` | `/api/eval/run` | Start a new eval run (dispatched via `BackgroundTasks`) |
| `GET` | `/api/eval/runs` | List all eval runs |
| `GET` | `/api/eval/runs/{id}` | Get run metadata |
| `GET` | `/api/eval/runs/{id}/results` | Per-question results |
| `GET` | `/api/eval/runs/{id}/status` | Live status for in-progress runs |
| `GET` | `/api/eval/compare` | Two-run diff with significance tests |

Long-running runs dispatch via FastAPI `BackgroundTasks` and report progress through an in-process `RunRegistry` (`src/api/services/eval_runs.py`).

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
| Document ID | Content-hash (SHA-256) | Idempotent re-ingestion |
