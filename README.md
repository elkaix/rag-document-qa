<div align="center">

<br />

<img src="https://img.shields.io/badge/RAG-Document_Q&A-0d74e7?style=for-the-badge&labelColor=24292d" alt="RAG Document Q&A" />

<br /><br />

# RAG Document Q&A

**Upload documents. Ask questions. Get cited answers.**

A production-grade Retrieval-Augmented Generation system with real-time streaming,
multi-provider LLM support, and built-in answer evaluation.

<br />

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vite.dev)
[![Tailwind](https://img.shields.io/badge/Tailwind-4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0-FF6F61?style=flat-square)](https://www.trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-2fbb4f?style=flat-square)](LICENSE)

<br />

<!-- Replace with actual screenshots when available -->
<!-- <img src="docs/images/chat.png" alt="Chat interface" width="820" /> -->

</div>

---

## Highlights

- **Full RAG Pipeline** — document ingestion, chunking, embedding, retrieval, and generation in one system
- **Real-Time Streaming** — token-by-token answer delivery over WebSocket with chain-of-thought reasoning
- **Multi-Provider LLM** — OpenAI, Anthropic, and Ollama with automatic provider detection
- **Answer Evaluation** — RAGAS-inspired faithfulness, relevancy, and precision scoring (LLM-as-judge)
- **7 File Formats** — PDF, DOCX, TXT, Markdown, HTML, CSV, JSON
- **Persistent Storage** — ChromaDB vector store + SQLite for conversations, messages, and sources
- **No API Keys Required** — runs fully standalone with local embeddings and dummy LLM fallback

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INDEXING PIPELINE                              │
│                                                                         │
│   📄 Document  →  Loader  →  Chunker  →  ChromaDB Embedder  →  HNSW   │
│      (7 formats)    │       (recursive)   (all-MiniLM-L6-v2)   Index   │
│                     │                                                   │
│                     └──→  SQLite (metadata, doc records)                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                 │
│                                                                         │
│   ❓ Question  →  Embed  →  Cosine Search  →  Top-K Chunks             │
│                                                    │                    │
│                              ┌─────────────────────┘                    │
│                              ▼                                          │
│                   ┌─── Reasoning (CoT) ───┐                             │
│                   │   gpt-4.1-nano        │                             │
│                   └───────────┬───────────┘                             │
│                               ▼                                         │
│                   ┌─── Generation ────────┐                             │
│                   │   user-selected model │──→  💬 Streaming Answer     │
│                   └───────────┬───────────┘     with Source Citations   │
│                               ▼                                         │
│                   ┌─── Evaluation ────────┐                             │
│                   │   Faithfulness check  │──→  📊 Quality Score        │
│                   └───────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

The **RAGBackend** facade orchestrates all components and persists state via `app.state`, so documents indexed through `/upload` are immediately searchable via `/chat`.

---

## Tech Stack

<table>
<tr>
<td width="140"><b>🖥️ Frontend</b></td>
<td>React 19 · TypeScript · Vite 7 · Tailwind CSS v4 · shadcn/ui · TanStack Query</td>
</tr>
<tr>
<td><b>⚙️ Backend</b></td>
<td>Python · FastAPI · Uvicorn · Pydantic v2 · SQLModel</td>
</tr>
<tr>
<td><b>🔍 Retrieval</b></td>
<td>ChromaDB (HNSW index) · all-MiniLM-L6-v2 embeddings (384-dim) · Cosine similarity</td>
</tr>
<tr>
<td><b>🤖 LLM</b></td>
<td>OpenAI (GPT-5, GPT-4.1) · Anthropic (Claude) · Ollama (Llama 3, Mistral, local models)</td>
</tr>
<tr>
<td><b>📄 Parsing</b></td>
<td>pypdf · python-docx · BeautifulSoup4</td>
</tr>
<tr>
<td><b>💾 Storage</b></td>
<td>SQLite (conversations, messages, evaluations) · ChromaDB (vector persistence)</td>
</tr>
<tr>
<td><b>📦 Deploy</b></td>
<td>Docker Compose · Multi-stage builds</td>
</tr>
</table>

---

## Features

### Document Ingestion
- **Drag-and-drop upload** with instant auto-indexing — no manual "upload" button needed
- **7 supported formats**: PDF, DOCX, TXT, Markdown, HTML, CSV, JSON (50 MB max)
- **3 chunking strategies**: fixed-size, recursive (paragraph-aware), semantic (sentence-level)
- **Smart filtering**: strips chunks under 20 chars or with >15% dots (table-of-contents noise)
- **Idempotent upsert**: re-uploading the same document overwrites existing chunks

### Streaming Chat
- **WebSocket protocol** delivers tokens in real-time as they generate
- **Two-phase generation**: lightweight reasoning pass (gpt-4.1-nano) followed by full answer
- **Chain-of-thought panel**: collapsible display of the model's reasoning process with timing
- **Markdown rendering**: GitHub Flavored Markdown with syntax-highlighted code blocks
- **Conversation history**: sliding window context (5 turns) for follow-up questions

### Source Attribution
- **Resizable sources panel** with per-chunk relevance scores
- **Collapsible source cards** showing document name, excerpt, and chunk/doc IDs
- **Score-based ordering**: most relevant chunks surface first
- **Full traceability**: every answer links back to specific document chunks

### Conversation Management
- **Persistent conversations** stored in SQLite with full message history
- **Pin, rename, search, export** (Markdown), and share (read-only public link)
- **Auto-generated titles** from first user message
- **Delete confirmation dialog** to prevent accidental loss

### Answer Evaluation (RAGAS-Inspired)
- **Faithfulness**: decomposes answer into atomic claims, validates each against retrieved context
- **Answer Relevancy**: judges whether the answer addresses the question
- **Context Precision**: assesses which retrieved chunks were actually useful
- **LLM-as-judge**: separate evaluation model (gpt-4.1-mini) to avoid self-evaluation bias
- **Real-time**: faithfulness runs automatically after each answer; full evaluation on-demand

### Document Browser
- **Collection statistics**: document count, total chunks, storage size, file type breakdown
- **Sortable table** with filename, chunk count, upload date, and file size
- **Chunk inspector**: view individual chunks to verify chunking quality

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/mohamed-elkholy95/rag-document-qa.git
cd rag-document-qa
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | [localhost:3000](http://localhost:3000) |
| API | [localhost:8001](http://localhost:8001) |
| Swagger Docs | [localhost:8001/docs](http://localhost:8001/docs) |

### Manual Setup

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.api.main                    # API on :8001

# Frontend (separate terminal)
cd frontend && npm install
npm run dev                               # Vite on :5173
```

### Connect an LLM (optional)

The system runs without any API keys — retrieval works with local ChromaDB embeddings,
and the LLM falls back to dummy responses. To enable real answers:

```bash
# Pick one or more:
export OPENAI_API_KEY="sk-..."            # GPT-5, GPT-4.1, o-series
export ANTHROPIC_API_KEY="sk-ant-..."     # Claude Opus, Sonnet, Haiku

# Or run a local model with Ollama (no key needed):
ollama pull llama3
```

---

## API Reference

### Upload & Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and index a single document |
| `POST` | `/api/upload/batch` | Batch upload multiple files |
| `GET` | `/api/documents` | List all indexed documents |
| `GET` | `/api/documents/{doc_id}/chunks` | Inspect individual chunks |
| `DELETE` | `/api/documents/{doc_id}` | Delete document and its chunks |

### Query & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Synchronous RAG query → answer + sources |
| `WS` | `/api/chat` | Streaming chat via WebSocket |

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/conversations` | List all conversations |
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations/{id}` | Get conversation with messages |
| `PATCH` | `/api/conversations/{id}` | Rename or pin a conversation |
| `DELETE` | `/api/conversations/{id}` | Delete (cascades messages + sources) |
| `GET` | `/api/conversations/search?q=` | Search conversations |
| `GET` | `/api/conversations/{id}/export` | Export as Markdown |
| `POST` | `/api/conversations/{id}/share` | Generate a share token |
| `GET` | `/api/shared/{token}` | View shared conversation (read-only) |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/messages/{id}/evaluate` | Run 3-metric evaluation |
| `GET` | `/api/messages/{id}/evaluation` | Get stored evaluation scores |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |

---

## WebSocket Protocol

The `/api/chat` endpoint streams responses through structured JSON events:

```jsonc
// Client sends:
{ "query": "What is RAG?", "top_k": 5, "model": "gpt-5-mini", "conversation_id": "..." }

// Server streams (in order):
{ "type": "status",    "content": "Searching indexed documents..." }
{ "type": "status",    "content": "Retrieved 5 chunks across 2 files" }
{ "type": "reasoning", "content": "Let me analyze the context..." }     // CoT tokens
{ "type": "token",     "content": "Retrieval" }                         // Answer tokens
{ "type": "token",     "content": "-Augmented" }
{ "type": "token",     "content": " Generation..." }
{ "type": "done",      "sources": [...], "message_id": "...", "conversation_id": "..." }
```

---

## Project Structure

```
src/
├── api/
│   ├── main.py                  # FastAPI app with lifespan, CORS, routers
│   ├── models.py                # Pydantic v2 request/response schemas
│   ├── dependencies.py          # Dependency injection helpers
│   └── routes/
│       ├── upload.py             # File upload + validation
│       ├── query.py              # REST query + WebSocket streaming
│       ├── documents.py          # Document CRUD + chunk inspection
│       ├── conversations.py      # Conversation CRUD + search/export/share
│       └── evaluation.py         # On-demand evaluation endpoints
├── models/
│   ├── conversation.py           # Conversation table (cascade relationships)
│   ├── message.py                # Message + MessageSource tables
│   ├── document.py               # DocumentRecord metadata
│   └── evaluation.py             # MessageEvaluation scores
├── backend.py                    # RAGBackend — stateful orchestration facade
├── config.py                     # Centralized configuration constants
├── database.py                   # SQLite + SQLModel setup
├── document_loader.py            # Multi-format parser (7 file types)
├── llm_handler.py                # Multi-provider LLM adapter with streaming
├── vector_store.py               # ChromaDB wrapper (embeddings + search)
├── evaluation.py                 # RAGAS-inspired scoring (3 metrics)
└── generator.py                  # Prompt templates + context assembly

frontend/src/
├── pages/
│   ├── chat.tsx                  # Streaming chat with resizable sources panel
│   ├── upload.tsx                # Drag-and-drop upload with auto-indexing
│   ├── documents.tsx             # Document library + chunk inspector
│   └── shared.tsx                # Read-only shared conversation view
├── components/
│   ├── chat/                     # ChatThread, ChatMessage, ThinkingPanel, SourcesPanel
│   ├── upload/                   # Dropzone, FileQueue
│   ├── documents/                # DocTable, DocStats, ChunkViewer
│   └── layout/                   # AppLayout, Sidebar (model selector, top-K, stats)
├── hooks/                        # useChat, useConversations, useDocuments, useSettings
└── api/                          # Typed HTTP + WebSocket client
```

---

## Configuration

All settings are centralized in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | 5 | Number of chunks retrieved per query |
| `DEFAULT_MODEL` | `gpt-5-mini` | Default answer generation model |
| `REASONING_MODEL` | `gpt-4.1-nano` | Chain-of-thought model (lightweight) |
| `EVAL_MODEL` | `gpt-4.1-mini` | Evaluation judge model |
| `SLIDING_WINDOW_SIZE` | 5 | Conversation turns kept in context |
| `API_PORT` | 8001 | Server port |

### LLM Provider Routing

Provider is auto-detected from the model name:

| Prefix | Provider | API Key |
|--------|----------|---------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| Everything else | Ollama (local) | None needed |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ChromaDB over Qdrant/Pinecone** | Zero external services — persistent vector store that runs in-process |
| **Recursive chunking as default** | Preserves paragraph and sentence boundaries vs. naive fixed-size windows |
| **Separate reasoning model** | Cheap CoT pass (gpt-4.1-nano) avoids doubling cost on the main model |
| **Separate evaluation model** | Prevents self-evaluation bias — a different model judges the answer |
| **WebSocket streaming** | Token-by-token delivery for responsive UX vs. waiting for full response |
| **SQLite + SQLModel** | Zero-ops relational storage — conversations, messages, and sources persist across restarts |
| **React over Streamlit** | Full control over real-time streaming, layout, and component architecture |
| **Multi-provider adapter** | Swap models by name with zero code changes — same interface for all providers |
| **Idempotent document upsert** | Re-uploading overwrites by content hash — no duplicate chunks |

---

## Testing

```bash
python -m pytest tests/ -v                                    # Full test suite
python -m pytest tests/ --cov=src --cov-report=term-missing   # With coverage report
python -m pytest tests/test_evaluation.py -v                  # Single module
```

Tests use ChromaDB's `EphemeralClient` for isolated vector store testing and deterministic fixtures from `conftest.py`.

---

## Environment Variables

All optional. The system runs fully without any keys using local ChromaDB embeddings and dummy LLM responses.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI models (GPT-5, GPT-4.1, o-series) |
| `ANTHROPIC_API_KEY` | Anthropic models (Claude Opus, Sonnet, Haiku) |

---

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95)

---

<div align="center">

<sub>Built with FastAPI · React · ChromaDB · scikit-learn</sub>

</div>
