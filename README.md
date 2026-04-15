<div align="center">

# RAG Document Q&A

Upload documents. Ask questions. Get answers with sources.

A full-stack **Retrieval-Augmented Generation** system built with FastAPI and React.

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br />

<img src="docs/images/chat.png" alt="Chat interface showing a LoRA question with streaming answer and source citations" width="820" />

<sub>Streaming chat with source citations and relevance scores</sub>

</div>

<br />

## What It Does

Upload PDFs, DOCX, TXT, Markdown, HTML, CSV, or JSON files. The system chunks them using recursive text splitting, builds a TF-IDF index, retrieves relevant passages via cosine similarity, and generates answers through any connected LLM -- with full source attribution.

**No API keys required to run.** The retrieval pipeline works standalone with TF-IDF embeddings. Connect an LLM (OpenAI, Anthropic, GLM, or local Ollama) for generated answers.

<br />

## Architecture

```
INDEXING     Document  ──>  Loader  ──>  Chunker  ──>  TF-IDF Embedder  ──>  Vector Store
                                          (recursive)     (scikit-learn)        (numpy)

QUERYING     Question  ──>  TF-IDF Embed  ──>  Cosine Search  ──>  Top-K Chunks  ──>  LLM  ──>  Answer
                                                                                    (streaming)
```

The **RAGBackend** facade wires all components together and persists state across requests via `app.state`, so documents indexed through `/upload` are immediately searchable via `/query`.

<br />

## Screenshots

<table>
<tr>
<td width="50%">

**Documents & Collection Stats**

<img src="docs/images/documents.png" alt="Documents page showing collection statistics and document table" width="400" />

</td>
<td width="50%">

**API (Swagger UI)**

<img src="docs/images/api.png" alt="FastAPI Swagger UI showing all API endpoints" width="400" />

</td>
</tr>
</table>

<br />

## Tech Stack

<table>
<tr>
<td><b>Frontend</b></td>
<td>React 19 &middot; TypeScript &middot; Vite &middot; Tailwind CSS &middot; shadcn/ui &middot; TanStack Query</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>Python &middot; FastAPI &middot; Uvicorn &middot; Pydantic v2</td>
</tr>
<tr>
<td><b>Retrieval</b></td>
<td>TF-IDF (scikit-learn) &middot; Cosine Similarity (numpy) &middot; In-Memory / Qdrant vector store</td>
</tr>
<tr>
<td><b>LLM</b></td>
<td>OpenAI &middot; Anthropic &middot; GLM (Zhipu AI) &middot; Ollama (local)</td>
</tr>
<tr>
<td><b>Parsing</b></td>
<td>pypdf &middot; python-docx &middot; BeautifulSoup4</td>
</tr>
<tr>
<td><b>Deploy</b></td>
<td>Docker Compose (API + nginx frontend)</td>
</tr>
</table>

<br />

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/mohamed-elkholy95/rag-document-qa.git
cd rag-document-qa
docker compose up --build
```

Frontend at `localhost:3000` -- API at `localhost:8001` -- Swagger at `localhost:8001/docs`

### Manual Setup

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.api.main              # API on :8001

# Frontend (separate terminal)
cd frontend && npm install
npm run dev                         # Vite on :5173
```

<br />

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and index a document |
| `POST` | `/api/upload/batch` | Batch upload multiple files |
| `POST` | `/api/query` | Ask a question, get answer + sources |
| `WS` | `/api/chat` | Streaming chat via WebSocket |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/documents/{id}/chunks` | Inspect document chunks |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `GET` | `/health` | Health check |

<br />

## Project Structure

```
src/
├── api/
│   ├── main.py                 # FastAPI app — lifespan, CORS, routers
│   ├── models.py               # Pydantic v2 request/response schemas
│   └── routes/
│       ├── upload.py            # File upload + validation
│       ├── query.py             # REST query + WebSocket streaming
│       └── documents.py         # Document CRUD
├── backend.py                   # RAGBackend — stateful pipeline facade
├── document_loader.py           # Multi-format loader (PDF, DOCX, HTML, CSV, JSON, TXT, MD)
├── chunker.py                   # Fixed-size + sentence-based chunking
├── embeddings.py                # TF-IDF embedder (scikit-learn)
├── vector_store.py              # Abstract store — InMemory + Qdrant backends
├── llm_handler.py               # Multi-provider LLM with streaming support
├── retriever.py                 # Cosine similarity search
├── generator.py                 # Context assembly + prompt templates
└── pipeline.py                  # Lightweight pipeline (used by tests)

frontend/src/
├── pages/                       # Chat, Upload, Documents
├── components/
│   ├── chat/                    # ChatThread, ChatInput, ChatMessage, SourcesPanel
│   ├── upload/                  # Dropzone, FileQueue
│   ├── documents/               # DocTable, DocStats, ChunkViewer
│   └── layout/                  # AppLayout, Sidebar (model selector, top-K, stats)
├── hooks/                       # useChat, useUpload, useDocuments, useSettings
└── api/                         # Typed API client + WebSocket helper
```

<br />

## Key Features

**Retrieval Pipeline**
- Three chunking strategies: fixed-size, recursive (paragraph-aware), semantic (sentence-level)
- Recursive chunking as default -- splits on `\n\n` > `\n` > `. ` > ` ` hierarchy
- TF-IDF sparse embeddings (zero model downloads, deterministic)
- Cosine similarity search with configurable top-K

**LLM Integration**
- Auto-detect provider from model name (`gpt-*` -> OpenAI, `claude-*` -> Anthropic, `glm-*` -> Zhipu, else -> Ollama)
- Token-by-token streaming over WebSocket
- Graceful fallback to dummy responses when no provider is available

**Frontend**
- Real-time streaming chat with markdown rendering
- Drag-and-drop upload with batch processing queue
- Resizable sources panel with relevance scores per chunk
- Document browser with chunk inspector
- Sidebar with model selection, top-K slider, and live collection stats

<br />

## Design Decisions

| Choice | Why |
|--------|-----|
| **TF-IDF over neural embeddings** | Zero dependencies, fast, deterministic -- good baseline before adding model complexity |
| **Recursive chunking** | Preserves paragraph/sentence structure vs. naive fixed-size windows |
| **In-memory vector store** | No infrastructure needed; Qdrant backend available when persistence matters |
| **Multi-provider LLM** | Adapter pattern -- swap models by name, zero code changes |
| **WebSocket streaming** | Token-by-token delivery for responsive UX |
| **React over Streamlit** | Full control over UX, real-time streaming, production component architecture |
| **Shared `app.state` backend** | Singleton ensures `/query` sees documents from `/ingest` |

<br />

## Testing

```bash
python -m pytest tests/ -v                                    # 24 tests
python -m pytest tests/ --cov=src --cov-report=term-missing   # with coverage
```

<br />

## Environment Variables

All optional. The system runs fully without any keys using TF-IDF retrieval + dummy LLM responses.

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI (`gpt-*` models) |
| `ANTHROPIC_API_KEY` | Anthropic (`claude-*` models) |
| `GLM_API_KEY` | Zhipu AI (`glm-*` models) |
| `QDRANT_URL` | Qdrant vector store backend |

<br />

## Author

**Mohamed Elkholy** -- [GitHub](https://github.com/mohamed-elkholy95)

---

<div align="center">
<sub>Built with Python, FastAPI, React, and scikit-learn</sub>
</div>
