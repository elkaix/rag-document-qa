<div align="center">

# RAG Document Q&A

**Retrieval-Augmented Generation** for intelligent document question answering

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-24%20passed-success?style=flat-square)](#testing)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

## Overview

A full-stack **RAG (Retrieval-Augmented Generation)** system that answers questions over document collections. Upload PDFs, DOCX, TXT, Markdown, HTML, CSV, or JSON files — the system chunks them, builds a TF-IDF index, retrieves relevant passages via cosine similarity, and generates answers through any connected LLM.

Built as a portfolio project to demonstrate end-to-end RAG pipeline design, from document parsing through to a production-style React frontend with streaming chat.

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                  React Frontend                  │
                        │       (Chat · Upload · Documents pages)          │
                        └────────────┬──────────────────┬──────────────────┘
                                REST │                  │ WebSocket
                                     ▼                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │                  FastAPI Backend                  │
                        │           /api/upload · /api/query · /api/chat   │
                        └────────────────────────┬─────────────────────────┘
                                                 │
                        ┌────────────────────────▼─────────────────────────┐
                        │                   RAGBackend                      │
                        │         (stateful facade, shared via app.state)   │
                        └───┬──────────┬──────────┬──────────┬─────────────┘
                            │          │          │          │
                     ┌──────▼──┐ ┌─────▼────┐ ┌──▼───┐ ┌───▼────────┐
                     │Document │ │  Text    │ │TF-IDF│ │   Vector   │
                     │ Loader  │ │ Chunker  │ │Embed.│ │   Store    │
                     └─────────┘ └──────────┘ └──────┘ └────────────┘
                                                              │
                                                       ┌──────▼──────┐
                                                       │ LLM Handler │
                                                       │ (multi-     │
                                                       │  provider)  │
                                                       └─────────────┘
```

**Two data flows through the pipeline:**

| Flow | Path |
|------|------|
| **Indexing** | Document &rarr; Loader &rarr; Chunker &rarr; TF-IDF Embedder &rarr; Vector Store |
| **Querying** | Question &rarr; TF-IDF Embedder &rarr; Cosine Search &rarr; Top-K Chunks &rarr; LLM &rarr; Answer |

## Features

**Backend**
- Multi-format document loading — PDF, DOCX, TXT, Markdown, HTML, CSV, JSON
- Three chunking strategies — fixed-size, recursive (paragraph-aware), and semantic (sentence-level)
- TF-IDF sparse embeddings with scikit-learn (zero model downloads)
- In-memory vector store with cosine similarity search (Qdrant backend available)
- Multi-provider LLM support — OpenAI, Anthropic, GLM (Zhipu), Ollama (local)
- Streaming responses via WebSocket
- Graceful fallback when no LLM is configured

**Frontend**
- React 19 SPA with TypeScript, Tailwind CSS, and shadcn/ui components
- Real-time streaming chat with WebSocket connection
- Drag-and-drop file upload with batch processing
- Document browser with chunk inspection
- Resizable sources panel showing retrieved passages with relevance scores
- Sidebar with model selection and top-K configuration
- Collection stats (documents, chunks, size, file types)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query |
| Backend | Python 3.14, FastAPI, Uvicorn, Pydantic v2 |
| NLP | TF-IDF (scikit-learn), cosine similarity (numpy) |
| LLM | OpenAI SDK, Anthropic SDK, Ollama, GLM/Zhipu AI |
| Document Parsing | pypdf, python-docx, BeautifulSoup4 |
| Testing | pytest, httpx |
| Deployment | Docker Compose (API + nginx frontend) |

## Quick Start

### Prerequisites

- Python 3.12+ and Node.js 22+
- (Optional) An LLM API key or local Ollama server — the system works without one using dummy responses

### Backend

```bash
git clone https://github.com/mohamed-elkholy95/rag-document-qa.git
cd rag-document-qa

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the API server (port 8001, Swagger UI at /docs)
python -m src.api.main
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # Vite dev server on port 5173
```

### Docker Compose

```bash
docker compose up --build
# Frontend on :3000, API on :8001
```

## Project Structure

```
rag-document-qa/
├── src/                          # Python backend
│   ├── api/
│   │   ├── main.py               # FastAPI app with lifespan, CORS, routers
│   │   ├── models.py             # Pydantic v2 request/response schemas
│   │   └── routes/
│   │       ├── upload.py         # POST /api/upload, /api/upload/batch
│   │       ├── query.py          # POST /api/query, WS /api/chat
│   │       └── documents.py      # GET/DELETE /api/documents
│   ├── backend.py                # RAGBackend — stateful pipeline facade
│   ├── document_loader.py        # Multi-format loader + 3 chunking strategies
│   ├── embeddings.py             # TF-IDF embedder (scikit-learn)
│   ├── vector_store.py           # Abstract store + InMemory/Qdrant backends
│   ├── llm_handler.py            # Multi-provider LLM with streaming
│   ├── retriever.py              # Cosine similarity retriever
│   ├── generator.py              # Context assembly + prompt template
│   ├── pipeline.py               # Lightweight RAG pipeline (used by tests)
│   ├── evaluation.py             # Retrieval quality metrics
│   ├── config.py                 # Constants and logging setup
│   └── utils.py                  # Shared utilities
├── frontend/                     # React SPA
│   ├── src/
│   │   ├── pages/                # Chat, Upload, Documents
│   │   ├── components/
│   │   │   ├── chat/             # ChatThread, ChatInput, ChatMessage, SourcesPanel
│   │   │   ├── upload/           # Dropzone, FileQueue
│   │   │   ├── documents/        # DocTable, DocStats, ChunkViewer
│   │   │   ├── layout/           # AppLayout, Sidebar
│   │   │   └── ui/               # shadcn/ui primitives
│   │   ├── hooks/                # useChat, useUpload, useDocuments, useSettings
│   │   └── api/                  # API client + TypeScript types
│   ├── Dockerfile                # Multi-stage build → nginx
│   └── package.json
├── tests/                        # pytest suite (24 tests)
│   ├── conftest.py               # Fixtures + deterministic embeddings
│   └── test_document_loader.py   # Loader, chunker, and validation tests
├── docker-compose.yml            # API + frontend services
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/upload` | Upload and index a single document |
| `POST` | `/api/upload/batch` | Upload multiple documents at once |
| `POST` | `/api/query` | Ask a question, get an answer with sources |
| `WS` | `/api/chat` | Streaming chat over WebSocket |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document and its chunks |
| `GET` | `/api/documents/{doc_id}/chunks` | Inspect chunks for a document |

Interactive API docs available at `/docs` (Swagger UI) when the server is running.

## Frontend Pages

| Page | Description |
|------|-------------|
| **Chat** | Streaming Q&A interface with a resizable sources panel showing retrieved chunks and relevance scores |
| **Upload** | Drag-and-drop file upload with a processing queue, batch upload support, and per-file status tracking |
| **Documents** | Collection overview with stats cards, document table, and a chunk viewer for inspecting how documents were split |

## LLM Providers

The system auto-detects the provider from the model name:

| Prefix | Provider | Required |
|--------|----------|----------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `glm-*` | GLM / Zhipu AI | `GLM_API_KEY` |
| Everything else | Ollama (local) | Ollama server running |

No API keys are required to run the system — without them, a dummy response is returned with the assembled context, which is useful for testing the retrieval pipeline in isolation.

## Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `OPENAI_API_KEY` | Only for OpenAI models | `LLMHandler` |
| `ANTHROPIC_API_KEY` | Only for Anthropic models | `LLMHandler` |
| `GLM_API_KEY` | Only for GLM models | `LLMHandler` |
| `GLM_BASE_URL` | Optional GLM endpoint override | `LLMHandler` |
| `QDRANT_URL` | Only for Qdrant vector backend | `QdrantVectorStore` |
| `VITE_API_URL` | Optional API base URL for frontend | Frontend API client |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test class
python -m pytest tests/test_document_loader.py::TestTextChunkerFixed -v
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **TF-IDF over neural embeddings** | Zero model downloads, fast, deterministic. Good baseline that demonstrates the retrieval concept before adding complexity. |
| **Recursive chunking as default** | Splits on paragraph &rarr; sentence &rarr; word boundaries, preserving document structure better than fixed-size windows. |
| **In-memory vector store** | No external services required. Qdrant backend available for persistence. |
| **Multi-provider LLM handler** | Demonstrates adapter pattern — swap providers by changing the model name, no code changes. |
| **WebSocket streaming** | Token-by-token delivery for responsive chat UX. Falls back to REST for non-streaming queries. |
| **React + Vite over Streamlit** | Production-grade frontend with full control over UX, real-time streaming, and component architecture. |
| **Shared backend via `app.state`** | Singleton pattern — pipeline state persists across requests so `/query` can find documents from `/upload`. |

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95)

---

<div align="center">
<sub>Built with Python, FastAPI, React, and scikit-learn</sub>
</div>
