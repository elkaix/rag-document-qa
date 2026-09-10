"""
FastAPI application entry point for the RAG Document Q&A system.

RAG Pipeline Position:
  This is the top-level WIRING module. It creates no business logic — it:
    1. Initialises shared resources (SQLite engine, ChromaDB collection) in the
       lifespan startup hook.
    2. Injects them into a single RAGBackend instance stored on app.state.
    3. Mounts all route modules (upload, query, documents, conversations).

What concept it teaches:
  The lifespan pattern in FastAPI — using an async context manager to set up
  shared state at startup and tear it down (if needed) at shutdown. This
  replaced the deprecated @app.on_event("startup") since FastAPI 0.100+.

Why this approach over alternatives:
  - Lifespan gives explicit startup/shutdown control with proper cleanup
  - A single RAGBackend on app.state (singleton pattern) means all routes
    share the same engine, vector store, and LLM handler
  - Routes access the backend via Depends(get_backend) for clean DI

Where it fits in the RAG pipeline:
  [MAIN.PY] is the outermost layer — it assembles and exposes the system.
  Everything beneath (RAGBackend, routes, models) is imported and wired here.
"""

from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    conversations_router,
    documents_router,
    evaluation_router,
    query_router,
    upload_router,
)
from src.api.routes.eval import router as eval_router
from src.api.services.eval_runs import RunRegistry
from src.backend import RAGBackend
from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    SQLITE_URL,
    allowed_origins,
    load_env,
)
from src.database import create_db_and_tables, get_engine
from src.observability import init_observability
from src.vector_store import ChromaVectorStore

# WHY here and not inside a library: this module is the application entry point,
#      so it is the one place allowed to pull .env into the process. It runs
#      before ALLOWED_ORIGINS is read below and before the lifespan builds any
#      provider client. Importing a library must never arm real credentials.
load_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup, clean up on shutdown.

    Startup creates three things:
      1. SQLite engine + tables — persistent metadata, conversations, messages
      2. ChromaDB PersistentClient + collection — persistent vector embeddings
      3. RAGBackend — facade that wires engine + collection together

    WHY: Creating these once in the lifespan (not per-request) means:
      - Data persists across requests (ingested docs survive between /ingest and /query)
      - Data persists across restarts (SQLite file + ChromaDB directory on disk)
      - No redundant engine/client creation overhead per request

    PATTERN: Singleton state — shared across all requests so ingested data persists
    between /ingest and /query calls.
    """
    # STEP 1: SQLite — create engine and ensure all tables exist
    engine = get_engine(SQLITE_URL)
    create_db_and_tables(engine)

    # STEP 2: ChromaDB — persistent client stores embeddings to disk
    # WHY PersistentClient: Unlike EphemeralClient (used in tests), this
    #     writes to CHROMA_PATH so vectors survive process restarts.
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    # WHY ChromaVectorStore.open: the store's distance→similarity conversion is
    #     only correct in cosine space, so the store itself owns that setting
    #     rather than trusting each construction site to remember it.
    vector_store = ChromaVectorStore.open(chroma_client, CHROMA_COLLECTION)

    # STEP 3: Wire everything into the backend facade
    app.state.engine = engine
    app.state.backend = RAGBackend(engine=engine, collection=vector_store.collection)

    # STEP 4: Create the eval run registry (in-memory, thread-safe).
    # WHY: The registry tracks in-flight eval runs across requests. It must
    # be a singleton on app.state so POST /api/eval/run and GET /api/eval/runs/{id}/status
    # share the same instance — otherwise status polls would see an empty registry.
    app.state.run_registry = RunRegistry()

    # STEP 5: Initialise OpenTelemetry tracing toward Phoenix (optional).
    # WHY: Called here — after core resources are ready — so span export never
    #      blocks startup. init_observability is fail-quiet: if Phoenix is
    #      unreachable it logs a warning and traces become no-ops. Passing None
    #      (env var absent) uses the function's built-in default endpoint.
    # TRADE-OFF: We don't gate on env var presence. The function handles None
    #            correctly and doing the check here would duplicate its logic.
    # WHY no os.getenv here: init_observability resolves OTLP_ENDPOINT itself.
    #      Reading it at both sites meant two places to change one setting.
    init_observability()

    yield

    # Shutdown: no explicit cleanup needed — SQLite and ChromaDB handle
    # their own file handles. If we needed cleanup, it would go here.


app = FastAPI(
    title="RAG Document Q&A API",
    version="1.0.0",
    lifespan=lifespan,
)

# BUG FIX: CORS used to allow all origins in the same app baked into the
#          production Docker image. In dev we still want to hit the API from
#          a Vite dev server on a different port, but production should
#          restrict. The parsing rule lives in src/config.py so a
#          security-relevant setting sits with the rest of configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# PATTERN: Each router handles one resource (upload, query, documents,
#          conversations). They all share the /api prefix. Mounting them
#          here keeps main.py thin — just assembly, no logic.
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(conversations_router)
app.include_router(evaluation_router)
# PATTERN: eval router is separate from the existing evaluation_router (which handles
# per-message evaluation). This router manages the full eval harness (runs, configs, compare).
app.include_router(eval_router)


@app.get("/health")
async def health():
    """Health check endpoint — returns 200 if the server is running."""
    return {"status": "healthy"}
