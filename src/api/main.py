"""FastAPI application for the RAG Document Q&A system.

Uses lifespan to create a shared RAGBackend instance that persists across
requests.  All business logic lives in the backend; the API layer is thin.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend import RAGBackend
from src.api.routes import upload_router, query_router, documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the shared RAG backend on startup."""
    app.state.backend = RAGBackend()
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
