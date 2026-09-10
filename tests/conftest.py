"""
Shared pytest fixtures for the RAG Document Q&A test suite.

All fixtures use mock/in-memory data — no external dependencies required.
Uses ChromaDB EphemeralClient for vector store fixtures (no disk I/O, no cleanup needed).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.x import ...` works
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import numpy as np
import pytest

from src.domain import Chunk, Document
from src.vector_store import ChromaVectorStore

# --------------------------------------------------------------------------- #
# CI mode: stub LLM provider calls                                             #
# --------------------------------------------------------------------------- #
#
# WHY: A handful of tests (test_query_after_ingest, test_*telemetry) construct
# a real RAGBackend whose query path hits openai's HTTP API. Locally that
# works because the developer has OPENAI_API_KEY set. In CI no real API key
# is available, so we stub openai.OpenAI() to avoid the AuthenticationError
# that would otherwise abort these tests.
#
# Activated by the CI_LLM_MOCK env var so local dev behavior is unchanged.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _stub_openai_provider():
    """Replace openai.OpenAI() with an in-process stub for the whole test session.

    The stub mimics the chat.completions.create() shape used by LLMHandler,
    returning a canned response with a content attribute and an id. No network
    call is made.

    BEFORE: stubbing was opt-in via CI_LLM_MOCK=1, so a developer machine with a
            .env made real, billable provider calls and the suite failed without
            a funded key.
    AFTER:  stubbing is the default; set RAG_QA_LIVE_LLM=1 to deliberately test
            against a real provider.
    WHY:    a test suite must not depend on ambient credentials, and must never
            spend money by default. The matching root-cause fix removed the
            import-time load_dotenv() from src/llm_handler (see src/config.py).
    """
    if os.getenv("RAG_QA_LIVE_LLM", "").lower() in ("1", "true", "yes"):
        yield
        return

    try:
        import openai
    except ImportError:
        yield
        return

    from types import SimpleNamespace

    STUB_TEXT = "Stubbed answer for CI."

    def _make_stub_response():
        choice = SimpleNamespace(
            message=SimpleNamespace(content=STUB_TEXT, role="assistant"),
            finish_reason="stop",
            index=0,
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        return SimpleNamespace(
            id="chatcmpl-stub",
            choices=[choice],
            usage=usage,
            model="stub",
            created=0,
            object="chat.completion",
        )

    def _make_stub_stream_chunks():
        # Two-token stream so consumers see at least one yield then a finish.
        for piece in (STUB_TEXT, ""):
            delta = SimpleNamespace(content=piece, role="assistant")
            choice = SimpleNamespace(delta=delta, finish_reason=None, index=0)
            yield SimpleNamespace(
                id="chatcmpl-stub",
                choices=[choice],
                model="stub",
                created=0,
                object="chat.completion.chunk",
            )

    class _StubCompletions:
        def create(self, **kwargs):
            if kwargs.get("stream"):
                return _make_stub_stream_chunks()
            return _make_stub_response()

    class _StubChat:
        def __init__(self):
            self.completions = _StubCompletions()

    class _StubOpenAIClient:
        def __init__(self, *args, **kwargs):
            self.chat = _StubChat()

    original = openai.OpenAI
    openai.OpenAI = _StubOpenAIClient
    try:
        yield
    finally:
        openai.OpenAI = original


# --------------------------------------------------------------------------- #
# Isolation: never touch the developer's persistent stores                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _isolate_app_state_dirs(tmp_path_factory):
    """Redirect the FastAPI lifespan's SQLite and ChromaDB paths into tmp.

    BEFORE: any test entering `with TestClient(app)` ran the real lifespan,
            which opens `data/rag.db` and `data/chroma/` — the developer's
            actual store. Confirmed by mtime: running one route test rewrote
            `data/chroma/chroma.sqlite3`. Tests that replaced `app.state.backend`
            with a mock did so only *after* startup, so the real stores were
            already open.
    AFTER:  the module globals `src.api.main` copied from `src.config` at import
            time point into a session-scoped tmp dir, so the lifespan builds its
            own throwaway stores.
    WHY session-scoped and autouse: a test that forgets this is exactly the case
            that corrupts local data, and the failure is silent. Opting in is the
            wrong default for something whose blast radius is the user's files.
    WHY patch `src.api.main` and not `src.config`: main.py does
            `from src.config import CHROMA_PATH, SQLITE_URL`, which copies the
            values at import; rebinding the config module would not be seen.
    """
    tmp = tmp_path_factory.mktemp("app_state")

    from src.api import main as api_main

    original = (api_main.CHROMA_PATH, api_main.SQLITE_URL)
    api_main.CHROMA_PATH = str(tmp / "chroma")
    api_main.SQLITE_URL = f"sqlite:///{tmp / 'rag.db'}"
    try:
        yield tmp
    finally:
        api_main.CHROMA_PATH, api_main.SQLITE_URL = original


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

EMBEDDING_DIM = 384
SAMPLE_TEXT = (
    "Retrieval-Augmented Generation (RAG) is a technique that enhances large language "
    "models by retrieving relevant documents from an external knowledge base before "
    "generating a response. This allows the model to access up-to-date information "
    "and produce more accurate, grounded answers.\n\n"
    "The retrieval step typically uses dense vector search, where both the query and "
    "documents are embedded into a shared vector space. The most similar documents "
    "are then passed as context to the language model along with the original query."
)

SAMPLE_TEXT_2 = (
    "Vector databases store embeddings and support efficient approximate nearest-neighbour "
    "search. Popular options include Qdrant, Pinecone, Weaviate, and Chroma. "
    "They enable semantic search at scale, handling millions of vectors with low latency."
)


# --------------------------------------------------------------------------- #
# Document fixtures                                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_document() -> Document:
    """A single Document instance with realistic content."""
    return Document(
        content=SAMPLE_TEXT,
        metadata={
            "filename": "rag_overview.txt",
            "file_type": "txt",
            "file_size_bytes": len(SAMPLE_TEXT.encode()),
        },
    )


@pytest.fixture
def sample_document_2() -> Document:
    """A second Document with different content."""
    return Document(
        content=SAMPLE_TEXT_2,
        metadata={
            "filename": "vector_dbs.txt",
            "file_type": "txt",
            "file_size_bytes": len(SAMPLE_TEXT_2.encode()),
        },
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    """Pre-built chunks from the sample document."""
    texts = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models.",
        "The retrieval step uses dense vector search where documents are embedded.",
        "The most similar documents are passed as context to the language model.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunk = Chunk(
            content=text,
            metadata={
                "filename": "rag_overview.txt",
                "chunk_index": i,
                "chunk_strategy": "fixed",
            },
            doc_id=sample_document.doc_id,
        )
        chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Embedding fixtures                                                           #
# --------------------------------------------------------------------------- #


def _make_deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Create a deterministic unit-norm embedding from text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
def chroma_collection():
    """
    Ephemeral ChromaDB collection for testing.

    WHY EphemeralClient: no disk I/O, no teardown needed — each test gets
    a fresh in-memory collection that vanishes when the fixture goes out of scope.
    PATTERN: cosine distance matches how the production vector store is configured.
    """
    # WHY .open: the cosine setting lives with the store, so a fixture cannot
    # drift from production's configuration. WHY embedding_function=None: we
    # supply deterministic embeddings via upsert(), so ChromaDB must not
    # auto-embed with all-MiniLM-L6-v2.
    return ChromaVectorStore.open(
        chromadb.EphemeralClient(), "test_docs", embedding_function=None
    ).collection


@pytest.fixture
def populated_vector_store(sample_chunks: list[Chunk], chroma_collection) -> ChromaVectorStore:
    """
    A ChromaVectorStore pre-loaded with sample_chunks and deterministic embeddings.

    Replaces the old InMemoryVectorStore-based fixture now that TF-IDF has been
    removed. Tests that need a ready-to-query store use this fixture directly.
    """
    store = ChromaVectorStore(chroma_collection)
    store.upsert(
        ids=[c.chunk_id for c in sample_chunks],
        documents=[c.content for c in sample_chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "filename": c.metadata.get("filename", ""),
                "chunk_index": c.metadata.get("chunk_index", 0),
            }
            for c in sample_chunks
        ],
        embeddings=[_make_deterministic_embedding(c.content) for c in sample_chunks],
    )
    return store


# --------------------------------------------------------------------------- #
# Tmp file helper                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """A temporary .txt file with sample content."""
    file = tmp_path / "test_document.txt"
    file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return file


@pytest.fixture
def tmp_json_file(tmp_path: Path) -> Path:
    """A temporary .json file."""
    import json

    data = {"title": "Test Document", "body": SAMPLE_TEXT, "tags": ["rag", "nlp"]}
    file = tmp_path / "test_data.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    return file


@pytest.fixture
def tmp_csv_file(tmp_path: Path) -> Path:
    """A temporary .csv file."""
    import csv

    file = tmp_path / "test_data.csv"
    rows = [
        ["name", "description", "category"],
        ["RAG", "Retrieval-Augmented Generation", "NLP"],
        ["BERT", "Bidirectional Encoder Representations", "NLP"],
        ["GPT", "Generative Pre-trained Transformer", "LLM"],
    ]
    with file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    return file


# --------------------------------------------------------------------------- #
# Eval run storage                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_eval_runs(tmp_path: Path, monkeypatch) -> Path:
    """Point the eval runs directory at a temp dir for the duration of a test.

    Returns the directory itself, so a test can assert against the filesystem.

    BEFORE: three test modules each carried their own copy of this fixture, and
            every copy set EVAL_RUNS_DIR and then `importlib.reload`ed the
            storage module — because the directory was a module-level constant
            bound at import time.
    AFTER:  storage resolves the directory per call, so setting the variable is
            enough. Storage functions also take `base_dir=` for callers that
            prefer injection over an environment variable.
    """
    runs = tmp_path / "eval_runs"
    runs.mkdir()
    monkeypatch.setenv("EVAL_RUNS_DIR", str(runs))
    return runs
