"""
Centralised configuration for the RAG Document Q&A system.

RAG Pipeline Position:
  This module sits at the foundation — every other module imports from here.
  Having a single source of truth for paths, sizes, and connection strings
  means you only change a value in one place and every consumer picks it up.

What concept it teaches:
  "Config-as-code" — keep all tunable constants (chunk size, model names,
  DB paths) in one typed file rather than scattered magic numbers.

Why this approach over alternatives:
  A plain Python file is zero-dependency and easy to read.  A future step
  (see Best Practices in CLAUDE.md) can swap this for pydantic-settings
  BaseSettings to load overrides from env vars without changing callers.

Where it fits in the RAG pipeline:
  Document → [CONFIG] → Chunks → Embeddings → Vector Store → Retrieval → Answer
  CONFIG is imported by every layer, so it defines the shape of the whole
  system.
"""
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configured once here so every module that does
#   logger = logging.getLogger(__name__)
# inherits this format without repeating basicConfig calls.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# WHY: __file__ gives us an absolute path to this file no matter how the
#      process is launched (CLI, pytest, uvicorn).  Two .parent calls walk
#      up from src/ to the project root.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# WHY: DATA_DIR is intentionally outside src/ so runtime artefacts (DB,
#      vector store) never end up tracked by git next to source files.
DATA_DIR: Path = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Chunking & retrieval defaults
# ---------------------------------------------------------------------------

# PATTERN: Seed the PRNG for reproducible TF-IDF splits and test fixtures.
RANDOM_SEED: int = 42

# TRADE-OFF: 512-char chunks balance context (enough text for meaning) vs.
#            precision (small enough for specific retrieval).  Overlap of 64
#            prevents answers that straddle chunk boundaries from being lost.
# SINGLE SOURCE: production ingestion (RAGBackend) and the eval harness both
#            read these — so "baseline" eval measures production's chunking
#            (issue #16, step 4c). These are production's actual shipped values.
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

TOP_K_RESULTS: int = 5

# WHY a strategy selector: the Retriever seam (ADR 0004) makes dense, reranked,
#      hybrid, and multi-query retrieval interchangeable. Production picks one
#      here, so an eval-validated chain is promoted by config, not by a rewrite.
#      "dense" is the behaviour-preserving default; "reranked" is wired;
#      "hybrid"/"multi_query" are recognised but deferred (see build_retriever).
RETRIEVER_STRATEGY: str = "dense"

# WHY 20: the reranked strategy over-fetches this many dense candidates before
#         the cross-encoder narrows them — wide enough to give the precise
#         reranker real choice, matching the eval-tuned default.
RERANK_OVER_FETCH_N: int = 20

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

API_HOST: str = "0.0.0.0"
API_PORT: int = 8001

# ---------------------------------------------------------------------------
# SQLite / SQLModel — chat-history persistence
# ---------------------------------------------------------------------------

# WHY: Storing the DB inside DATA_DIR (not the project root) keeps the repo
#      root clean and makes the .gitignore pattern "data/rag.db" unambiguous.
SQLITE_PATH: Path = DATA_DIR / "rag.db"

# WHY: SQLModel (and SQLAlchemy underneath) expect a URL string, not a Path
#      object.  We derive it from SQLITE_PATH so the two never drift apart.
SQLITE_URL: str = f"sqlite:///{SQLITE_PATH}"

# ---------------------------------------------------------------------------
# ChromaDB — persistent vector store
# ---------------------------------------------------------------------------

# WHY: ChromaDB's PersistentClient wants a plain string path, not a Path.
#      Keeping it under DATA_DIR means one .gitignore entry covers the whole
#      data/ subtree.
CHROMA_PATH: str = str(DATA_DIR / "chroma")

# PATTERN: A single collection name constant prevents typos when the same
#          name must be used in both the ingest and query code paths.
CHROMA_COLLECTION: str = "documents"

# ---------------------------------------------------------------------------
# Chat / LLM settings
# ---------------------------------------------------------------------------

# WHY: A named default model constant lets the UI and API share the same
#      fallback without hard-coding the string in multiple places.
DEFAULT_MODEL: str = "gpt-5-mini"

# WHY a dedicated reasoning model: The chain-of-thought pass produces short,
#      throwaway scaffolding (3-5 sentences). Running it through the same
#      expensive model as the final answer doubles cost for no quality gain.
#
# WHY gpt-4.1-nano (not gpt-5-nano): The GPT-5 family are *reasoning* models
#      that consume hidden "reasoning tokens" from the completion budget
#      BEFORE emitting visible output. With a small budget (~512 tokens),
#      the hidden reasoning eats everything and the visible stream is empty
#      — which defeats the whole point of a visible CoT panel. gpt-4.1-nano
#      is a plain generation model (same $0.10/$0.40 per 1M pricing) whose
#      tokens all end up in the visible stream. Perfect for scaffolding.
REASONING_MODEL: str = "gpt-4.1-nano"

# WHY a dedicated evaluation model: Using the same model that generated the
#      answer to judge itself creates self-evaluation bias -- models are less
#      likely to flag their own hallucinations. A mid-tier model like
#      gpt-4.1-mini is cheap enough for real-time faithfulness checks while
#      strong enough to catch factual errors.
EVAL_MODEL: str = "gpt-4.1-mini"

# WHY: Sliding-window chat history keeps the LLM context window manageable.
#      5 turns (10 messages) is a practical balance: enough context for
#      follow-up questions, small enough to stay within token budgets.
SLIDING_WINDOW_SIZE: int = 5

# WHY: Truncating long document titles in the UI prevents layout overflow
#      while still giving the user enough text to identify the source.
MAX_TITLE_LENGTH: int = 60

# ---------------------------------------------------------------------------
# Runtime directory bootstrap
# ---------------------------------------------------------------------------

# WHY: We create DATA_DIR at import time so every downstream module (DB
#      initialiser, ChromaDB client) can assume the directory exists.
#      parents=True handles the case where the whole data/ tree is absent.
for _dir in [DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
