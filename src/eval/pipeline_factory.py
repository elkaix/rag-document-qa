"""
EvalPipeline factory — builds a fresh, isolated RAG pipeline from an
EvalConfig + dataset name.

Eval Harness Position:
  EvalConfig + dataset → [PIPELINE FACTORY] → EvalPipeline
                                                  ↓
                                          ingest → query → teardown
                                          (used by EvalRunner)

Design decisions:
  - Ephemeral Chroma collection per (config, dataset) so two concurrent
    runs cannot pollute each other's vectors. Random suffix on the
    collection name guards against collisions.
  - Per-stage timings via time.perf_counter() so the runner can record
    p50/p95/p99 latency at aggregation time.
  - Token counting: tiktoken if available, word-count×1.3 fallback —
    eval should not hard-fail because a tokenizer for a new model
    isn't installed.
  - Test doubles (DummyLLM) inject via *_override params; production
    uses LLMHandler(model_name).

Return type of query():
  - Returns list[SearchResult] (from vector_store.SearchResult), not
    list[Chunk]. SearchResult carries chunk_id + score which the runner
    needs for retrieval metrics. The test asserts only isinstance(chunks, list)
    so this type is compatible.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb

from src.document_loader import TextChunker
from src.eval.config import EvalConfig
from src.eval.schemas import EvalQuestion
from src.llm_handler import LLMHandler
from src.vector_store import ChromaVectorStore, SearchResult

logger = logging.getLogger(__name__)

# WHY: one-time warning flag for tiktoken fallback — we don't want the
#      warning to spam on every token-count call throughout an eval run.
_tiktoken_warned = False

try:
    import tiktoken as _tiktoken  # type: ignore
except ImportError:
    _tiktoken = None  # type: ignore


def _count_tokens(text: str, model: str) -> int:
    """Count tokens in text, falling back to word-count * 1.3 if tiktoken fails.

    WHY the fallback: tiktoken doesn't know every model (new OpenAI releases
    ship before tiktoken is updated). Eval should not hard-fail on a missing
    tokenizer — a ±30% estimate is fine for cost/latency tracking.

    Args:
        text: The text to count tokens for.
        model: Model name used to select the tiktoken encoding.

    Returns:
        Estimated token count (int).
    """
    global _tiktoken_warned
    if _tiktoken is not None:
        try:
            enc = _tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # Unknown model for tiktoken — fall through to word estimate
            pass

    if not _tiktoken_warned:
        logger.warning(
            "tiktoken not installed or model %r unknown — "
            "using word-count × 1.3 for token estimates.",
            model,
        )
        _tiktoken_warned = True
    return int(len(text.split()) * 1.3)


# --------------------------------------------------------------------------- #
# EvalPipeline                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class EvalPipeline:
    """An isolated, ephemeral RAG pipeline for one (config, dataset) eval run.

    Teaches: the factory pattern — consumers never call __init__ directly;
    they use build_pipeline() which sets up all components and wires them
    into a coherent, ready-to-use pipeline.

    Why ephemeral Chroma: each pipeline gets its own collection with a
    random suffix, so parallel eval runs cannot pollute each other's vectors.
    teardown() deletes the collection after the run completes.
    """

    chunker: TextChunker
    vector_store: ChromaVectorStore
    llm: Any  # LLMHandler or a test double with .generate(prompt, system_prompt)
    judge_llm: Any  # Same contract as llm
    config: EvalConfig
    dataset_name: str

    # Private: needed for teardown() — ChromaVectorStore doesn't own the client.
    # WHY not reach into vector_store._collection._client: that would couple us
    # to ChromaDB internals that could change. Own the client reference here.
    _client: chromadb.ClientAPI = field(repr=False, default=None)  # type: ignore[assignment]
    _collection_name: str = field(repr=False, default="")

    def ingest(self, questions: list[EvalQuestion]) -> None:
        """Upsert question contexts into the vector store.

        For squad_v2_dev_200:
          Each question carries its context in metadata["context"]. The context
          IS the chunk — SQuAD is designed so each question has exactly one
          supporting passage. question.id serves as the Chroma document ID,
          which makes gold_chunk_id lookup trivial in the retrieval metric.

        For ml_papers_v1:
          Loads corpus_manifest.json, reads each pinned PDF via DocumentLoader,
          chunks it with this pipeline's TextChunker, and upserts the chunks.
          If the manifest is missing or empty, ingest is a no-op (logged).

        Args:
            questions: Gold-labeled questions from the dataset loader.
        """
        if self.dataset_name == "squad_v2_dev_200":
            self._ingest_squad(questions)
        elif self.dataset_name == "ml_papers_v1":
            self._ingest_ml_papers()
        else:
            logger.warning(
                "Unknown dataset %r — ingest is a no-op.", self.dataset_name
            )

    def _ingest_squad(self, questions: list[EvalQuestion]) -> None:
        """Upsert each question's context as one Chroma document.

        PATTERN: question.id == chunk_id == gold_chunk_id. This alignment
        means the retrieval metric can check retrieved IDs directly against
        EvalQuestion.gold_chunk_ids without any translation layer.
        """
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for q in questions:
            ctx = q.metadata.get("context", "")
            if not ctx:
                logger.warning("SQuAD question %s has no context — skipping.", q.id)
                continue
            ids.append(q.id)
            documents.append(ctx)
            # WHY include doc_id matching chunk_id: ChromaVectorStore.delete_by_doc_id
            # filters on metadata["doc_id"]. We don't use deletion in eval, but
            # keeping the schema consistent with production reduces surprises.
            metadatas.append({"doc_id": q.id, "question_id": q.id})

        if ids:
            self.vector_store.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("Ingested %d SQuAD contexts into '%s'.", len(ids), self._collection_name)

    def _ingest_ml_papers(self) -> None:
        """Load, chunk, and upsert PDFs listed in corpus_manifest.json.

        WHY graceful no-op on missing manifest: the ML Papers dataset ships
        with an empty manifest skeleton so tests pass before any PDFs are
        labeled. A missing manifest is not an error for the eval harness —
        it means no papers have been added yet.
        """
        import json
        from pathlib import Path

        from src.document_loader import DocumentLoader

        manifest_path = Path("eval_data/ml_papers_v1/corpus_manifest.json")
        if not manifest_path.exists():
            logger.info("ML Papers manifest not found at %s — ingest is a no-op.", manifest_path)
            return

        with manifest_path.open() as f:
            manifest = json.load(f)

        papers = manifest.get("papers", [])
        if not papers:
            logger.info("ML Papers manifest has no papers — ingest is a no-op.")
            return

        loader = DocumentLoader()
        for paper in papers:
            local_path = Path(paper["local_path"])
            try:
                doc = loader.load(local_path)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping paper %s: %s", paper.get("id"), exc)
                continue

            chunks = self.chunker.chunk(doc)
            if not chunks:
                continue

            self.vector_store.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[{"doc_id": c.doc_id, "paper_id": paper.get("id", "")} for c in chunks],
            )
            logger.info(
                "Ingested paper %s: %d chunks.", paper.get("id"), len(chunks)
            )

    def query(self, question: str) -> tuple[list[SearchResult], str, dict]:
        """Retrieve relevant chunks and generate an answer with timing + cost telemetry.

        Pipeline step: QUERYING — embed question → cosine search → build
        context → generate answer → measure tokens + cost.

        Args:
            question: Natural language question from the eval set.

        Returns:
            Tuple of:
              - list[SearchResult]: Retrieved chunks ordered by descending similarity.
              - str: Generated answer.
              - dict: Telemetry with keys:
                  timings_ms: {"retrieve": float, "generate": float}
                  tokens:     {"prompt": int, "completion": int}
                  cost_usd:   float
        """
        from src.eval import pricing

        top_k = self.config.pipeline.retriever.top_k

        # ---- RETRIEVAL ---------------------------------------------------------
        t0 = time.perf_counter()
        results = self.vector_store.query(query_text=question, top_k=top_k)
        t1 = time.perf_counter()
        retrieve_ms = (t1 - t0) * 1000.0

        # ---- CONTEXT ASSEMBLY --------------------------------------------------
        context = "\n\n".join(r.content for r in results)

        system_prompt = (
            "You are a helpful assistant. Answer the question based solely on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # WHY count both: the LLM sees system_prompt + user_prompt as prompt tokens.
        full_prompt_text = (system_prompt + "\n" + user_prompt) if system_prompt else user_prompt
        model = self.config.pipeline.generator.model

        # ---- GENERATION --------------------------------------------------------
        t2 = time.perf_counter()
        answer = self.llm.generate(user_prompt, system_prompt=system_prompt)
        t3 = time.perf_counter()
        generate_ms = (t3 - t2) * 1000.0

        # ---- TOKEN COUNTING ----------------------------------------------------
        prompt_tokens = _count_tokens(full_prompt_text, model)
        completion_tokens = _count_tokens(answer, model)

        # ---- COST ESTIMATION ---------------------------------------------------
        cost = pricing.cost_usd(model, prompt_tokens, completion_tokens)

        telemetry = {
            "timings_ms": {"retrieve": retrieve_ms, "generate": generate_ms},
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
            "cost_usd": cost,
        }
        return results, answer, telemetry

    def teardown(self) -> None:
        """Delete the ephemeral Chroma collection and release the client reference.

        WHY idempotent: the test suite may call teardown() in a finally block
        after the collection was already deleted explicitly. A double-call must
        be a silent no-op, not an exception.
        """
        if self._client is None or not self._collection_name:
            return  # already torn down

        try:
            self._client.delete_collection(self._collection_name)
            logger.debug("Deleted Chroma collection '%s'.", self._collection_name)
        except Exception as exc:
            # Not-found is the common idempotency case; log and continue.
            logger.debug("teardown: delete_collection raised (already gone?): %s", exc)
        finally:
            # Null out so a subsequent call is a no-op (idempotency guarantee).
            self._client = None  # type: ignore[assignment]
            self._collection_name = ""


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_pipeline(
    config: EvalConfig,
    dataset_name: str,
    llm_override: object | None = None,
    judge_llm_override: object | None = None,
) -> EvalPipeline:
    """Construct a fresh, isolated EvalPipeline for a (config, dataset) pair.

    Teaches: the factory function pattern — all wiring happens here so callers
    receive a ready-to-use object with no boilerplate. Each call produces an
    independent Chroma collection so concurrent runs don't share state.

    Args:
        config: Validated EvalConfig specifying chunker, retriever, generator,
                and eval parameters.
        dataset_name: Name key identifying which dataset the pipeline handles
                      (e.g. "squad_v2_dev_200", "ml_papers_v1").
        llm_override: If provided, use this object as the answer LLM instead
                      of building an LLMHandler. Primarily for test doubles.
        judge_llm_override: If provided, use this object as the judge LLM.

    Returns:
        A configured EvalPipeline ready for ingest() → query() → teardown().
    """
    # ---- Chunker ---------------------------------------------------------------
    chunker_cfg = config.pipeline.chunker
    chunker = TextChunker(
        chunk_size=chunker_cfg.chunk_size,
        chunk_overlap=chunker_cfg.chunk_overlap,
        strategy=chunker_cfg.strategy,
    )

    # ---- Chroma collection (ephemeral — lives only for this pipeline) ----------
    # WHY EphemeralClient: no disk I/O, no port, no cleanup needed beyond
    # client.delete_collection(). Perfectly isolated per build_pipeline() call.
    # WHY random suffix: prevents name collisions if two pipelines with the
    # same config+dataset names are built in the same process.
    # NOTE: First call auto-downloads all-MiniLM-L6-v2 ONNX (~80MB) if not cached.
    collection_name = f"eval_{config.name}_{dataset_name}_{uuid.uuid4().hex[:6]}"
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=collection_name,
        # WHY cosine: ChromaVectorStore converts distance→similarity via
        # score = max(0, 1 - distance). This only makes sense in cosine space
        # where distance ∈ [0, 2] and identical vectors have distance 0.
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(collection=collection)

    # ---- LLM handlers ----------------------------------------------------------
    llm = llm_override if llm_override is not None else LLMHandler(config.pipeline.generator.model)
    judge_llm = (
        judge_llm_override if judge_llm_override is not None
        else LLMHandler(config.eval.judge_model)
    )

    return EvalPipeline(
        chunker=chunker,
        vector_store=vector_store,
        llm=llm,
        judge_llm=judge_llm,
        config=config,
        dataset_name=dataset_name,
        _client=client,
        _collection_name=collection_name,
    )
