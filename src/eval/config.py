"""
EvalConfig — typed pipeline-and-eval configuration loaded from YAML.

Eval Harness Position:
  configs/eval/*.yaml → load_config() → EvalConfig → EvalRunner.run()

Design decisions:
  - Pydantic v2 with nested models (ChunkerCfg, RetrieverCfg, etc.) so
    each subsystem owns its own validation surface.
  - Literal["..."] on dataset names and chunker strategy gives schema-
    level rejection of typos before the runner spins up.
  - YAML over JSON for human authorability — eval configs are written
    by hand, not generated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ChunkerCfg(BaseModel):
    """Chunking strategy configuration.

    Teaches: how chunking parameters propagate from config to the pipeline.
    Why Literal: rejects typos ("recusive") at validation time, not at runtime.
    Pipeline position: INDEXING step — Document → [Chunker] → Embeddings.
    """

    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrieverCfg(BaseModel):
    """Retriever configuration.

    Teaches: separating retrieval hyperparameters from the retriever implementation.
    Pipeline position: QUERYING step — Embeddings → [Retriever] → Top-K chunks.
    """

    top_k: int = 5


class GeneratorCfg(BaseModel):
    """LLM generator configuration.

    Teaches: optional reasoning model pattern — some providers support a
    lightweight "thinking" model before the final answer model.
    Pipeline position: QUERYING step — Chunks → [Generator] → Answer.
    """

    model: str = "gpt-5-mini"
    # WHY: reasoning_model is optional — None disables the CoT pre-pass.
    reasoning_model: str | None = "gpt-4.1-nano"


class EmbedderCfg(BaseModel):
    """Embedder selection. None/default = ChromaDB built-in ONNX (current behavior).

    Phase 2 lever 2b: swap ChromaDB's default MiniLM (384-dim) for BAAI/bge-small-en-v1.5
    (also 384-dim) which is domain-tuned for retrieval. The factory wires this as a
    Chroma EmbeddingFunction at collection-creation time, so ChromaVectorStore.upsert/query
    auto-embed without per-call code changes.
    """

    model_config = ConfigDict(extra="forbid")
    name: Literal["chroma_default", "bge_small_en_v1_5"] = "chroma_default"


class HybridCfg(BaseModel):
    """BM25 + dense retrieval with Reciprocal Rank Fusion.

    Phase 2 lever 2c: combine sparse (BM25) and dense (vector) signal. Disabled by default;
    when enabled, the retriever fetches top-N candidates from each side and fuses them
    with RRF: score(d) = sum over r in {dense, sparse} of 1 / (rrf_k + rank_r(d)).
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60


class RerankerCfg(BaseModel):
    """Cross-encoder rerank top-N → final-K. None = no rerank (current behavior).

    Phase 2 lever 2d: improve precision by re-scoring the top-N retrieved candidates
    with a dedicated relevance model (ms-marco-MiniLM-L-6-v2). Adds latency but no
    LLM cost.
    """

    model_config = ConfigDict(extra="forbid")
    model: Literal["ms_marco_minilm_l6_v2"] | None = None
    rerank_top_n: int = 20
    final_top_k: int = 5


class QueryRewriterCfg(BaseModel):
    """LLM-based query expansion. None = no rewrite (current behavior).

    Phase 2 lever 2e: ask an LLM to produce up to N alternative phrasings of the user
    query, retrieve against each, then deduplicate. Costs one LLM call per question;
    captured in the cost ledger under the 'rewriter' bucket.
    """

    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    max_expansions: int = 3


class RefusalHandlerCfg(BaseModel):
    """Answerability gate. enabled=False = current behavior.

    Phase 2 lever 2g: when the top-1 retrieval similarity falls below `similarity_threshold`,
    short-circuit to `no_answer_text` instead of calling the generator. This trades
    answer_correctness on borderline-answerable questions for refusal_correctness on
    truly unanswerable ones — exactly the trade-off the SQuAD v2 dev set surfaces.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    similarity_threshold: float = 0.35
    no_answer_text: str = "I don't have enough information to answer that."


class PipelineCfg(BaseModel):
    """Aggregates all pipeline-level sub-configs into one validated structure.

    Phase 2 additions are all default-off so existing baseline configs keep validating
    unchanged. Each new block is a documented lever; see configs/eval/phase2/*.yaml
    for tier-by-tier toggles.
    """

    chunker: ChunkerCfg
    embedder: EmbedderCfg = Field(default_factory=EmbedderCfg)
    retriever: RetrieverCfg
    hybrid: HybridCfg = Field(default_factory=HybridCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)
    query_rewriter: QueryRewriterCfg = Field(default_factory=QueryRewriterCfg)
    generator: GeneratorCfg
    refusal_handler: RefusalHandlerCfg = Field(default_factory=RefusalHandlerCfg)


class EvalCfg(BaseModel):
    """Evaluation harness parameters.

    Teaches: statistical evaluation design — bootstrap CI and permutation
    tests are the two workhorses for comparing RAG pipeline variants.
    Why seed: reproducibility across runs and machines.
    """

    datasets: list[Literal["squad_v2_dev_200", "ml_papers_v1"]]
    judge_model: str = "gpt-4.1-mini"
    bootstrap_n: int = 1000
    permutation_n: int = 10000
    seed: int = 42
    # Phase 2: hard per-run spend ceiling. EvalRunner aborts when the cumulative
    # generator + judge + rewriter cost crosses this. None disables the guard.
    spend_ceiling_usd: float | None = None


class EvalConfig(BaseModel):
    """Root configuration model for one named evaluation run.

    Each YAML file in configs/eval/ maps 1:1 to one EvalConfig instance.
    The name field doubles as a human-readable label in eval reports.
    """

    name: str
    description: str = ""
    pipeline: PipelineCfg
    # TRADE-OFF: `eval` shadows the Python builtin, but as a field name on a
    # Pydantic model it is unambiguous — accessed as cfg.eval.datasets, never
    # called as a function. Kept for schema clarity over renaming.
    eval: EvalCfg


def load_config(path: Path) -> EvalConfig:
    """Load a YAML config file, validate it, and return an EvalConfig.

    Args:
        path: Filesystem path to a YAML config file.

    Returns:
        Validated EvalConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If the YAML content fails schema validation.
    """
    # WHY: Raise FileNotFoundError explicitly rather than letting yaml.safe_load
    # raise a less informative OSError. Callers can distinguish "wrong path"
    # from "bad content" without inspecting exception types.
    if not path.exists():
        raise FileNotFoundError(f"Eval config not found: {path}")

    raw = yaml.safe_load(path.read_text())
    return EvalConfig.model_validate(raw)
