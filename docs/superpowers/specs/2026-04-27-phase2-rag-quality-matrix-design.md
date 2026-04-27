# Phase 2 — RAG Quality Matrix (Design Spec)

> **Status:** approved 2026-04-27. Pending implementation plan via `superpowers:writing-plans`.
> **Author:** Mohamed Elkholy
> **Predecessor:** [`2026-04-26-rag-eval-harness-phase-1-design.md`](./2026-04-26-rag-eval-harness-phase-1-design.md) (Phase 1, eval harness)
> **Eval baseline being targeted:** `eval_runs/2026-04-27_191733_baseline_squad_only_c925492` — 200 SQuAD v2 dev questions, real OpenAI API.

---

## 1. Goal + Scope

Phase 2 produces a layered ablation matrix that measures the lift each major RAG architectural lever buys on top of the Phase 1 baseline, then ships a portfolio-grade results document attributing the lift to mechanism.

### 1.1 Goal

Run 10 evals (1 baseline re-run + 6 single-lever tiers + 3 model-sweep variants under tier 2f) through the existing Phase 1 harness, with each tier inheriting from the previous best so adjacent comparisons attribute lift to one mechanism at a time. Write findings.

### 1.2 In scope

- 10 eval YAML configs under `configs/eval/phase2/`, each runnable through `python -m src.eval.cli run`.
- 6 new pipeline modules wired into `EvalPipeline` factory (not into prod `RAGBackend`):
  `BgeEmbedder`, `BM25HybridRetriever`, `CrossEncoderReranker`, `QueryRewriter`, `RefusalHandler`. (`SemanticChunker` already exists.)
- Schema extension to `EvalConfig.pipeline` with 5 new sub-config blocks, all backward-compatible (defaults match current behavior).
- `docs/PHASE2_RESULTS.md`: methodology, per-tier metric chart, paired-significance table, "winning stack" recipe, ≥ 1 sentence finding per lever.

### 1.3 Out of scope

- Wiring any new module into the production `RAGBackend` / live `/chat` path. Phase 2 is measurement only.
- Adding `ml_papers_v1` to the matrix (deferred to Phase 3 once labeled).
- Auto-promotion of the winning stack to runtime defaults (Phase 2.5).

### 1.4 Success criteria

- All 10 runs land on disk, visible in `/eval`, pairwise compare reports render.
- `PHASE2_RESULTS.md` ships with the chart + significance table + per-tier finding.
- Total API spend ≤ $5 (hard ceiling, no rollback).

### 1.5 Branching

Phase 2 PR-A is stacked on `feature/eval-harness-1d` (current Phase 1 PR #4 tip). Once Phase 1 merges, PR-A retargets to `main`.

---

## 2. Architecture

The existing `EvalPipeline` factory in `src/eval/runner/pipeline_factory.py` already takes an `EvalConfig` and assembles a runtime pipeline. Phase 2 extends the factory's switch statements; it does not introduce a new orchestrator.

### 2.1 Module map (6 new files, all under `src/`)

```
src/
├── document_loader.py             # existing — SemanticChunker already exists at line 470
├── eval/
│   ├── retrievers/                # NEW package
│   │   ├── __init__.py
│   │   ├── bm25_hybrid.py         # BM25HybridRetriever (RRF fusion)
│   │   └── reranker.py            # CrossEncoderReranker (ms-marco-MiniLM)
│   ├── embedders/                 # NEW package
│   │   ├── __init__.py
│   │   └── bge_small.py           # BgeEmbedder wrapping bge-small-en-v1.5
│   ├── transforms/                # NEW package
│   │   ├── __init__.py
│   │   ├── query_rewriter.py      # LLM-based query expansion
│   │   └── refusal_handler.py     # answerability gate + no-answer prompt
│   └── runner/
│       └── pipeline_factory.py    # existing — extended with new branches
```

Every new module is single-responsibility and budgeted under the 250-line CLAUDE.md ceiling.

### 2.2 Lever-by-lever wiring

| Tier | Module | Plug-in point in `EvalPipeline` | Notes |
|------|--------|---------------------------------|-------|
| 2a `chunking_semantic` | extends existing `document_loader.SemanticChunker` | chunker switch (no new code) | Sentence-aware accumulation with sentence overlap. |
| 2b `embedder_bge` | `eval/embedders/bge_small.py` | embedder switch in factory | Uses `sentence-transformers` `BAAI/bge-small-en-v1.5` (33M params, on-device). |
| 2c `hybrid_bm25` | `eval/retrievers/bm25_hybrid.py` | retriever switch | Wraps `rank_bm25` + ChromaDB; RRF fusion on top-20 from each side. |
| 2d `rerank_crossenc` | `eval/retrievers/reranker.py` | post-retrieval hook | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-20 → top-5. |
| 2e `query_rewrite` | `eval/transforms/query_rewriter.py` | pre-retrieval hook | One LLM call per query; 1–3 expansions, dedup-merged. Uses `gpt-4.1-nano`. |
| 2f `answer_model` | no new module | generator switch (existing) | YAML-only — sweeps `gpt-5-mini`, `gpt-4.1-mini`, `claude-haiku-4-5`. |
| 2g `refusal_handler` | `eval/transforms/refusal_handler.py` | post-retrieval gate | Threshold on top-1 similarity; if low, short-circuit to no-answer text. |

### 2.3 Data flow per query (final tier 2g)

```
query
  └─► QueryRewriter ─► {q, q', q''}              (2e)
        └─► HybridRetriever (BM25 + BGE)          (2b, 2c)
              └─► CrossEncoderReranker top-5      (2d)
                    └─► RefusalHandler            (2g)
                          ├─► low conf ─► "I don't know"
                          └─► high conf ─► Generator (gpt-5-mini)  (2f)
                                            └─► answer + telemetry
```

Each tier's config disables levers introduced after it (e.g., tier 2c sets `reranker.model: null`, `query_rewriter.model: null`).

### 2.4 Dependency additions

| Tier needing it | Dep | Size |
|------|-----|------|
| 2a | `nltk` (sentence tokenization, already pulled by project deps — verify in PR) | 0 |
| 2b | `sentence-transformers` already in deps (used by judge embedder) | 0 |
| 2c | `rank-bm25` | tiny, pure-Python |
| 2d | `sentence-transformers` cross-encoder model — pulled at runtime | model ~80MB |
| 2e | none — uses existing `LLMHandler` | 0 |
| 2f | none | 0 |
| 2g | none — pure logic | 0 |

Net new third-party deps: **just `rank-bm25`**.

### 2.5 File-size guard

The only existing file at risk of crossing the 250-line ceiling is `src/eval/runner/pipeline_factory.py`. If it crosses, split into:
- `pipeline_factory.py` — orchestration (entry point + composition)
- `factory_levers.py` — per-lever construction helpers (one builder function per lever)

---

## 3. Config Schema + Factory Contract

### 3.1 Schema extensions (additive, backward-compatible)

All new fields default to "off / current behavior" so `baseline.yaml` and `baseline_squad_only.yaml` keep validating unchanged.

```python
# src/eval/config.py — extending PipelineCfg

class EmbedderCfg(BaseModel):
    """NEW — embedder selection. None = ChromaDB default ONNX (current behavior)."""
    model_config = ConfigDict(extra="forbid")
    name: Literal["chroma_default", "bge_small_en_v1_5"] = "chroma_default"

class HybridCfg(BaseModel):
    """NEW — BM25 + dense fusion. enabled=False is the default (pure dense)."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    bm25_top_k: int = 20            # candidates from sparse side
    dense_top_k: int = 20           # candidates from dense side
    rrf_k: int = 60                 # standard RRF constant

class RerankerCfg(BaseModel):
    """NEW — cross-encoder rerank top-N → top-K. None = no rerank."""
    model_config = ConfigDict(extra="forbid")
    model: Literal["ms_marco_minilm_l6_v2"] | None = None
    rerank_top_n: int = 20
    final_top_k: int = 5

class QueryRewriterCfg(BaseModel):
    """NEW — LLM query expansion. None = no rewrite."""
    model_config = ConfigDict(extra="forbid")
    model: str | None = None        # e.g. "gpt-4.1-nano"
    max_expansions: int = 3

class RefusalHandlerCfg(BaseModel):
    """NEW — answerability gate. enabled=False = current behavior."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    similarity_threshold: float = 0.35
    no_answer_text: str = "I don't have enough information to answer that."

# Extends PipelineCfg
class PipelineCfg(BaseModel):
    chunker: ChunkerCfg
    embedder: EmbedderCfg = Field(default_factory=EmbedderCfg)         # NEW
    retriever: RetrieverCfg
    hybrid: HybridCfg = Field(default_factory=HybridCfg)               # NEW
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)         # NEW
    query_rewriter: QueryRewriterCfg = Field(default_factory=QueryRewriterCfg)  # NEW
    generator: GeneratorCfg
    refusal_handler: RefusalHandlerCfg = Field(default_factory=RefusalHandlerCfg)  # NEW
```

`extra="forbid"` rejects typos at YAML load time; backward-compat is enforced by a test that loads existing baseline configs without modification.

### 3.2 Factory contract

```python
# src/eval/runner/pipeline_factory.py — extended

def build_eval_pipeline(cfg: EvalConfig) -> EvalPipeline:
    chunker  = _build_chunker(cfg.pipeline.chunker)
    embedder = _build_embedder(cfg.pipeline.embedder)               # NEW switch
    base_retriever = _build_retriever(cfg.pipeline, embedder)       # existing + hybrid

    return EvalPipeline(
        chunker=chunker,
        embedder=embedder,
        retriever=base_retriever,
        rewriter=_build_rewriter(cfg.pipeline.query_rewriter),      # NEW (None when off)
        reranker=_build_reranker(cfg.pipeline.reranker),            # NEW (None when off)
        refusal_handler=_build_refusal(cfg.pipeline.refusal_handler),  # NEW (None when off)
        generator=_build_generator(cfg.pipeline.generator),
    )
```

Inside `EvalPipeline.query`:

```python
def query(self, q: str) -> EvalQueryResult:
    queries = self.rewriter.expand(q) if self.rewriter else [q]
    candidates = self.retriever.retrieve_many(queries)              # dedup inside
    if self.reranker:
        candidates = self.reranker.rerank(q, candidates)
    if self.refusal_handler and self.refusal_handler.should_refuse(candidates):
        return self.refusal_handler.refuse_response(timings=...)
    return self.generator.answer(q, candidates, timings=...)
```

Each lever is a no-op pass-through when its config is default; existing baselines route through identical control flow as today.

### 3.3 The 10-run matrix

All YAMLs under `configs/eval/phase2/`. Each inherits the previous tier's settings and toggles one field.

| File | Lever toggled | Key field deltas |
|------|----------------|----------------|
| `phase2_baseline.yaml` | (none — re-runs baseline at top of matrix) | identical to `baseline_squad_only.yaml` |
| `phase2a_chunking.yaml` | semantic chunker | `chunker.strategy: semantic` |
| `phase2b_embedder.yaml` | BGE embedder | + `embedder.name: bge_small_en_v1_5` |
| `phase2c_hybrid.yaml` | BM25 + dense | + `hybrid.enabled: true` |
| `phase2d_rerank.yaml` | cross-encoder rerank | + `reranker.model: ms_marco_minilm_l6_v2`, `rerank_top_n: 20` |
| `phase2e_rewrite.yaml` | query rewriting | + `query_rewriter.model: gpt-4.1-nano` |
| `phase2f_models_gpt5mini.yaml` | sweep variant A | + `generator.model: gpt-5-mini` (current) |
| `phase2f_models_gpt41mini.yaml` | sweep variant B | + `generator.model: gpt-4.1-mini` |
| `phase2f_models_haiku.yaml` | sweep variant C | + `generator.model: claude-haiku-4-5` |
| `phase2g_refusal.yaml` | refusal handler | + `refusal_handler.enabled: true` |

**Total runs: 10.** Conservative cost ceiling: $0.10/run × 10 = $1.00, well under the $5 budget.

---

## 4. Testing Strategy

Each new module ships with tests that lock its contract independently of the eval pipeline. Tests are tiered by speed: unit tests run on every CI invocation; integration tests run on demand.

### 4.1 Unit tests (per new module, fast, no LLM calls)

| Module | Assertions |
|--------|-----------|
| `BgeEmbedder` | `embed_documents([s])` returns a 384-dim vector. Cosine sim of `"cat"` and `"feline"` > `"cat"` and `"airplane"`. |
| `BM25HybridRetriever` | RRF fusion: given two ranked lists `A=[a,b,c]` and `B=[c,b,a]` with `rrf_k=60`, fused order is `c, b, a`. |
| `CrossEncoderReranker` | Given a query and 5 candidates with one obvious match, the match ranks first after `rerank()`. ≤ 10s. |
| `QueryRewriter` | `model=None` → `expand(q)` returns `[q]` unchanged. With stubbed LLMHandler returning fixed expansions, `expand(q)` returns deduped `[q, q', q'']`. |
| `RefusalHandler` | `should_refuse(candidates)` returns `True` when top-1 similarity < threshold; `False` otherwise. Empty candidates → refuse. |
| `SemanticChunker` (existing) | Re-chunking the same text returns identical chunks (regression test for determinism). |
| `PipelineCfg` schema | Loading `baseline.yaml` (no Phase-2 fields) validates with all defaults. Loading `phase2g_refusal.yaml` validates with `enabled: true`. Unknown field raises `ValidationError`. |

### 4.2 Factory tests (composition, no LLM calls)

`tests/eval/test_pipeline_factory_phase2.py`:
- For each of the 10 tier configs, `build_eval_pipeline(cfg)` returns an `EvalPipeline` whose attributes match expectations: `pipeline.rewriter is None` for tiers ≤ 2d, `pipeline.reranker is not None` for tiers ≥ 2d, etc.
- Defaults round-trip: a config with no `hybrid`/`reranker` blocks builds a pipeline equivalent to baseline.

### 4.3 End-to-end smoke (with stubbed LLM, no API spend)

`tests/eval/test_phase2_smoke.py`:
- Builds tier 2g pipeline against a tiny fixture corpus (3 docs).
- Sends one answerable question → non-refusal answer. Sends one nonsense question → refusal text.
- Asserts the right modules fired by reading `EvalPipeline.timings_ms` keys (`rewrite`, `retrieve`, `rerank`, `refusal_check`, `generate`).

### 4.4 Eval-data sanity tests (no API spend)

1. `tests/eval/test_phase2_configs_load.py` — every YAML under `configs/eval/phase2/` validates and produces a buildable pipeline.
2. `tests/eval/test_phase2_squad_dataset.py` — the SQuAD200 dataset loader still returns 200 questions with the same `questions.jsonl` SHA after Phase-2 changes (catches accidental dataset corruption).

### 4.5 Integration runs (real API, gated)

The 10 actual eval runs. Not in `tests/`; live in `eval_runs/`. Triggered manually:

```bash
make phase2-run    # or shell loop:
for cfg in configs/eval/phase2/*.yaml; do
    python -m src.eval.cli run --config "$cfg"
done
```

The harness is the test. Each run produces a comparable artifact; pairwise compare runs after the matrix completes.

---

## 5. Delivery Sequence + Risks

### 5.1 PR-A — pipeline extensions (~1500 lines, code-only)

Branched off `feature/eval-harness-1d`. Title: `feat(eval): pipeline extensions for Phase 2 RAG quality matrix`.

| # | Commit | Files |
|---|--------|-------|
| 1 | `feat(eval): extend PipelineCfg with Phase 2 sub-configs` | `src/eval/config.py`, `tests/eval/test_config.py` |
| 2 | `feat(eval): add BgeEmbedder for domain-tuned dense retrieval` | `src/eval/embedders/bge_small.py`, tests |
| 3 | `feat(eval): add BM25HybridRetriever with RRF fusion` | `src/eval/retrievers/bm25_hybrid.py`, tests, `requirements.txt` (+ `rank-bm25`) |
| 4 | `feat(eval): add CrossEncoderReranker (ms-marco-MiniLM)` | `src/eval/retrievers/reranker.py`, tests |
| 5 | `feat(eval): add QueryRewriter for LLM-based expansion` | `src/eval/transforms/query_rewriter.py`, tests |
| 6 | `feat(eval): add RefusalHandler with similarity gate` | `src/eval/transforms/refusal_handler.py`, tests |
| 7 | `feat(eval): wire Phase 2 levers into EvalPipeline factory` | `src/eval/runner/pipeline_factory.py`, factory tests, smoke test |
| 8 | `chore(eval): add Phase 2 tier configs under configs/eval/phase2/` | 10 YAML files |

**Acceptance for PR-A merge:**
- All unit + factory + smoke tests green.
- `pip install -r requirements.txt` adds exactly one entry (`rank-bm25`).
- `python -m src.eval.cli run --config configs/eval/phase2/phase2_baseline.yaml` succeeds end-to-end with stubbed-LLM mode.

### 5.2 PR-B — experiments + writeup (~10 run dirs + 1 doc, data-only)

Branched off whatever PR-A merges into. Title: `docs(eval): Phase 2 RAG quality matrix — results + findings`.

| # | Commit | Content |
|---|--------|---------|
| 1 | `chore(eval): execute Phase 2 baseline run` | `eval_runs/<id>_phase2_baseline/` |
| 2–7 | one commit per tier 2a–2e and 2g | run dir per tier |
| 8 | `chore(eval): execute Phase 2 tier 2f — answer-model sweep` | 3 run dirs (gpt-5-mini, gpt-4.1-mini, claude-haiku-4-5) |
| 9 | `feat(eval): pairwise compare reports for the Phase 2 matrix` | HTML reports under `eval_runs/_compare/` |
| 10 | `docs(eval): Phase 2 results — methodology, chart, findings` | `docs/PHASE2_RESULTS.md` |
| 11 | `docs(readme): link Phase 2 results from main README` | one-liner + link in `README.md` |

**Acceptance for PR-B merge:**
- 10 run dirs land on disk and render in `/eval`.
- `docs/PHASE2_RESULTS.md` contains: methodology, per-tier metric chart, paired-significance table, "winning stack" recipe, ≥ 1 finding per lever.
- API spend ledger documented in the writeup (target ≤ $5 actual).

### 5.3 Sequencing relative to Phase 1

PR-A stacks on Phase 1 PR #4 (`feature/eval-harness-1d`). When #4 merges, PR-A retargets to `main`. Same pattern as Phase 1 stack.

### 5.4 Risks + mitigations

| # | Risk | Mitigation |
|---|------|------------|
| R1 | A tier's metric drops sharply (e.g., hybrid hurts on SQuAD because BM25 dominates and dense gets averaged down) | Hard-budget rule: keep the data, write the negative finding. Don't tune to fit. |
| R2 | Cross-encoder model download fails offline | Cache in `~/.cache/huggingface`; tier 2d test asserts model loads from cache after first pull. |
| R3 | `rank-bm25` adds tokenization cost on long ML-papers chunks | SQuAD-only matrix sidesteps this. Phase 3 gets a pre-tokenized corpus index. |
| R4 | LLM-based query rewriting adds 200 extra LLM calls (one per query) → cost spike | Use `gpt-4.1-nano` for rewriting. Estimated marginal: $0.02/run. |
| R5 | Refusal handler over-refuses → answer_correctness regresses | Threshold 0.35 starts conservative. Tier 2g exists to *measure* this trade-off; both metrics reported. |
| R6 | Tier 2c (hybrid) requires re-indexing Chroma with BGE embeddings; tier 2b's index isn't reusable | EvalPipeline factory rebuilds the index per run anyway (existing pattern). No new infra. |
| R7 | docker-compose `eval_runs/` mount means PR-B's commits include large run-dir artifacts in git | `.gitignore` rule for run-internal binaries; commit only `metrics.json`, `cost.json`, `metadata.json`, `config.yaml`. Skip `questions.jsonl` if it crosses 1MB; reference by hash instead. |
| R8 | Hand-running 10 evals takes hours | Add a `make phase2-matrix` target that runs the full sweep with one command. Total wall time estimate: 6–8 hours unattended. |

---

## 6. Decisions Locked in This Spec

For the implementation plan to refer back to:

1. **Lane:** end-to-end overhaul (lane 3 of brainstorm Q2).
2. **Structure:** layered stack with retrieval-quality-first ordering (option A of brainstorm Q5).
3. **Eval set:** SQuAD-only (deferring `ml_papers_v1` to Phase 3).
4. **Budget rule:** hard $5 ceiling, no tier rollback (option 1 of brainstorm Q6).
5. **Deliverable:** data + portfolio writeup (option 2 of brainstorm Q7).
6. **Approach:** 2 PRs — PR-A pipeline extensions, PR-B experiments + writeup (option 3 of brainstorm Q-final).
7. **Branching:** off `feature/eval-harness-1d`; retargets to `main` after Phase 1 merges.
