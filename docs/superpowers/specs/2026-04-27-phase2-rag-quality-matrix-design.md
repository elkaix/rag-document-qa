# Phase 2 — RAG Quality Matrix (Design Spec)

> **Status:** revised 2026-04-27 after spec review. Pending implementation plan via `superpowers:writing-plans`.
> **Author:** Mohamed Elkholy
> **Predecessor:** [`2026-04-26-rag-eval-harness-phase-1-design.md`](./2026-04-26-rag-eval-harness-phase-1-design.md) (Phase 1, eval harness)
> **Eval baseline being targeted:** `eval_runs/2026-04-27_191733_baseline_squad_only_c925492` — 200 SQuAD v2 dev questions, real OpenAI API.

---

## 1. Goal + Scope

Phase 2 produces a layered ablation matrix that measures the lift each major RAG architectural lever buys on top of the Phase 1 baseline, then ships a portfolio-grade results document attributing the lift to mechanism.

### 1.1 Goal

Run 9 evals (1 baseline re-run + 5 single-lever tiers + 3 answer-model comparison variants) through the existing Phase 1 harness, with each tier in the layered chain inheriting from the previous tier so adjacent comparisons attribute lift to one mechanism at a time. Write findings.

> **Why not 10:** Tier 2a (semantic chunking) is **deferred to Phase 3**. The current SQuAD ingest path (`src/eval/pipeline_factory.py::_ingest_squad`) stores each question's gold context as a single Chroma document and never invokes the chunker, so toggling `chunker.strategy` cannot produce measurable lift on the SQuAD-only matrix. The chunker only fires on `ml_papers_v1` ingest (`_ingest_ml_papers`), so semantic-vs-recursive lift is a Phase 3 question.

### 1.2 In scope

- 10 eval YAML configs under `configs/eval/phase2/`, each runnable through `python -m src.eval.cli run`.
- 5 new pipeline modules wired into the `EvalPipeline` factory (not into prod `RAGBackend`):
  `BgeEmbedder`, `BM25HybridRetriever`, `CrossEncoderReranker`, `QueryRewriter`, `RefusalHandler`.
- Schema extension to `EvalConfig.pipeline` with 5 new sub-config blocks, all backward-compatible (defaults match current behavior).
- `docs/PHASE2_RESULTS.md`: methodology, per-tier metric chart, paired-significance table, "winning stack" recipe, ≥ 1 sentence finding per lever.

### 1.3 Out of scope

- Wiring any new module into the production `RAGBackend` / live `/chat` path. Phase 2 is measurement only.
- Adding `ml_papers_v1` to the matrix (deferred to Phase 3 once labeled).
- Auto-promotion of the winning stack to runtime defaults (Phase 2.5).

### 1.4 Success criteria

- All 8 distinct runs land on disk, visible in `/eval`, pairwise compare reports render.
- `PHASE2_RESULTS.md` ships with the chart + significance table + per-tier finding.
- Total API spend ≤ $5 (hard ceiling, no rollback). **Spend ledger covers all three LLM call sites** — generator, judge, and (if enabled) query-rewriter (see §4.6).

### 1.5 Branching

Phase 2 PR-A is stacked on `feature/eval-harness-1d` (current Phase 1 PR #4 tip). Once Phase 1 merges, PR-A retargets to `main`.

---

## 2. Architecture

The existing `EvalPipeline` factory at `src/eval/pipeline_factory.py` (`build_pipeline(config: EvalConfig, dataset_name: str, ...)` at line 269) already takes an `EvalConfig` and a dataset name and assembles a runtime pipeline. `EvalRunner` in `src/eval/runner.py:49` calls it directly. Phase 2 extends the existing function's switch statements and `EvalPipeline.__init__` signature; it does not introduce a new orchestrator and does not move the file.

### 2.1 Module map (5 new files, all under `src/`)

```
src/
├── eval/
│   ├── pipeline_factory.py        # existing — build_pipeline() extended with new branches
│   ├── runner.py                  # existing — _score_question extended with cost capture
│   ├── retrievers/                # NEW package
│   │   ├── __init__.py
│   │   ├── bm25_hybrid.py         # BM25HybridRetriever (RRF fusion)
│   │   └── reranker.py            # CrossEncoderReranker (ms-marco-MiniLM)
│   ├── embedders/                 # NEW package
│   │   ├── __init__.py
│   │   └── bge_small.py           # BgeEmbedder — Chroma EmbeddingFunction adapter
│   └── transforms/                # NEW package
│       ├── __init__.py
│       ├── query_rewriter.py      # LLM-based query expansion
│       └── refusal_handler.py     # answerability gate + no-answer prompt
```

Every new module is single-responsibility and budgeted under the 250-line CLAUDE.md ceiling. `pipeline_factory.py` is the only existing file at risk; if it crosses 250 lines, split into `pipeline_factory.py` (orchestration) + `factory_levers.py` (per-lever construction helpers).

### 2.2 Lever-by-lever wiring

| Tier | Module | Plug-in point in `EvalPipeline` | Notes |
|------|--------|---------------------------------|-------|
| ~~2a `chunking_semantic`~~ | — | — | **Deferred to Phase 3** (no-op on SQuAD ingest path). |
| 2b `embedder_bge` | `eval/embedders/bge_small.py` | passed to Chroma collection as its `embedding_function` at collection-creation time inside `build_pipeline`; `ChromaVectorStore.upsert/query` then auto-embeds via the collection function (see §3.2). | Uses `sentence-transformers` `BAAI/bge-small-en-v1.5` (33M params, 384-dim, on-device). |
| 2c `hybrid_bm25` | `eval/retrievers/bm25_hybrid.py` | retriever switch | Wraps `rank_bm25` + `ChromaVectorStore`; RRF fusion on top-20 from each side. |
| 2d `rerank_crossenc` | `eval/retrievers/reranker.py` | post-retrieval hook | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-20 → top-5. |
| 2e `query_rewrite` | `eval/transforms/query_rewriter.py` | pre-retrieval hook | One LLM call per query; 1–3 expansions, dedup-merged. Uses `gpt-4.1-nano`. |
| 2g `refusal_handler` | `eval/transforms/refusal_handler.py` | post-retrieval gate | Threshold on top-1 similarity; if low, short-circuit to no-answer text. |
| 2f `answer_model` (parallel comparison) | no new module | generator switch (existing) | YAML-only — three runs sweep `gpt-5-mini`, `gpt-4.1-mini`, `claude-haiku-4-5`, all on the **full 2g stack**. Reported as a side-by-side comparison, not a layered tier (see §3.3.2). |

### 2.3 Data flow per query (final tier 2g)

```
query
  └─► QueryRewriter ─► {q, q', q''}              (2e)
        └─► HybridRetriever (BM25 + BGE)          (2b, 2c)
              └─► CrossEncoderReranker top-5      (2d)
                    └─► RefusalHandler            (2g)
                          ├─► low conf ─► "I don't know"
                          └─► high conf ─► Generator (gpt-5-mini)
                                            └─► answer + telemetry
```

Each tier's config disables levers introduced after it (e.g., tier 2c sets `reranker.model: null`, `query_rewriter.model: null`). Tier 2f re-uses the 2g pipeline three times with three different `generator.model` values; it is a comparison, not a chain link.

### 2.4 Dependency additions

| Tier needing it | Dep | Size |
|------|-----|------|
| 2b | `sentence-transformers` already in deps (used by judge embedder) | 0 |
| 2c | `rank-bm25` | tiny, pure-Python |
| 2d | `sentence-transformers` cross-encoder model — pulled at runtime | model ~80MB |
| 2e | none — uses existing `LLMHandler` | 0 |
| 2f | none | 0 |
| 2g | none — pure logic | 0 |

Net new third-party deps: **just `rank-bm25`**.

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

`build_pipeline(config: EvalConfig, dataset_name: str, ...)` in `src/eval/pipeline_factory.py:269` is extended in place. The embedder is selected first because the Chroma collection must be created with the right embedding function:

```python
# src/eval/pipeline_factory.py — extended

def build_pipeline(
    config: EvalConfig, dataset_name: str, ...,
) -> EvalPipeline:
    embedding_function = _build_embedding_function(config.pipeline.embedder)  # NEW
    collection = _make_collection(name=..., embedding_function=embedding_function)
    vector_store = ChromaVectorStore(collection=collection)
    chunker = _build_chunker(config.pipeline.chunker)

    base_retriever = _build_retriever(config.pipeline, vector_store)          # extended for hybrid
    return EvalPipeline(
        dataset_name=dataset_name,
        chunker=chunker,
        vector_store=vector_store,
        retriever=base_retriever,
        rewriter=_build_rewriter(config.pipeline.query_rewriter),             # NEW (None when off)
        reranker=_build_reranker(config.pipeline.reranker),                   # NEW (None when off)
        refusal_handler=_build_refusal(config.pipeline.refusal_handler),      # NEW (None when off)
        generator=_build_generator(config.pipeline.generator),
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

#### 3.2.1 BGE embedder wiring (resolves spec-review H3)

`BgeEmbedder` is **a Chroma `EmbeddingFunction` adapter**, not a separate vector encoder the pipeline calls explicitly. `_build_embedding_function(EmbedderCfg)` returns one of:

- `EmbedderCfg.name == "chroma_default"` → `chromadb.utils.embedding_functions.DefaultEmbeddingFunction()` (current behavior; ChromaDB's built-in ONNX MiniLM, 384-dim).
- `EmbedderCfg.name == "bge_small_en_v1_5"` → `BgeEmbedder()`, a `chromadb.api.types.EmbeddingFunction` subclass that loads `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384-dim) and exposes `__call__(input: list[str]) -> list[list[float]]`.

The embedding function is set on the Chroma `Collection` at creation time. `ChromaVectorStore.upsert(documents=..., metadatas=...)` and `ChromaVectorStore.query(query_text=..., top_k=...)` then auto-embed via the collection's function (`src/vector_store.py:109-145`, `:154-175`); no per-call code change.

Because `EvalRunner` recreates the collection on every run (existing pattern: each run is named after the run_id and is destroyed afterwards), there is **no risk of dimension mixing** between the default embedder and BGE, and **no migration work** between tiers.

Tests for `BgeEmbedder`:
1. Conforms to `EmbeddingFunction` ABC (returns `list[list[float]]` of 384-dim vectors).
2. Works through `ChromaVectorStore` end-to-end on a 3-doc fixture (upsert → query → top-1 ID is the expected doc).
3. Cosine distance on the synonym pair `("cat", "feline")` is smaller than the unrelated pair `("cat", "airplane")` — proves the model is loading the right weights, not a stub.

### 3.3 The 9-run matrix

All YAMLs under `configs/eval/phase2/`. The matrix has two parts:

#### 3.3.1 Layered chain (6 runs)

Each tier inherits the previous tier's settings and toggles **one** field. `generator.model` stays at `gpt-5-mini` throughout the chain so the model variable is held constant; the answer-model question is answered separately by §3.3.2.

| File | Lever toggled | Key field deltas (cumulative on previous tier) |
|------|----------------|----------------|
| `phase2_baseline.yaml` | (none — baseline re-run) | identical to `baseline_squad_only.yaml`; serves as the chain's anchor and lets the runner re-attribute spend to Phase 2 |
| `phase2b_embedder.yaml` | BGE embedder | + `embedder.name: bge_small_en_v1_5` |
| `phase2c_hybrid.yaml` | BM25 + dense | + `hybrid.enabled: true` |
| `phase2d_rerank.yaml` | cross-encoder rerank | + `reranker.model: ms_marco_minilm_l6_v2`, `rerank_top_n: 20`, `final_top_k: 5` |
| `phase2e_rewrite.yaml` | query rewriting | + `query_rewriter.model: gpt-4.1-nano`, `max_expansions: 3` |
| `phase2g_refusal.yaml` | refusal handler | + `refusal_handler.enabled: true`, `similarity_threshold: 0.35`; **`generator.model: gpt-5-mini`** (held constant) |

#### 3.3.2 Answer-model comparison (3 runs, parallel to the chain)

Tier 2f is **not** a chain link; it is three side-by-side runs that share the **full 2g pipeline** and vary **only** the `generator.model`. This decouples answer-model choice from the layered question and lets the writeup report a clean model comparison on a fixed retrieval/refusal stack. Each YAML inherits everything from `phase2g_refusal.yaml` and overrides `generator.model`:

| File | `generator.model` |
|------|-------------------|
| `phase2f_models_gpt5mini.yaml`  | `gpt-5-mini` (this run is identical to `phase2g_refusal.yaml`; PR-B re-uses that artifact rather than re-running it) |
| `phase2f_models_gpt41mini.yaml` | `gpt-4.1-mini` |
| `phase2f_models_haiku.yaml`     | `claude-haiku-4-5` |

**Total runs: 9** (6 chain + 3 model-comparison; the gpt-5-mini variant of 2f is the same artifact as `phase2g_refusal.yaml`, so PR-B authors 9 distinct YAMLs but only 8 runs need to execute).

#### 3.3.3 Significance comparisons reported

The writeup reports paired permutation tests on:
- 5 chain comparisons: 2b–baseline, 2c–2b, 2d–2c, 2e–2d, 2g–2e.
- 2 cross-model comparisons: gpt-4.1-mini–gpt-5-mini, claude-haiku-4-5–gpt-5-mini (each holding the rest of the 2g stack constant).

**Conservative cost ceiling:** $0.10/run × 8 distinct runs = $0.80, well under the $5 budget.

---

## 4. Testing Strategy

Each new module ships with tests that lock its contract independently of the eval pipeline. Tests are tiered by speed: unit tests run on every CI invocation; integration tests run on demand.

### 4.1 Unit tests (per new module, fast, no LLM calls)

| Module | Assertions |
|--------|-----------|
| `BgeEmbedder` | `embed_documents([s])` returns a 384-dim vector. Cosine sim of `"cat"` and `"feline"` > `"cat"` and `"airplane"`. |
| `BM25HybridRetriever` | RRF fusion on asymmetric inputs (avoids tied scores): `A=[a,b,c,d]`, `B=[d,a]`, `rrf_k=60` → fused order `a, d, b, c`. (`a` wins from joint coverage at 1/61+1/62; `d` is second from rank-1 in B at 1/61+1/64; then `b` at 1/62; then `c` at 1/63.) |
| `CrossEncoderReranker` | Given a query and 5 candidates with one obvious match, the match ranks first after `rerank()`. ≤ 10s. |
| `QueryRewriter` | `model=None` → `expand(q)` returns `[q]` unchanged. With stubbed LLMHandler returning fixed expansions, `expand(q)` returns deduped `[q, q', q'']`. |
| `RefusalHandler` | `should_refuse(candidates)` returns `True` when top-1 similarity < threshold; `False` otherwise. Empty candidates → refuse. |
| `SemanticChunker` (existing) | Re-chunking the same text returns identical chunks (regression test for determinism). |
| `PipelineCfg` schema | Loading `baseline.yaml` (no Phase-2 fields) validates with all defaults. Loading `phase2g_refusal.yaml` validates with `enabled: true`. Unknown field raises `ValidationError`. |

### 4.2 Factory tests (composition, no LLM calls)

`tests/eval/test_pipeline_factory_phase2.py`:
- For each of the 9 Phase 2 YAML configs, `build_pipeline(config, dataset_name="squad_v2_dev_200")` returns an `EvalPipeline` whose attributes match expectations: `pipeline.rewriter is None` for tiers ≤ 2d; `pipeline.reranker is not None` for tiers ≥ 2d; `pipeline.refusal_handler is not None` only for tier 2g and the three 2f variants.
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

The 8 distinct eval runs. Not in `tests/`; live in `eval_runs/`. Triggered manually:

```bash
make phase2-matrix    # or shell loop:
for cfg in configs/eval/phase2/*.yaml; do
    python -m src.eval.cli run --config "$cfg"
done
```

The harness is the test. Each run produces a comparable artifact; pairwise compare runs after the matrix completes.

### 4.6 Cost ledger (resolves spec-review M4)

Phase 1's `cost.json` aggregates only generator-side cost because that's the only call site `EvalPipeline.query` knew about. Phase 2 has three LLM call sites (generator, judge, query-rewriter) and the hard $5 ceiling has to be enforceable across all of them. The fix is local:

1. **Per-question record extension.** `EvalResult.cost_usd` becomes `EvalResult.cost_breakdown: dict[str, float]` (`{"generator": ..., "rewriter": ..., "judge": ...}`) plus `cost_usd` (sum). Schema migration is back-compatible — older records read with `cost_breakdown` defaulting to `{"generator": cost_usd}`.
2. **Generator cost.** Already captured in `EvalPipeline.query` telemetry. No change.
3. **Rewriter cost.** New: `QueryRewriter.expand` returns `(queries, cost_usd, prompt_tokens, completion_tokens)`. `EvalPipeline.query` adds the rewriter cost into the per-question breakdown.
4. **Judge cost.** `_score_question` in `src/eval/runner.py:312` already calls each judge through `LLMHandler`. Extend the judge wrappers (`src/eval/metrics/judge_*.py`) to return `(score, cost_usd, prompt_tokens, completion_tokens)`; sum into the per-question record.
5. **Aggregator.** `cost.json` totals `cost_breakdown` instead of `cost_usd`. Adds three new lines (`generator_total`, `rewriter_total`, `judge_total`) plus the existing `total_usd`.
6. **Guardrail.** `EvalRunner` reads `eval.spend_ceiling_usd: float | None` from the config (NEW field, defaults to None). When set, the runner aborts with a clear error if the running cumulative cost exceeds it. Phase 2 configs set `spend_ceiling_usd: 1.50` per run; the matrix-level ceiling stays in the writeup ledger.

This addition is part of PR-A (commit 7 — factory wiring) so PR-B can rely on the new schema when it lands.

---

## 5. Delivery Sequence + Risks

### 5.1 PR-A — pipeline extensions (~1700 lines, code-only)

Branched off `feature/eval-harness-1d`. Title: `feat(eval): pipeline extensions for Phase 2 RAG quality matrix`.

| # | Commit | Files |
|---|--------|-------|
| 1 | `feat(eval): extend PipelineCfg with Phase 2 sub-configs and spend ceiling` | `src/eval/config.py`, `tests/eval/test_config.py` |
| 2 | `feat(eval): extend cost ledger to capture generator + judge + rewriter spend` | `src/eval/schemas.py` (`EvalResult.cost_breakdown`), `src/eval/runner.py` (`_score_question`), judge wrappers under `src/eval/metrics/`, aggregator, tests |
| 3 | `feat(eval): add BgeEmbedder as Chroma EmbeddingFunction adapter` | `src/eval/embedders/bge_small.py`, tests (incl. end-to-end through `ChromaVectorStore`) |
| 4 | `feat(eval): add BM25HybridRetriever with RRF fusion` | `src/eval/retrievers/bm25_hybrid.py`, tests, `requirements.txt` (+ `rank-bm25`) |
| 5 | `feat(eval): add CrossEncoderReranker (ms-marco-MiniLM)` | `src/eval/retrievers/reranker.py`, tests |
| 6 | `feat(eval): add QueryRewriter for LLM-based expansion (with cost capture)` | `src/eval/transforms/query_rewriter.py`, tests |
| 7 | `feat(eval): add RefusalHandler with similarity gate` | `src/eval/transforms/refusal_handler.py`, tests |
| 8 | `feat(eval): wire Phase 2 levers into build_pipeline + spend guardrail` | `src/eval/pipeline_factory.py`, factory tests, smoke test, runner spend-ceiling enforcement |
| 9 | `feat(eval): add `archive` subcommand to copy small run artifacts to a tracked tree` | `src/eval/cli.py`, tests |
| 10 | `chore(eval): add Phase 2 tier configs under configs/eval/phase2/` | 9 YAML files |

**Acceptance for PR-A merge:**
- All unit + factory + smoke tests green.
- `pip install -r requirements.txt` adds exactly one entry (`rank-bm25`).
- Existing baseline configs (`baseline.yaml`, `baseline_squad_only.yaml`) still load and produce the same pipeline shape as before Phase 2 (tested in commit 1).
- `python -m src.eval.cli run --config configs/eval/phase2/phase2_baseline.yaml` succeeds end-to-end with stubbed-LLM mode.
- `python -m src.eval.cli archive <run_id> --to /tmp/test_archive/` produces a folder containing exactly `metrics.json`, `cost.json`, `metadata.json`, `config.yaml`.

### 5.2 PR-B — experiments + writeup (8 distinct runs + 1 doc, data-only)

Branched off whatever PR-A merges into. Title: `docs(eval): Phase 2 RAG quality matrix — results + findings`.

**Artifact location (resolves spec-review M6).** `eval_runs/` is gitignored at `.gitignore:15` and stays that way (run directories can be hundreds of MB and contain large `questions.jsonl` files). PR-B does **not** commit the live `eval_runs/<id>/` directories. Instead, PR-A adds a small CLI helper:

```bash
python -m src.eval.cli archive <run_id> --to docs/phase2/runs/<short_id>/
```

which copies only the small, reviewable artifacts (`metrics.json`, `cost.json`, `metadata.json`, `config.yaml`) into the tracked location. `questions.jsonl` is excluded from the archive (large) but its SHA goes into `metadata.json` so reviewers can verify the writeup against a re-run.

| # | Commit | Content |
|---|--------|---------|
| 1 | `chore(eval): execute Phase 2 baseline run` | `docs/phase2/runs/<short_id>_phase2_baseline/{metrics,cost,metadata,config}.json` (run dir stays local in `eval_runs/`) |
| 2–6 | one commit per tier 2b, 2c, 2d, 2e, 2g | archived artifacts per tier |
| 7 | `chore(eval): execute Phase 2 tier 2f — answer-model comparison (gpt-4.1-mini, claude-haiku-4-5)` | 2 archived run dirs (gpt-5-mini variant re-uses `phase2g_refusal` artifact) |
| 8 | `feat(eval): pairwise compare reports for the Phase 2 matrix` | HTML reports under `docs/phase2/compare/` |
| 9 | `docs(eval): Phase 2 results — methodology, chart, findings` | `docs/PHASE2_RESULTS.md` |
| 10 | `docs(readme): link Phase 2 results from main README` | one-liner + link in `README.md` |

**Acceptance for PR-B merge:**
- 8 archived run dirs land under `docs/phase2/runs/`. The full `eval_runs/<id>/` continues to render in the local `/eval` UI but isn't part of the PR diff.
- `docs/PHASE2_RESULTS.md` contains: methodology, per-tier metric chart, paired-significance table for the 5 chain comparisons + 2 cross-model comparisons, "winning stack" recipe, ≥ 1 finding per lever.
- API spend ledger from `cost.json` totals (now including judge + rewriter, see §4.6) documented in the writeup. Target ≤ $5 actual.

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
| R7 | `eval_runs/` is gitignored, so committing live run dirs would either fail or require force-add | Resolved by §5.2: `cli archive` copies the small artifacts (`metrics.json`, `cost.json`, `metadata.json`, `config.yaml`) into the tracked `docs/phase2/runs/` tree. `questions.jsonl` SHA is recorded in `metadata.json` for reviewer-side verification. |
| R8 | Hand-running 8 distinct evals takes hours | Add a `make phase2-matrix` target that runs the full sweep with one command. Total wall time estimate: 5–7 hours unattended. |

---

## 6. Decisions Locked in This Spec

For the implementation plan to refer back to:

1. **Lane:** end-to-end overhaul (lane 3 of brainstorm Q2).
2. **Structure:** layered stack with retrieval-quality-first ordering (option A of brainstorm Q5).
3. **Eval set:** SQuAD-only (deferring `ml_papers_v1` to Phase 3).
4. **Budget rule:** hard $5 ceiling, no tier rollback (option 1 of brainstorm Q6). Enforced per-run via `eval.spend_ceiling_usd` (§4.6) covering generator + judge + rewriter call sites.
5. **Deliverable:** data + portfolio writeup (option 2 of brainstorm Q7).
6. **Approach:** 2 PRs — PR-A pipeline extensions, PR-B experiments + writeup (option 3 of brainstorm Q-final).
7. **Branching:** off `feature/eval-harness-1d`; retargets to `main` after Phase 1 merges.

### 6.1 Revisions from spec review (2026-04-27)

- **H1.** Tier 2a (semantic chunking) deferred to Phase 3 — chunker is bypassed on SQuAD ingest path. Matrix shrinks to 9 YAMLs / 8 distinct runs.
- **H2.** Factory references updated to `src/eval/pipeline_factory.py::build_pipeline(config, dataset_name, ...)`. `EvalRunner` lives in `src/eval/runner.py`.
- **H3.** `BgeEmbedder` is a Chroma `EmbeddingFunction` adapter, installed on the collection at creation time. No per-call code change in `EvalPipeline`. Factory creates a fresh collection per run, eliminating dimension-mixing risk.
- **M4.** Cost ledger covers all three LLM call sites (generator, judge, rewriter) via `EvalResult.cost_breakdown`; spend ceiling enforced at runner level (§4.6).
- **M5.** Tier 2f reframed as a parallel three-way answer-model comparison on top of the full 2g stack. Tier 2g pins `generator.model: gpt-5-mini` to keep the chain's model variable constant.
- **M6.** Run artifacts archived to `docs/phase2/runs/` (tracked) via a new `cli archive` subcommand. Live `eval_runs/` stays gitignored.
- **L7.** RRF unit test rewritten with asymmetric inputs (`A=[a,b,c,d]`, `B=[d,a]`) so all four scores are distinct.
