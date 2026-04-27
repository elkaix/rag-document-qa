# RAG Eval Harness — Phase 1 Design Spec

**Status:** Draft, awaiting review.
**Phase:** 1 of 3 (Applied/GenAI track of A+C portfolio strategy).
**Predecessor:** `2026-04-19-rag-evaluation-pipeline-design.md` — implemented per-message LLM-as-judge metrics; this Phase **builds on those**, it does not replace them.

---

## 0. Why This Phase Exists

The repo already has:
- A production-ready RAG pipeline (FastAPI + ChromaDB + WebSocket streaming + React).
- Per-message LLM-as-judge scoring (`src/evaluation.py`) for faithfulness, answer relevancy, context precision — fired in real-time per chat answer and stored in SQLite.

What it does **not** have:
- A labeled gold dev set with ground-truth answers and supporting passages.
- Retrieval-quality metrics with ground truth: Recall@k, MRR, nDCG.
- Answer Correctness vs a gold answer (the existing relevancy metric is reference-free).
- Refusal correctness on unanswerable questions.
- Statistical rigor: bootstrap confidence intervals and paired significance tests.
- A reproducible batch eval runner over a frozen dev set with versioned configs.
- A comparison surface to diff two pipeline configurations on the same dev set.
- LLM observability (per-stage spans, token counts, cost) viewable as traces.

Phase 1 closes those gaps. This is the foundation Phase 2 (hybrid search, reranking, query rewriting) and Phase 3 (custom embedder, custom reranker, distillation) need to make defensible "X beats Y by Z%" claims.

## 1. Goal

Turn the existing RAG system into a **measurable** system. After Phase 1, every pipeline change can be evaluated on retrieval quality, generation quality, latency, and cost — with statistical rigor — and the results are visible in the UI.

## 2. Success Criteria

A staff ML engineer reviewing the repo can:

1. Run `python -m src.eval.cli run --config configs/eval/baseline.yaml` and get a complete eval report in <10 minutes on the SQuAD-200 + ml_papers_v1 sets.
2. Open `/eval` in the React UI, pick two runs, and see a side-by-side metric comparison with significance markers.
3. Read `eval_data/ml_papers_v1/LABELING_GUIDE.md` and understand exactly how the gold set was constructed.
4. Open Phoenix at `localhost:6006` and trace a single query through `retrieve → generate` with per-span latencies, token counts, and cost in USD.
5. Run `pytest tests/test_eval_*.py -v` and see all tests pass on a fresh checkout.

## 3. Scope

### In scope
- Two frozen labeled dev sets (SQuAD-200 anchor + ml_papers_v1 domain).
- Retrieval metrics over gold chunk IDs (Recall@k, MRR@k, nDCG@k for k ∈ {1, 3, 5, 10}).
- Answer Correctness vs gold answer (semantic similarity + LLM-as-judge factual match).
- Refusal correctness for unanswerable questions.
- Operational metrics: latency p50/p95/p99 per stage, cost in USD per query.
- Bootstrap confidence intervals (n=1000) on every aggregated metric.
- Paired permutation test (n=10000) for two-run comparison.
- Eval CLI + FastAPI routes + React `/eval` page with list/detail/compare views.
- OpenTelemetry SDK instrumentation, OTLP export to Phoenix, per-stage timings rendered inline in chat answer footer.
- Reuse of existing `src/evaluation.py` LLM-as-judge functions for faithfulness, relevancy, context precision in batch mode.

### Out of scope (deferred)
- Hybrid search / BM25 → **Phase 2**.
- Cross-encoder reranking → **Phase 2**.
- Query rewriting / HyDE → **Phase 2**.
- A/B traffic split on live `/api/chat` → **Phase 2**.
- Training a custom embedder → **Phase 3**.
- Fine-tuning a cross-encoder → **Phase 3**.
- Distillation experiments → **Phase 3**.
- Continuous eval on every commit (CI) → out of portfolio scope; could revisit if hiring signal justifies it.

## 4. Eval Sets

### 4.1 SQuAD-200 anchor set

- **Source:** `squad_v2` Hugging Face dataset, `validation` split.
- **Sampling:** seeded random sample of 200 (question, context, answers) tuples; ~20% drawn from `is_unanswerable=True` rows so the refusal metric has signal.
- **Frozen artifact:** `eval_data/squad_v2_dev_200/questions.jsonl` checked into git, plus `seed.txt` so anyone can regenerate.
- **Corpus:** the 200 source contexts are ingested into a separate Chroma collection (`squad_v2_dev_200_corpus`) so SQuAD evaluation does not pollute the user-facing collection.

### 4.2 ml_papers_v1 domain set

- **Corpus:** 5–10 ML/AI papers covering transformers, BERT, RAG, DPR, ColBERT, RAGAS, LoRA, Sentence-BERT, BGE, and *The Annotated Transformer*. Exact list and SHA-256 of each PDF stored in `eval_data/ml_papers_v1/corpus_manifest.json`.
- **Q/A:** 50 hand-labeled rows with the same schema as SQuAD; ~10 marked `is_unanswerable=True` to test refusal on out-of-scope questions.
- **Labeling rubric:** `eval_data/ml_papers_v1/LABELING_GUIDE.md` documents:
  - Difficulty buckets (definition recall / multi-step reasoning / cross-paper).
  - Anti-patterns to avoid (questions answerable from question text alone, ambiguous wording, etc.).
  - Requirements for `gold_chunk_ids` (must point at chunks that *causally* support the answer, not merely mention the topic).
  - Inter-annotator considerations (if a second labeler is used, agreement protocol).
- **Corpus is ingested into a separate collection** (`ml_papers_v1_corpus`) for the same isolation reason as 4.1.

### 4.3 Schema (shared)

```python
class EvalQuestion(BaseModel):
    id: str                     # stable hash of question text
    question: str
    gold_answer: str | None     # None when is_unanswerable
    gold_chunk_ids: list[str]   # ≥1 chunks that causally support answer; [] if unanswerable
    is_unanswerable: bool = False
    metadata: dict[str, Any] = {}   # source_paper, paper_section, difficulty, etc.
```

## 5. Metric Set (Tier 3)

### Retrieval (over `gold_chunk_ids`)
- **Recall@k** for k ∈ {1, 3, 5, 10}.
- **MRR@k** for k ∈ {1, 3, 5, 10}.
- **nDCG@k** for k ∈ {1, 3, 5, 10}.

### Generation
- **Faithfulness** — reuse `src.evaluation.evaluate_faithfulness`.
- **Answer Relevancy** — reuse `src.evaluation.evaluate_answer_relevancy`.
- **Context Precision** — reuse `src.evaluation.evaluate_context_precision`.
- **Context Recall** (NEW) — fraction of gold-supporting passages that appear in retrieved context (computed directly from `gold_chunk_ids` ∩ `retrieved_chunk_ids`, no LLM judge needed).
- **Answer Correctness** (NEW) — two sub-scores combined:
  - `cosine(embed(answer), embed(gold_answer))` using the existing all-MiniLM-L6-v2.
  - LLM-as-judge factual match (returns 0.0 / 0.5 / 1.0).
  - Final score = mean of the two; both stored separately in `details`.

### Refusal
- **Refusal Correctness** (NEW) — for `is_unanswerable=True` rows: 1.0 if answer correctly refuses, 0.0 otherwise. Refusal detected by (a) regex match on refusal phrases (fast path) and (b) LLM-as-judge fallback when regex is ambiguous.

### Operational
- Per-stage latency (`retrieve`, `generate`) in ms, aggregated as p50/p95/p99.
- Per-query cost in USD computed from `(prompt_tokens, completion_tokens) × model price table`.
- Per-query token counts (prompt + completion).

### Statistical wrappers
- Every aggregate metric is reported as `mean ± 95% bootstrap CI` (n=1000 resamples).
- For two-run comparison, `paired permutation test` (n=10000) returns a p-value per metric.

## 6. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Eval Layer (NEW)                                               │
│                                                                 │
│  src/eval/                                                      │
│  ├── __init__.py                                                │
│  ├── schemas.py            EvalQuestion, EvalResult,            │
│  │                         AggregatedMetric, RunMetadata        │
│  ├── datasets/                                                  │
│  │   ├── __init__.py                                            │
│  │   ├── squad_v2.py       sample 200 seeded; ingest corpus     │
│  │   └── ml_papers.py      load jsonl; verify manifest SHA-256s │
│  ├── metrics/                                                   │
│  │   ├── __init__.py                                            │
│  │   ├── retrieval.py      recall_at_k, mrr_at_k, ndcg_at_k    │
│  │   ├── generation.py     thin wrappers around evaluation.py  │
│  │   │                     + answer_correctness, context_recall │
│  │   ├── refusal.py        refusal_correctness                  │
│  │   └── operational.py    latency/cost/token aggregators       │
│  ├── statistics.py         bootstrap_ci, paired_permutation     │
│  ├── pricing.py            { model_id → (prompt $/1M, comp $/1M)} │
│  ├── config.py             EvalConfig pydantic loaded from YAML │
│  ├── runner.py             RunEval(config) → writes run dir     │
│  ├── compare.py            two-run diff → HTML + CompareResult  │
│  ├── storage.py            list/load/save eval runs             │
│  └── cli.py                argparse: run, compare, list, show   │
│                                                                 │
│  src/api/routes/eval.py    POST   /api/eval/run                 │
│                            GET    /api/eval/runs                │
│                            GET    /api/eval/runs/{id}           │
│                            GET    /api/eval/compare?a=&b=       │
│                            GET    /api/eval/configs             │
│                                                                 │
│  src/observability.py      OTel tracer setup, OTLP exporter,    │
│                            @traced_stage decorator              │
│                                                                 │
│  configs/eval/                                                  │
│  ├── baseline.yaml         current default pipeline             │
│  └── README.md             how to author new configs            │
│                                                                 │
│  frontend/src/pages/EvalPage.tsx                                │
│  ├── EvalRunsList                                               │
│  ├── EvalRunDetail         per-question table + metrics chart   │
│  ├── EvalCompareView       side-by-side bars + signif markers   │
│  └── NewEvalRunDialog      config picker → POST /api/eval/run   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Existing RAG (unchanged interface, instrumented)               │
│                                                                 │
│  RAGBackend.query() now wrapped in OTel spans:                  │
│    ├── span: rag.retrieve  (chunk_count, top_k, distances[])    │
│    └── span: rag.generate  (model, prompt_tokens,               │
│                             completion_tokens, cost_usd)        │
│  Spans exported via OTLP → Phoenix (localhost:6006)             │
│                                                                 │
│  Per-stage timings + tokens + cost also returned in API         │
│  response and rendered as small footer under each chat answer:  │
│    "Retrieve 142ms · Generate 2.1s · 4,217 tok · $0.0083"      │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Data Contracts

### EvalConfig (YAML)
```yaml
# configs/eval/baseline.yaml
name: "baseline"
description: "Current production defaults"
pipeline:
  chunker:
    strategy: "recursive"
    chunk_size: 512
    chunk_overlap: 64
  retriever:
    top_k: 5
  generator:
    model: "gpt-5-mini"
    reasoning_model: "gpt-4.1-nano"
eval:
  datasets: ["squad_v2_dev_200", "ml_papers_v1"]
  judge_model: "gpt-4.1-mini"
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
```

### EvalResult (per-question row in `questions.jsonl`)
```python
class EvalResult(BaseModel):
    question_id: str
    dataset: str                          # "squad_v2_dev_200" | "ml_papers_v1"
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[str]           # text, for the UI
    generated_answer: str
    metrics: dict[str, float]             # see §5
    metric_details: dict[str, Any]        # per-claim breakdowns, etc.
    timings_ms: dict[str, float]          # {retrieve: 142, generate: 2103}
    tokens: dict[str, int]                # {prompt: 3417, completion: 800}
    cost_usd: float
    error: str | None = None              # populated if pipeline raised
```

### AggregatedMetric (in `metrics.json`)
```python
class AggregatedMetric(BaseModel):
    metric_name: str
    dataset: str | None                   # None = combined across datasets
    mean: float
    ci_low: float                         # 2.5th percentile of bootstrap dist
    ci_high: float                        # 97.5th percentile
    n: int
```

### RunMetadata (in `metadata.json`)
```python
class RunMetadata(BaseModel):
    run_id: str                           # YYYY-MM-DD_HHMMSS_<config-name>_<git-sha-short>
    config_name: str
    config_path: str
    git_sha: str
    started_at: datetime
    finished_at: datetime
    env_hash: str                         # hash of requirements.txt
    eval_set_versions: dict[str, str]     # {dataset: sha256-of-questions.jsonl}
    n_questions: int
    n_errors: int
```

### CompareResult (returned by `/api/eval/compare`)
```python
class MetricDelta(BaseModel):
    metric_name: str
    dataset: str | None
    a_mean: float
    a_ci: tuple[float, float]
    b_mean: float
    b_ci: tuple[float, float]
    delta: float                          # b_mean - a_mean
    p_value: float                        # paired permutation test
    significant: bool                     # p_value < 0.05

class CompareResult(BaseModel):
    run_a: RunMetadata
    run_b: RunMetadata
    deltas: list[MetricDelta]
    per_question_diff: list[dict]         # questions where a and b disagreed most
```

## 8. Run Directory Layout

```
eval_runs/
  2026-04-26_143022_baseline_a3f9c1/
    config.yaml                  # frozen copy of the EvalConfig
    metadata.json                # RunMetadata
    questions.jsonl              # one EvalResult per line
    metrics.json                 # list[AggregatedMetric]
    cost.json                    # totals + per-question breakdown
    traces/                      # exported OTel spans (one .jsonl per question)
    report.html                  # standalone shareable report (regenerated on demand)
```

`eval_runs/` is gitignored. The labeled dev sets in `eval_data/` are checked in.

## 9. Observability

- **`src/observability.py`** initializes an OTel `TracerProvider` with an OTLP HTTP exporter pointing to `http://localhost:6006/v1/traces` (Phoenix). If Phoenix is unreachable, the SDK fails quietly — eval and chat continue.
- A `@traced_stage("rag.retrieve")` decorator wraps the retriever and generator entry points in `RAGBackend`; span attributes carry `top_k`, `chunk_count`, `distances`, `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`.
- The **API response contract is extended** so the frontend gets per-stage timings, tokens, and cost without parsing OTel:
  ```python
  class StageTelemetry(BaseModel):
      retrieve_ms: float
      generate_ms: float
      prompt_tokens: int
      completion_tokens: int
      cost_usd: float
  ```
  Returned both in the REST `/api/query` response and as a final WebSocket event:
  ```json
  {"type": "telemetry", "content": {"retrieve_ms": 142, "generate_ms": 2103, "prompt_tokens": 3417, "completion_tokens": 800, "cost_usd": 0.0083}}
  ```
- The frontend renders a small, muted footer under each assistant bubble:
  > *Retrieve 142ms · Generate 2.1s · 4,217 tok · $0.0083*

## 10. API Surface (new)

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/eval/configs` | List available `configs/eval/*.yaml`. |
| `POST` | `/api/eval/run` | Body `{config_name}`. Kicks off a background task; returns `{run_id, status_url}`. |
| `GET`  | `/api/eval/runs` | List all runs with `RunMetadata` summaries. |
| `GET`  | `/api/eval/runs/{id}` | Full run: metadata + aggregated metrics + per-question results (paginated). |
| `GET`  | `/api/eval/runs/{id}/status` | Status polling for in-flight runs. |
| `GET`  | `/api/eval/compare?a={id_a}&b={id_b}` | `CompareResult`. |

All endpoints use the modern `Annotated[..., Depends(get_backend)]` injection pattern.

## 11. Frontend Surface (new)

A new top-level route `/eval` with three sub-views (managed by React Router):

- **Runs list** (`/eval`) — table of all runs, sortable by date / config / aggregated faithfulness / Recall@5 / cost. "New Run" button opens `NewEvalRunDialog`.
- **Run detail** (`/eval/runs/:id`) — top: aggregated metrics (bars with CI whiskers), per-stage latency chart, cost summary. Bottom: paginated per-question table — click row to see question, retrieved chunks, generated answer, gold answer, per-metric scores with judge reasoning.
- **Compare** (`/eval/compare?a=&b=`) — side-by-side aggregated bars with significance markers (★ for p<0.05); below, "Top regressions" and "Top wins" — the questions where the two runs diverged most.

Reuses existing TanStack Query, Tailwind, MarkdownRenderer, and shadcn/ui components. Charts via `recharts` (already a transitive React-friendly chart lib; if not present, add it — minimal bundle impact).

## 12. Error Handling

| Failure | Behavior |
|---------|----------|
| Pipeline raises on a single question | Log; mark that `EvalResult.error`; continue. |
| LLM-judge call fails after retry | Mark that metric `null` for the question; bootstrap drops null rows for that metric only. |
| Phoenix / OTel collector unreachable | OTel SDK is fail-quiet; eval and chat both proceed without traces. |
| Bootstrap on <30 samples | Still compute, but `RunMetadata.warnings` includes "low-N CI" flag; UI shows warning icon. |
| Two runs use different eval-set versions | `compare` endpoint refuses with 409, instructing to re-run on matched versions. |

Errors are logged via Python `logging` to stderr; the eval runner does not retry transient errors automatically beyond the per-judge single retry.

## 13. Testing

| Test file | Coverage |
|-----------|----------|
| `tests/test_eval_metrics_retrieval.py` | Hand-computed expected values for `recall_at_k`, `mrr_at_k`, `ndcg_at_k` on synthetic gold/retrieved sets including: perfect retrieval, missing gold, gold at rank N, ties. |
| `tests/test_eval_metrics_generation.py` | `answer_correctness` and `context_recall` with synthetic inputs; mocked LLM-judge returns. |
| `tests/test_eval_metrics_refusal.py` | Refusal detector regex on positive/negative phrasings; LLM-judge fallback path. |
| `tests/test_eval_statistics.py` | Bootstrap CI on a known-mean distribution converges within tolerance; permutation test p-value sanity (rejects on synthetic effect, accepts on null). |
| `tests/test_eval_runner_smoke.py` | 5-question synthetic eval set + dummy LLMHandler + ephemeral Chroma → end-to-end run produces valid `metrics.json` and `questions.jsonl`. |
| `tests/test_eval_compare.py` | Compare two synthetic runs; verify deltas and significance markers. |
| `tests/test_observability.py` | `@traced_stage` records expected attributes; failure to reach OTLP collector is non-fatal. |

All tests use deterministic seeds. No tests hit live LLM APIs.

## 14. File Inventory & Line Budgets (per CLAUDE.md ≤250 lines/file rule)

**New files (estimated lines):**

| File | Est. LoC | Responsibility |
|------|---------|----------------|
| `src/eval/__init__.py` | 5 | Exports |
| `src/eval/schemas.py` | 80 | Pydantic models from §7 |
| `src/eval/datasets/squad_v2.py` | 100 | Sample, ingest, freeze |
| `src/eval/datasets/ml_papers.py` | 80 | Load, verify, ingest |
| `src/eval/metrics/retrieval.py` | 90 | `recall_at_k`, `mrr_at_k`, `ndcg_at_k` |
| `src/eval/metrics/generation.py` | 120 | Wraps existing `evaluation.py`; adds `answer_correctness`, `context_recall` |
| `src/eval/metrics/refusal.py` | 70 | Regex + LLM-judge fallback |
| `src/eval/metrics/operational.py` | 80 | Aggregate timings/cost/tokens |
| `src/eval/statistics.py` | 100 | Bootstrap CI, paired permutation |
| `src/eval/pricing.py` | 60 | Model price table + cost calc |
| `src/eval/config.py` | 60 | `EvalConfig` YAML loader |
| `src/eval/runner.py` | 220 | Orchestrator (the big one — split if it grows) |
| `src/eval/compare.py` | 150 | Diff two runs, generate HTML |
| `src/eval/storage.py` | 100 | List/load/save runs |
| `src/eval/cli.py` | 120 | argparse subcommands |
| `src/api/routes/eval.py` | 200 | Routes from §10 |
| `src/observability.py` | 90 | OTel setup + decorator |
| `frontend/src/pages/EvalPage.tsx` | 150 | Route + sub-view dispatcher |
| `frontend/src/components/eval/RunsList.tsx` | 200 | Table, sort, filter, "New Run" |
| `frontend/src/components/eval/RunDetail.tsx` | 220 | Metrics chart + per-question table |
| `frontend/src/components/eval/CompareView.tsx` | 220 | Side-by-side bars + signif markers |
| `frontend/src/components/eval/NewEvalRunDialog.tsx` | 130 | Config picker + submit |
| `frontend/src/api/eval.ts` | 100 | API client + TanStack hooks |

**Modified files:**

| File | Change |
|------|--------|
| `src/backend.py` | Wrap `query` and `stream_query` with `@traced_stage`; emit telemetry payload. |
| `src/api/routes/query.py` | Send new `telemetry` WebSocket event after `done`. |
| `src/api/main.py` | Register `eval` router; init observability on startup. |
| `frontend/src/hooks/use-chat.ts` | Handle `telemetry` event; expose on message object. |
| `frontend/src/components/chat/chat-message.tsx` | Render telemetry footer. |
| `frontend/src/App.tsx` | Add `/eval/*` routes. |
| `frontend/src/components/layout/sidebar.tsx` | Add "Evaluation" nav link. |
| `requirements.txt` | Add `datasets`, `arize-phoenix`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `pyyaml`. |
| `frontend/package.json` | Add `recharts` if not already present. |
| `Architecture.md` | Add "Evaluation" section reflecting the new layer. |
| `docker-compose.yml` | Add `phoenix` service on port 6006 (optional, profile-gated). |

## 15. Educational Code Style — Phase 1 specifics

Per CLAUDE.md "Educational Code Style", every new module gets:

1. **Architecture header** placing it in the eval pipeline:
   ```python
   """
   Retrieval metrics — Recall@k, MRR@k, nDCG@k.

   Eval Harness Position:
     EvalRunner → Pipeline → [METRICS] ← gold_chunk_ids
                              ^^^^^^^^
     Pure functions over (gold_chunk_ids, retrieved_chunk_ids). No I/O,
     no LLM calls. Fast, deterministic, unit-testable in isolation.
   """
   ```
2. **WHY / PATTERN / TRADE-OFF / BUG-FIX comments** on non-obvious decisions, e.g. why bootstrap with paired resampling for two-run comparison, why nDCG uses log2 in the discount.
3. **Google-style docstrings** on all public functions with Args / Returns / Raises.
4. The labeling guide (`LABELING_GUIDE.md`) is itself an educational artifact — it documents the *thinking* behind the gold set, which is the rare skill reviewers reward.

## 16. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM-judge cost per eval run is non-trivial (~200 questions × ~5 judge calls). | Default `judge_model` is `gpt-4.1-mini`; document expected cost per run in CLI output before starting; cache judge outputs by (question_id, answer_hash, metric) so reruns of identical pipelines are free. |
| 50 hand-labeled Q/As take longer than expected. | Phase 1 can ship with SQuAD-200 only; ml_papers_v1 added incrementally. CLI supports running on either set. |
| Observability stack adds setup friction for new contributors. | Phoenix is optional (docker-compose profile); OTel SDK fails quiet; everything works without it. README documents the no-trace path. |
| `runner.py` exceeds the 250-line ceiling. | Split into `runner/orchestrator.py` + `runner/aggregator.py` if it grows past 220 lines during implementation. |
| Compare-two-runs UI requires charts not currently in the repo. | Add `recharts` (well-maintained, ~100KB gzipped, broadly used) — already justified in §11. |

## 17. Verification Plan

Before declaring Phase 1 complete:

1. `python -m pytest tests/ -v` — all green, including new eval tests.
2. `python -m src.eval.cli run --config configs/eval/baseline.yaml` — completes without errors, writes a valid run directory.
3. Manual smoke: open `/eval`, verify the run appears, open detail view, verify per-question rows render with all metric columns populated.
4. Manual smoke: launch two runs with different `top_k`, open `/eval/compare`, verify deltas are non-zero and significance markers render.
5. `docker compose --profile observability up` (new profile), open `localhost:6006`, run a query in the chat UI, verify trace appears with both spans.
6. Manual smoke: chat answer footer shows non-zero numbers for retrieve/generate/tokens/cost.
7. `Architecture.md` updated to reflect the new layer.

## 18. Open Questions (none blocking)

- Pricing table maintenance: hard-coded in `src/eval/pricing.py` for now (Phase 1). If model prices change frequently, revisit in Phase 2 to load from a JSON file or env override.
- Whether to expose `/api/eval/run` to unauthenticated callers in dev — currently the backend has no auth at all, so this matches existing posture; revisit when/if auth is added.

---

*End of Phase 1 design. After user review, the next step is invoking `superpowers:writing-plans` to produce a step-by-step implementation plan.*
