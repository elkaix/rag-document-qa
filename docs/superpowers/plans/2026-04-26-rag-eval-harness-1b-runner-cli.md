# RAG Eval Harness — Sub-plan 1B: Runner + Storage + CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Sub-plan 1A complete (eval engine modules in place).

**Goal:** Take the pure metric/dataset modules from 1A and wire them into a runnable, reproducible eval system. After this sub-plan, `python -m src.eval.cli run --config configs/eval/baseline.yaml` produces a complete run directory; `python -m src.eval.cli compare A B` produces a diff with significance markers; `python -m src.eval.cli list` enumerates past runs.

**Architecture:** A `EvalConfig` Pydantic model loaded from YAML, an `EvalRunner` that owns the full run lifecycle (config → ingest → loop questions → score → write artifacts), a `storage` layer that reads/writes `eval_runs/<run_id>/`, a `compare` module for two-run diffs, and an argparse `cli` that ties them together. Pipeline-under-test is built fresh per run from the config so two runs use isolated Chroma collections.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML, argparse, jinja2 (HTML report), existing RAGBackend / ChromaVectorStore / TextChunker / DocumentLoader / LLMHandler.

**Spec:** [`docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md`](../specs/2026-04-26-rag-eval-harness-phase-1-design.md)

---

## File Structure

**New files:**

| Path | Responsibility |
|------|----------------|
| `src/eval/config.py` | `EvalConfig` Pydantic model + `load_config(path)` YAML loader. |
| `src/eval/storage.py` | `list_runs`, `load_run`, `save_run`, `compute_run_id`. |
| `src/eval/pipeline_factory.py` | Build a RAG pipeline (chunker + Chroma collection + retriever + LLM) from an `EvalConfig.pipeline` block. |
| `src/eval/runner.py` | `EvalRunner.run(config) -> RunMetadata`. Orchestrates ingest → loop → score → persist. |
| `src/eval/aggregator.py` | Take `list[EvalResult]` → `list[AggregatedMetric]` (per-dataset + combined) using bootstrap. |
| `src/eval/compare.py` | `compare_runs(id_a, id_b) -> CompareResult`; per-question-diff selector. |
| `src/eval/report.py` | `render_html(run | compare) -> str` via jinja2; standalone HTML page. |
| `src/eval/cli.py` | argparse subcommands: `run`, `list`, `show`, `compare`. |
| `configs/eval/baseline.yaml` | The default config matching current production defaults. |
| `configs/eval/README.md` | How to author new configs. |
| `templates/eval/run_report.html.j2` | Jinja2 template for single-run HTML report. |
| `templates/eval/compare_report.html.j2` | Jinja2 template for two-run compare HTML report. |

**Modified files:**

| Path | Change |
|------|--------|
| `requirements.txt` | Add `pyyaml>=6` and `jinja2>=3`. |
| `src/eval/__init__.py` | Re-export `EvalConfig`, `EvalRunner`, `compare_runs`, `list_runs`, `load_run`. |

**Tests:**
`tests/test_eval_config.py`, `test_eval_storage.py`, `test_eval_pipeline_factory.py`, `test_eval_runner.py`, `test_eval_aggregator.py`, `test_eval_compare.py`, `test_eval_cli.py`, `test_eval_report.py`.

---

## Test Doubles

Two harnesses used across runner / cli tests:

```python
class DummyLLM:
    """Returns deterministic JSON for any judge prompt; '<answer>' for generations."""
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if "JSON" in (system_prompt or "") or "JSON" in prompt:
            return json.dumps({"score": 1.0, "claims": [], "chunks": [],
                               "factual_match": 1.0, "is_refusal": False,
                               "reasoning": "ok"})
        return "<dummy answer>"

def ephemeral_chroma() -> ChromaVectorStore:
    """EphemeralClient + a unique collection name; auto-cleaned per test."""
```

`tests/conftest.py` is extended with two fixtures: `dummy_llm` (DummyLLM) and `tmp_eval_runs` (creates `tmp_path/eval_runs/`, sets it as the `EVAL_RUNS_DIR` env var for the duration of a test).

---

## Task 1 — `requirements.txt` deps + jinja2 templates dir

**Files:** `requirements.txt`, `templates/eval/.gitkeep`

- [ ] Append `pyyaml>=6` and `jinja2>=3` to `requirements.txt`.
- [ ] `mkdir -p templates/eval && touch templates/eval/.gitkeep`.
- [ ] `pip install -r requirements.txt`; verify both import.
- [ ] Commit: `chore(eval): add pyyaml and jinja2 deps`.

---

## Task 2 — `src/eval/config.py`

**Files:** `tests/test_eval_config.py`, `src/eval/config.py`

**`EvalConfig` model** (matches spec §7 YAML):

```python
class ChunkerCfg(BaseModel):
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

class RetrieverCfg(BaseModel):
    top_k: int = 5

class GeneratorCfg(BaseModel):
    model: str = "gpt-5-mini"
    reasoning_model: str | None = "gpt-4.1-nano"

class PipelineCfg(BaseModel):
    chunker: ChunkerCfg
    retriever: RetrieverCfg
    generator: GeneratorCfg

class EvalCfg(BaseModel):
    datasets: list[Literal["squad_v2_dev_200", "ml_papers_v1"]]
    judge_model: str = "gpt-4.1-mini"
    bootstrap_n: int = 1000
    permutation_n: int = 10000
    seed: int = 42

class EvalConfig(BaseModel):
    name: str
    description: str = ""
    pipeline: PipelineCfg
    eval: EvalCfg
```

**API:** `load_config(path: Path) -> EvalConfig` — reads YAML via `yaml.safe_load`, validates via `EvalConfig.model_validate`. Bad YAML or schema raises with descriptive message.

**Test cases:**
- Round-trip: `EvalConfig(...).model_dump()` → write YAML → reload → equal.
- Missing required field raises `ValidationError`.
- Unknown dataset name raises (Literal validation).
- Defaults applied when omitted.

`configs/eval/baseline.yaml` is created in this task with the spec §7 contents.

Commit: `feat(eval): add EvalConfig YAML loader and baseline config`.

---

## Task 3 — `src/eval/storage.py`

**Files:** `tests/test_eval_storage.py`, `src/eval/storage.py`

**API:**
- `EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", "eval_runs"))` — overridable for tests.
- `compute_run_id(config_name: str, started_at: datetime, git_sha: str) -> str` → `YYYY-MM-DD_HHMMSS_<config-name>_<sha7>`.
- `save_run(run_dir, metadata, results, aggregated, config_yaml_text) -> None` — writes `metadata.json`, `questions.jsonl` (one EvalResult per line), `metrics.json` (list[AggregatedMetric]), `cost.json` (dict from `aggregate_costs/tokens`), `config.yaml` (raw text passed in).
- `load_run(run_id) -> dict` → `{metadata: RunMetadata, results: list[EvalResult], aggregated: list[AggregatedMetric], cost: dict}`.
- `list_runs() -> list[RunMetadata]` — scans `EVAL_RUNS_DIR/` for subdirs containing `metadata.json`, sorted by `started_at` descending.
- `delete_run(run_id) -> None` — `shutil.rmtree`; refuses to delete outside `EVAL_RUNS_DIR`.

**Test cases:**
- `compute_run_id` is deterministic for given inputs; sha truncated to 7.
- `save_run` then `load_run` round-trips identical data.
- `list_runs` returns sorted-descending by start time; ignores subdirs without metadata.json.
- `delete_run` removes the dir; raises if `run_id` contains `..` or `/`.

Commit: `feat(eval): add storage layer for eval run directories`.

---

## Task 4 — `src/eval/pipeline_factory.py`

**Files:** `tests/test_eval_pipeline_factory.py`, `src/eval/pipeline_factory.py`

**API:** `build_pipeline(config: EvalConfig, dataset_name: str, dummy_llm: object | None = None) -> EvalPipeline` where `EvalPipeline` is a small dataclass:

```python
@dataclass
class EvalPipeline:
    chunker: TextChunker
    vector_store: ChromaVectorStore   # ephemeral collection per (config, dataset)
    llm: LLMHandler                    # or dummy if injected
    judge_llm: LLMHandler              # always built from config.eval.judge_model
    config: EvalConfig
    dataset_name: str

    def ingest(self, questions: list[EvalQuestion]) -> None: ...
    def query(self, question: str) -> tuple[list[Chunk], str, dict]:
        """Returns (retrieved_chunks, answer, telemetry).
        telemetry = {timings_ms, tokens, cost_usd}."""
    def teardown(self) -> None: ...
```

**Implementation notes:**
- `vector_store` uses `chromadb.EphemeralClient()` + collection name `eval_<config_name>_<dataset_name>_<random6>` so two runs in flight don't collide.
- `ingest`:
  - For `squad_v2_dev_200`: each question's `metadata["context"]` becomes one Chroma document with `id == question.id` (which is also the gold_chunk_id).
  - For `ml_papers_v1`: load `corpus_manifest.json`, run each pinned PDF through `DocumentLoader` + `TextChunker`, upsert chunks. Gold-chunk-ids in questions.jsonl must match the chunk IDs the chunker produces (this is enforced by labeling guide).
- `query`: instrument with `time.perf_counter()` around each stage; estimate token counts using `tiktoken` if available, else word-count×1.3 heuristic; compute cost via `cost_usd`.

**Test cases (with DummyLLM + ephemeral Chroma):**
- `build_pipeline` with `squad_v2_dev_200` → `ingest([q1, q2, q3])` works; `query(q1.question)` returns chunks and a non-empty answer.
- Telemetry has all expected keys with non-negative values.
- `teardown` deletes the ephemeral collection.

Commit: `feat(eval): add EvalPipeline factory bridging EvalConfig to RAG components`.

---

## Task 5 — `src/eval/aggregator.py`

**Files:** `tests/test_eval_aggregator.py`, `src/eval/aggregator.py`

**API:** `aggregate(results: list[EvalResult], config: EvalConfig) -> list[AggregatedMetric]`

**Behavior:**
- For each `metric_name` present in any `r.metrics`:
  - Per-dataset row: collect values where `r.dataset == d` and metric present, run `bootstrap_ci(values, config.eval.bootstrap_n, config.eval.seed)`, emit `AggregatedMetric(metric_name=m, dataset=d, mean=..., ci_low=..., ci_high=..., n=len(values))`.
  - Combined row: same but across all datasets, `dataset=None`.
- Skip metric/dataset combos with <3 non-NaN samples; record a warning string the runner appends to `RunMetadata.warnings`.

**Test cases:**
- 30 synthetic results across 2 datasets, 2 metrics → 6 aggregated rows (2 metrics × 3: dataset_a, dataset_b, combined).
- Metric with 1 sample → skipped + warning emitted.

Commit: `feat(eval): add metric aggregator with per-dataset and combined rows`.

---

## Task 6 — `src/eval/runner.py`

**Files:** `tests/test_eval_runner.py`, `src/eval/runner.py`

**API:**
```python
class EvalRunner:
    def __init__(self, config: EvalConfig, *, llm_factory=None, storage=None): ...
    def run(self) -> RunMetadata: ...
```

**Lifecycle:**
1. `started_at = datetime.now(UTC)`; collect `git_sha` from `git rev-parse HEAD`; compute `env_hash` from `requirements.txt` SHA-256; compute `run_id`.
2. For each dataset in `config.eval.datasets`:
   - Load questions (via 1A loaders).
   - Build pipeline (`pipeline_factory.build_pipeline`); `pipeline.ingest(questions)`.
   - For each question: `pipeline.query(...)` → compute all applicable metrics (retrieval if `gold_chunk_ids`; generation if `gold_answer`; refusal always; LLM-judge faithfulness/relevancy/context-precision; context-recall) → assemble `EvalResult`. Wrap in `try/except` so a single question failure marks `result.error` and proceeds.
   - `pipeline.teardown()`.
3. `aggregated = aggregate(all_results, config)`.
4. `cost = aggregate_costs(all_results) | aggregate_tokens(all_results)`.
5. `metadata.finished_at = now`; `n_errors = sum(1 for r in results if r.error)`.
6. `storage.save_run(run_dir, metadata, all_results, aggregated, config_yaml_text)`.
7. Return `metadata`.

**Test cases (using DummyLLM + ephemeral Chroma + 5-question synthetic squad slice):**
- End-to-end run completes without errors; produces non-empty `metrics.json` and 5-line `questions.jsonl`.
- A pipeline-raising question does not abort the run; `n_errors == 1`.
- Same config + same DummyLLM + fixed seed → bit-identical `metrics.json` (modulo timing fields).

Commit: `feat(eval): add EvalRunner orchestrator end-to-end`.

---

## Task 7 — `src/eval/compare.py`

**Files:** `tests/test_eval_compare.py`, `src/eval/compare.py`

**API:** `compare_runs(id_a: str, id_b: str) -> CompareResult`.

**Algorithm:**
1. `load_run(id_a)` and `load_run(id_b)`.
2. Validate `eval_set_versions` match; raise `ValueError("eval set version mismatch")` otherwise.
3. For each `(metric_name, dataset)` present in BOTH aggregated lists:
   - Pair per-question values by `question_id`; intersect IDs that appear in both runs.
   - Run `paired_permutation_test(a_values, b_values, n_resamples=10000, seed=42)`.
   - Build `MetricDelta` with `a_mean`/`a_ci` from A's `AggregatedMetric` and same for B; `delta = b_mean - a_mean`; `significant = p_value < 0.05`.
4. `per_question_diff`: for the chosen "headline" metric (`recall_at_5` if present, else first metric alphabetically), pick the 10 questions with the largest `|b - a|` and emit `{question_id, question, gold_answer, a_answer, a_score, b_answer, b_score, delta}`.
5. Return `CompareResult`.

**Test cases:**
- Two runs from synthetic data (one with constant +0.1 metric shift) → all `MetricDelta.delta ≈ 0.1`, all `significant=True`.
- Mismatched eval-set versions → raises `ValueError`.
- `per_question_diff` is sorted by `|delta|` descending and capped at 10.

Commit: `feat(eval): add two-run comparison with paired significance tests`.

---

## Task 8 — `src/eval/report.py`

**Files:** `tests/test_eval_report.py`, `src/eval/report.py`, `templates/eval/run_report.html.j2`, `templates/eval/compare_report.html.j2`

**API:**
- `render_run_html(run: dict) -> str` — single-run report: header (run metadata), aggregated metrics table with CI bars, cost summary, per-question table (paginated client-side).
- `render_compare_html(compare: CompareResult) -> str` — side-by-side aggregated bars, top regressions, top wins.

**Templates:** Use jinja2 `Environment(loader=FileSystemLoader("templates/eval"))`. Both templates are standalone (inline minimal CSS, no external assets) so the file is portable.

**Test cases:**
- `render_run_html` on synthetic data returns a string containing `<table`, the run_id, and each metric_name.
- `render_compare_html` contains both run IDs and a `★` marker on at least one significant delta.

Commit: `feat(eval): add jinja2-based HTML report renderer for runs and comparisons`.

---

## Task 9 — `src/eval/cli.py`

**Files:** `tests/test_eval_cli.py`, `src/eval/cli.py`

**Subcommands:**
- `python -m src.eval.cli run --config <path>` → invokes `EvalRunner(load_config(path)).run()`; prints `run_id` and a one-line summary on completion.
- `python -m src.eval.cli list` → prints a table of all runs (run_id, started_at, config_name, n_questions, n_errors, headline metric).
- `python -m src.eval.cli show <run_id>` → prints aggregated metrics; with `--html` flag, generates `report.html` in the run dir and prints its path.
- `python -m src.eval.cli compare <id_a> <id_b>` → prints delta table; with `--html` flag, generates `compare_<a>_<b>.html`.

**Implementation:** standard argparse subparsers; main module guard runs `cli.main()`.

**Test cases (use `subprocess.run` with the test fixtures for ephemeral storage):**
- `run --config <baseline path>` with DummyLLM env override produces a run dir with all expected files.
- `list` prints the just-created run.
- `show <id>` prints non-empty aggregated rows.
- `compare A B` after two runs prints a delta table with at least one row.

Commit: `feat(eval): add argparse CLI for run/list/show/compare`.

---

## Task 10 — Update `src/eval/__init__.py` exports + integration smoke test

**Files:** `src/eval/__init__.py` (modify), `tests/test_eval_integration.py` (new)

- Re-export new symbols: `EvalConfig`, `load_config`, `EvalRunner`, `compare_runs`, `list_runs`, `load_run`, `save_run`.
- `tests/test_eval_integration.py`: full end-to-end test — load `configs/eval/baseline.yaml`, run with DummyLLM on a 5-question SQuAD slice, verify run dir + metrics, then run a SECOND run with `top_k=10` override, then `compare_runs` returns a populated `CompareResult`.

Commit: `feat(eval): wire 1B exports and add end-to-end integration smoke test`.

---

## Sub-plan 1B Completion Checklist

- [ ] All `tests/test_eval_*.py` pass.
- [ ] `python -m src.eval.cli run --config configs/eval/baseline.yaml` completes on a real (non-dummy) backend, producing a run dir with all five artifacts.
- [ ] `python -m src.eval.cli list` shows the run.
- [ ] `python -m src.eval.cli compare <A> <B>` produces a delta table.
- [ ] `report.html` opens cleanly in a browser.
- [ ] No file in `src/eval/` exceeds 250 lines (split `runner.py` into `runner/orchestrator.py` + `runner/aggregator.py` if it grows past 220 during implementation, as flagged in spec §16).
- [ ] `git status` clean.

Once green: proceed to **Sub-plan 1C — API + frontend**.
