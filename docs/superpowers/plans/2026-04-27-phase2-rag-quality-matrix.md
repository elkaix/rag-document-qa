# Phase 2 RAG Quality Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a layered RAG ablation matrix that measures the lift each major architectural lever buys on top of the Phase 1 SQuAD-200 baseline, then ship a portfolio writeup attributing the lift to mechanism.

**Architecture:** Two PRs stacked on `feature/eval-harness-1d`. PR-A adds 5 new pipeline modules (`BgeEmbedder`, `BM25HybridRetriever`, `CrossEncoderReranker`, `QueryRewriter`, `RefusalHandler`), extends `EvalConfig.pipeline` with backward-compatible sub-configs, refactors the cost ledger to cover all three LLM call sites (generator + judge + rewriter), and wires everything into `src/eval/pipeline_factory.py::build_pipeline`. PR-B executes 8 distinct evals against SQuAD-200, archives the small artifacts to `docs/phase2/runs/`, and ships `docs/PHASE2_RESULTS.md`.

**Tech Stack:** Python 3.12 + Pydantic v2 + ChromaDB EphemeralClient + sentence-transformers (BGE-small + ms-marco-MiniLM cross-encoder) + rank-bm25 + pytest + the existing eval harness from Phase 1. Frontend untouched.

**Spec source of truth:** [`docs/superpowers/specs/2026-04-27-phase2-rag-quality-matrix-design.md`](../specs/2026-04-27-phase2-rag-quality-matrix-design.md)

---

## File Structure

### New files (PR-A)

| Path | Responsibility |
|------|----------------|
| `src/eval/embedders/__init__.py` | Re-export `BgeEmbedder`. |
| `src/eval/embedders/bge_small.py` | Chroma `EmbeddingFunction` adapter loading `BAAI/bge-small-en-v1.5` via `sentence-transformers`. |
| `src/eval/retrievers/__init__.py` | Re-export `BM25HybridRetriever`, `CrossEncoderReranker`. |
| `src/eval/retrievers/bm25_hybrid.py` | BM25 + dense retriever with Reciprocal Rank Fusion. Wraps a `ChromaVectorStore`. |
| `src/eval/retrievers/reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` post-retrieval reranker. |
| `src/eval/transforms/__init__.py` | Re-export `QueryRewriter`, `RefusalHandler`. |
| `src/eval/transforms/query_rewriter.py` | LLM-based query expansion; returns answer text + token usage. |
| `src/eval/transforms/refusal_handler.py` | Pure-logic similarity gate; returns refusal text on low confidence. |
| `tests/test_eval_embedder_bge.py` | Unit + Chroma-integration tests for `BgeEmbedder`. |
| `tests/test_eval_retriever_bm25_hybrid.py` | RRF unit test + retrieval integration. |
| `tests/test_eval_retriever_reranker.py` | Cross-encoder rerank order. |
| `tests/test_eval_transform_rewriter.py` | Pass-through + stubbed-LLM expansion. |
| `tests/test_eval_transform_refusal.py` | Threshold gate + empty-candidates edge. |
| `tests/test_eval_cost_ledger.py` | Aggregator sums generator + judge + rewriter spend. |
| `tests/test_eval_cli_archive.py` | `cli archive` copies the four small artifacts only. |
| `tests/test_eval_pipeline_factory_phase2.py` | Each Phase 2 YAML produces a pipeline whose attributes match the tier toggles. |
| `tests/fixtures/phase2_corpus/*.txt` | Three tiny text docs for smoke + integration tests. |
| `configs/eval/phase2/phase2_baseline.yaml` | Baseline re-run anchor. |
| `configs/eval/phase2/phase2b_embedder.yaml` | + BGE embedder. |
| `configs/eval/phase2/phase2c_hybrid.yaml` | + BM25 hybrid. |
| `configs/eval/phase2/phase2d_rerank.yaml` | + cross-encoder rerank. |
| `configs/eval/phase2/phase2e_rewrite.yaml` | + query rewriting. |
| `configs/eval/phase2/phase2g_refusal.yaml` | + refusal handler. |
| `configs/eval/phase2/phase2f_models_gpt5mini.yaml` | 2g stack with `gpt-5-mini` (re-uses 2g artifact). |
| `configs/eval/phase2/phase2f_models_gpt41mini.yaml` | 2g stack with `gpt-4.1-mini`. |
| `configs/eval/phase2/phase2f_models_haiku.yaml` | 2g stack with `claude-haiku-4-5`. |

### Modified files (PR-A)

| Path | Change |
|------|--------|
| `src/eval/config.py` | Add 5 new sub-config models + `EvalCfg.spend_ceiling_usd`; extend `PipelineCfg`. |
| `src/eval/schemas.py` | Add `EvalResult.cost_breakdown: dict[str, float]`. |
| `src/eval/pricing.py` | Add `gpt-4.1-nano`, `claude-haiku-4-5` to `MODEL_PRICES` (verify `gpt-4.1-mini` present). |
| `src/eval/metrics/generation.py` | Judge functions return `(score, details, prompt_tokens, completion_tokens, cost_usd)`. |
| `src/eval/runner.py` | `_score_question` collects judge usage; `_query_one` collects rewriter usage; spend-ceiling enforcement. |
| `src/eval/pipeline_factory.py` | `build_pipeline` builds embedder/hybrid/reranker/rewriter/refusal; `EvalPipeline.query` invokes them. |
| `src/llm_handler.py` | Add `generate_with_usage(prompt, system_prompt) -> tuple[str, int, int]`. |
| `src/eval/cli.py` | Add `archive` subcommand. |
| `src/eval/aggregator.py` | Sum `cost_breakdown` into `cost.json` totals. |
| `requirements.txt` | + `rank-bm25`. |

### New files (PR-B)

| Path | Responsibility |
|------|----------------|
| `docs/phase2/runs/<short_id>_phase2_*/{metrics,cost,metadata,config}.json` | Archived small artifacts. |
| `docs/phase2/compare/*.html` | Pairwise compare reports (5 chain + 2 model). |
| `docs/PHASE2_RESULTS.md` | Methodology, chart, significance table, findings. |

### Modified files (PR-B)

| Path | Change |
|------|--------|
| `README.md` | Link to `docs/PHASE2_RESULTS.md`. |

---

## Branching

```bash
git checkout feature/eval-harness-1d
git pull origin feature/eval-harness-1d
git checkout -b feature/phase2-pipeline-extensions
```

PR-A targets `feature/eval-harness-1d`. After Phase 1 merges, retarget PR-A to `main`. PR-B branches from PR-A's head once PR-A is reviewable.

---

# PR-A — Pipeline Extensions

## Task 1: Extend `PipelineCfg` schema with Phase 2 sub-configs

**Files:**
- Modify: `src/eval/config.py`
- Modify: `tests/test_eval_config.py` (existing test file)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_config.py`:

```python
# --- Phase 2 schema additions ----------------------------------------------

import pytest
from pydantic import ValidationError
from src.eval.config import (
    EvalConfig, EmbedderCfg, HybridCfg, RerankerCfg,
    QueryRewriterCfg, RefusalHandlerCfg,
)

def test_phase2_subconfigs_default_to_off(tmp_path):
    """Loading an existing baseline-shape YAML must produce all-default Phase 2 blocks."""
    yaml_text = """
name: legacy_baseline
description: existing config without phase 2 blocks
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    cfg = load_config(p)
    # All new blocks present with defaults
    assert cfg.pipeline.embedder.name == "chroma_default"
    assert cfg.pipeline.hybrid.enabled is False
    assert cfg.pipeline.reranker.model is None
    assert cfg.pipeline.query_rewriter.model is None
    assert cfg.pipeline.refusal_handler.enabled is False
    assert cfg.eval.spend_ceiling_usd is None

def test_phase2_subconfig_typed_values(tmp_path):
    """Phase 2 fields validate to the right types."""
    yaml_text = """
name: phase2g
description: refusal handler enabled
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  embedder: {name: bge_small_en_v1_5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  query_rewriter: {model: gpt-4.1-nano, max_expansions: 3}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
  refusal_handler: {enabled: true, similarity_threshold: 0.35}
eval:
  datasets: [squad_v2_dev_200]
  spend_ceiling_usd: 1.5
"""
    p = tmp_path / "phase2g.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    cfg = load_config(p)
    assert cfg.pipeline.embedder.name == "bge_small_en_v1_5"
    assert cfg.pipeline.hybrid.enabled is True
    assert cfg.pipeline.reranker.model == "ms_marco_minilm_l6_v2"
    assert cfg.pipeline.query_rewriter.model == "gpt-4.1-nano"
    assert cfg.pipeline.refusal_handler.similarity_threshold == 0.35
    assert cfg.eval.spend_ceiling_usd == 1.5

def test_phase2_unknown_field_rejected(tmp_path):
    """extra='forbid' must reject unknown keys at load time."""
    yaml_text = """
name: bad
description: typo in field name
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  hybrid: {enabld: true}    # typo on purpose
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    with pytest.raises(ValidationError):
        load_config(p)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_config.py::test_phase2_subconfigs_default_to_off \
       tests/test_eval_config.py::test_phase2_subconfig_typed_values \
       tests/test_eval_config.py::test_phase2_unknown_field_rejected -v
```
Expected: FAIL with `ImportError` for `EmbedderCfg`/etc.

- [ ] **Step 3: Implement the schema extension**

Insert into `src/eval/config.py` immediately before the `class PipelineCfg` definition:

```python
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
```

Replace the existing `class PipelineCfg` block with:

```python
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
```

In the same file, locate `class EvalCfg(BaseModel):` and append `spend_ceiling_usd: float | None = None` after the existing `seed: int = 42` line:

```python
    seed: int = 42
    # Phase 2: hard per-run spend ceiling. EvalRunner aborts when the cumulative
    # generator + judge + rewriter cost crosses this. None disables the guard.
    spend_ceiling_usd: float | None = None
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_eval_config.py -v
```
Expected: all tests PASS, including the existing baseline tests.

- [ ] **Step 5: Commit**

```bash
git add src/eval/config.py tests/test_eval_config.py
git commit -m "feat(eval): extend PipelineCfg with Phase 2 sub-configs and spend ceiling"
```

---

## Task 2: Cost ledger covers generator + judge + rewriter

**Files:**
- Modify: `src/eval/schemas.py`
- Modify: `src/eval/metrics/generation.py`
- Modify: `src/eval/runner.py`
- Modify: `src/eval/aggregator.py`
- Modify: `src/llm_handler.py`
- Modify: `src/eval/pricing.py`
- Create: `tests/test_eval_cost_ledger.py`
- Modify: `tests/test_eval_runner.py` (existing — extend, don't break)

- [ ] **Step 1: Write failing tests for the new cost ledger surface**

Create `tests/test_eval_cost_ledger.py`:

```python
"""Tests for the Phase 2 cost ledger covering generator + judge + rewriter spend."""

from __future__ import annotations

from src.eval.schemas import EvalResult


def test_eval_result_has_cost_breakdown_field():
    """EvalResult must carry a cost_breakdown dict with per-bucket spend."""
    r = EvalResult(
        question_id="q1",
        dataset="squad_v2_dev_200",
        retrieved_chunk_ids=[],
        retrieved_chunks=[],
        generated_answer="",
        metrics={},
        timings_ms={},
        tokens={},
        cost_usd=0.0,
        cost_breakdown={"generator": 0.0, "judge": 0.0, "rewriter": 0.0},
    )
    assert r.cost_breakdown["generator"] == 0.0
    assert r.cost_breakdown["judge"] == 0.0
    assert r.cost_breakdown["rewriter"] == 0.0


def test_eval_result_cost_breakdown_defaults():
    """cost_breakdown must default to a generator-only dict when omitted."""
    r = EvalResult(
        question_id="q1",
        dataset="squad_v2_dev_200",
        retrieved_chunk_ids=[],
        retrieved_chunks=[],
        generated_answer="",
        metrics={},
        timings_ms={},
        tokens={},
        cost_usd=0.05,
    )
    # back-compat default: existing records read as generator-only
    assert r.cost_breakdown == {"generator": 0.05, "judge": 0.0, "rewriter": 0.0}


def test_aggregator_sums_cost_breakdown_into_totals():
    """aggregate_costs must surface per-bucket totals alongside total_usd."""
    from src.eval.metrics.operational import aggregate_costs

    results = [
        EvalResult(
            question_id="q1", dataset="d", retrieved_chunk_ids=[], retrieved_chunks=[],
            generated_answer="", metrics={}, timings_ms={}, tokens={},
            cost_usd=0.10,
            cost_breakdown={"generator": 0.04, "judge": 0.05, "rewriter": 0.01},
        ),
        EvalResult(
            question_id="q2", dataset="d", retrieved_chunk_ids=[], retrieved_chunks=[],
            generated_answer="", metrics={}, timings_ms={}, tokens={},
            cost_usd=0.20,
            cost_breakdown={"generator": 0.08, "judge": 0.10, "rewriter": 0.02},
        ),
    ]
    summary = aggregate_costs(results)
    assert round(summary["total_usd"], 4) == 0.30
    assert round(summary["generator_total_usd"], 4) == 0.12
    assert round(summary["judge_total_usd"], 4) == 0.15
    assert round(summary["rewriter_total_usd"], 4) == 0.03


def test_llm_handler_generate_with_usage_returns_tokens():
    """LLMHandler.generate_with_usage must return (text, prompt_tokens, completion_tokens)."""
    from src.llm_handler import LLMHandler

    # Use the dummy fallback path — no API key needed.
    handler = LLMHandler("__dummy__")
    text, prompt_tokens, completion_tokens = handler.generate_with_usage(
        prompt="What is RAG?", system_prompt="Be brief."
    )
    assert isinstance(text, str)
    assert isinstance(prompt_tokens, int) and prompt_tokens > 0
    assert isinstance(completion_tokens, int) and completion_tokens >= 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_eval_cost_ledger.py -v
```
Expected: FAIL — `cost_breakdown` field missing, `aggregate_costs` missing the new keys, `generate_with_usage` undefined.

- [ ] **Step 3: Add `cost_breakdown` to `EvalResult`**

In `src/eval/schemas.py`, locate `class EvalResult` and add after the `cost_usd: float` line:

```python
    cost_usd: float
    # Phase 2: per-bucket breakdown. Defaults to generator-only when absent so
    # Phase 1 records continue to round-trip through model_validate.
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def _backfill_cost_breakdown(self) -> "EvalResult":
        if not self.cost_breakdown:
            object.__setattr__(self, "cost_breakdown", {
                "generator": self.cost_usd,
                "judge": 0.0,
                "rewriter": 0.0,
            })
        return self
```

Add `from pydantic import model_validator` to the imports at the top of the file if not already present.

- [ ] **Step 4: Add `generate_with_usage` to `LLMHandler`**

In `src/llm_handler.py`, immediately after the existing `def generate` method (around line 162), insert:

```python
    def generate_with_usage(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response and return text plus prompt/completion token counts.

        WHY a separate method: the existing `generate()` returns only `str` and is
        called in many places that don't need usage. Phase 2's cost ledger needs
        token counts on every LLM call; rather than break callers, we add a parallel
        method that uses the existing tokenizer to estimate counts client-side.

        Args:
            prompt: User message.
            system_prompt: Optional system instructions.

        Returns:
            (response_text, prompt_tokens, completion_tokens).
        """
        from src.eval._telemetry import count_tokens

        text = self.generate(prompt, system_prompt=system_prompt)
        full_prompt = (system_prompt + "\n" + prompt) if system_prompt else prompt
        prompt_tokens = count_tokens(full_prompt, self._model)
        completion_tokens = count_tokens(text, self._model)
        return text, prompt_tokens, completion_tokens
```

(Verify the attribute is `self._model` by reading the class init; if it's `self.model`, adjust accordingly.)

- [ ] **Step 5: Extend `aggregate_costs` in `src/eval/metrics/operational.py`**

Locate the existing `def aggregate_costs(results: list[EvalResult])`. Add the per-bucket totals before returning the summary dict:

```python
def aggregate_costs(results: list[EvalResult]) -> dict[str, float]:
    """Sum cost_usd and per-bucket breakdown across all results."""
    total = sum(r.cost_usd for r in results)
    n = len(results) if results else 1
    generator_total = sum(r.cost_breakdown.get("generator", 0.0) for r in results)
    judge_total = sum(r.cost_breakdown.get("judge", 0.0) for r in results)
    rewriter_total = sum(r.cost_breakdown.get("rewriter", 0.0) for r in results)
    return {
        "total_usd": total,
        "mean_usd_per_query": total / n,
        "generator_total_usd": generator_total,
        "judge_total_usd": judge_total,
        "rewriter_total_usd": rewriter_total,
    }
```

(Preserve the existing `total_prompt`/`total_completion` keys if `aggregate_tokens` lives in the same function; if separate, keep them untouched.)

- [ ] **Step 6: Pricing — register Phase 2 models**

In `src/eval/pricing.py`, locate the `MODEL_PRICES` dict and add (verify which are missing first):

```python
    # Phase 2 additions
    "claude-haiku-4-5": ModelPrice(prompt_per_1m=1.0, completion_per_1m=5.0),
    "gpt-4.1-nano": ModelPrice(prompt_per_1m=0.10, completion_per_1m=0.40),
    # gpt-4.1-mini and gpt-5-mini should already be present
```

If `gpt-4.1-mini` or `gpt-5-mini` are missing, add them too with current published rates (consult OpenAI's pricing page; document the source in a comment).

- [ ] **Step 7: Wire judge cost capture into `_score_question`**

In `src/eval/metrics/generation.py`, change the judge functions to return cost. Take `_judge_factual_match` as the model:

```python
def _judge_factual_match(
    generated: str,
    gold: str,
    llm,
) -> tuple[float, str, int, int]:
    """Score factual agreement and return (score, reasoning, prompt_tokens, completion_tokens)."""
    # ... existing prompt construction unchanged ...
    raw, prompt_tokens, completion_tokens = llm.generate_with_usage(
        user_prompt, system_prompt=system_prompt,
    )
    # ... existing JSON parsing unchanged, returning (score, reasoning, prompt_tokens, completion_tokens)
    try:
        parsed = json.loads(stripped)
        score = max(0.0, min(1.0, float(parsed["factual_match"])))
        reasoning = str(parsed.get("reasoning", ""))
        return score, reasoning, prompt_tokens, completion_tokens
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0, "Judge returned malformed JSON; defaulting to 0.0", prompt_tokens, completion_tokens
```

Apply the same pattern to `judge_faithfulness`, `judge_answer_relevancy`, `judge_context_precision`, and the LLM call inside `answer_correctness`. Each returns `(score, details, prompt_tokens, completion_tokens)` where `details` is the existing dict.

In `src/eval/runner.py`, in `_score_question`, accumulate judge tokens/cost into a local variable and write into `metric_details["judge_cost_usd"]` and `metric_details["judge_tokens"]`. Return them so `_query_one` can write them into `EvalResult.cost_breakdown`. Sketch:

```python
def _score_question(question, chunks, answer, judge_llm) -> tuple[dict, dict, dict]:
    """Returns (metrics, metric_details, judge_usage) where judge_usage =
    {'cost_usd': float, 'prompt_tokens': int, 'completion_tokens': int}."""
    # ... existing scoring loop, accumulating tokens/cost from each judge call ...
    judge_prompt_tokens = 0
    judge_completion_tokens = 0
    judge_cost = 0.0
    judge_model = ...   # read from runner config
    # for each judge call:
    score, details, p_t, c_t = judge_function(...)
    judge_prompt_tokens += p_t
    judge_completion_tokens += c_t
    judge_cost += pricing.cost_usd(judge_model, p_t, c_t)
    # ...
    return metrics, metric_details, {
        "cost_usd": judge_cost,
        "prompt_tokens": judge_prompt_tokens,
        "completion_tokens": judge_completion_tokens,
    }
```

In `_query_one`, after the existing telemetry assembly, add judge usage to the breakdown:

```python
metrics, metric_details, judge_usage = _score_question(question, chunks, answer, judge_llm)
generator_cost = telemetry["cost_usd"]
return EvalResult(
    ...,
    cost_usd=generator_cost + judge_usage["cost_usd"],
    cost_breakdown={
        "generator": generator_cost,
        "judge": judge_usage["cost_usd"],
        "rewriter": telemetry.get("rewriter_cost_usd", 0.0),
    },
    ...,
)
```

(`rewriter_cost_usd` defaults to 0.0 here; Task 6 wires it in when the rewriter is enabled.)

- [ ] **Step 8: Spend-ceiling enforcement**

In `src/eval/runner.py`, inside `EvalRunner.run`, immediately after appending each `EvalResult`, add:

```python
ceiling = config.eval.spend_ceiling_usd
if ceiling is not None:
    cumulative = sum(r.cost_usd for r in all_results)
    if cumulative > ceiling:
        raise RuntimeError(
            f"Spend ceiling exceeded: ${cumulative:.4f} > ${ceiling:.4f} "
            f"after {len(all_results)} questions. Aborting run."
        )
```

- [ ] **Step 9: Run all eval tests and verify they pass**

```bash
pytest tests/test_eval_cost_ledger.py tests/test_eval_runner.py \
       tests/test_eval_metrics_generation.py tests/test_eval_aggregator.py -v
```
Expected: all PASS. Any pre-existing test that asserted exact dict shape on `aggregate_costs` may need a one-line update to allow the new keys.

- [ ] **Step 10: Commit**

```bash
git add src/eval/schemas.py src/eval/runner.py src/eval/metrics/generation.py \
        src/eval/metrics/operational.py src/eval/pricing.py src/llm_handler.py \
        tests/test_eval_cost_ledger.py tests/test_eval_runner.py \
        tests/test_eval_metrics_generation.py
git commit -m "feat(eval): cost ledger covers generator + judge + rewriter spend"
```

---

## Task 3: `BgeEmbedder` — Chroma EmbeddingFunction adapter

**Files:**
- Create: `src/eval/embedders/__init__.py`
- Create: `src/eval/embedders/bge_small.py`
- Create: `tests/test_eval_embedder_bge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_embedder_bge.py`:

```python
"""Tests for BgeEmbedder — a Chroma EmbeddingFunction adapter for BAAI/bge-small-en-v1.5."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped to amortize the model-load cost across tests."""
    from src.eval.embedders import BgeEmbedder
    return BgeEmbedder()


def test_returns_384_dim_vectors(embedder):
    out = embedder(["hello world"])
    assert len(out) == 1
    assert len(out[0]) == 384
    assert all(isinstance(x, float) for x in out[0])


def test_synonyms_closer_than_unrelated(embedder):
    """Sanity check that the right model is loaded — not a stub."""
    import numpy as np
    a, b, c = embedder(["cat", "feline", "airplane"])
    a, b, c = np.array(a), np.array(b), np.array(c)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert cos(a, b) > cos(a, c), "BGE should rank cat~feline > cat~airplane"


def test_chroma_collection_uses_embedder(embedder):
    """End-to-end: a Chroma collection created with BgeEmbedder retrieves the right doc."""
    import chromadb
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="test_bge_e2e",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    coll.upsert(
        ids=["d1", "d2", "d3"],
        documents=[
            "Cats are small carnivorous mammals often kept as pets.",
            "Airplanes are powered flying vehicles with fixed wings.",
            "Dogs are domesticated descendants of wolves.",
        ],
    )
    res = coll.query(query_texts=["What is a feline?"], n_results=1)
    assert res["ids"][0][0] == "d1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_embedder_bge.py -v
```
Expected: FAIL — `ImportError: cannot import name 'BgeEmbedder'`.

- [ ] **Step 3: Implement `BgeEmbedder`**

Create `src/eval/embedders/bge_small.py`:

```python
"""BgeEmbedder — Chroma EmbeddingFunction adapter for BAAI/bge-small-en-v1.5.

Pipeline position:
    Document → Chunks → [BgeEmbedder] → Vectors (384-dim) → ChromaDB

Phase 2 lever 2b. The factory installs this on the Chroma collection at
creation time; ChromaVectorStore.upsert/query then auto-embeds via this
function with no per-call code change.

Why bge-small-en-v1.5:
    - 384 dim — same as ChromaDB's default ONNX MiniLM, so dimension-comparable.
    - Strong on MTEB retrieval benchmarks (top-tier 33M-param model).
    - Loadable via `sentence-transformers`, which is already a project dep.
"""

from __future__ import annotations

from typing import Sequence

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class BgeEmbedder(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding function backed by sentence-transformers.

    Caches the SentenceTransformer model on the instance to avoid re-loading
    on every call. Each instance is safe to share across one collection.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        # WHY lazy import: sentence-transformers is heavy. Only import when an
        # instance is created so module import remains cheap for tests that
        # never construct one.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL_NAME)

    def __call__(self, input: Documents) -> Embeddings:
        """Encode a batch of documents into 384-dim vectors.

        Args:
            input: List of strings to embed.

        Returns:
            List of 384-element float lists, one per input document.
        """
        # WHY tolist(): sentence-transformers returns a numpy array; Chroma
        # expects a plain list[list[float]] for serialization.
        vectors = self._model.encode(list(input), normalize_embeddings=True)
        return vectors.tolist()

    @staticmethod
    def name() -> str:
        """Required by Chroma >= 0.4.x for embedding-function identification."""
        return "bge_small_en_v1_5"
```

Create `src/eval/embedders/__init__.py`:

```python
"""Phase 2 embedder package — pluggable Chroma EmbeddingFunction adapters."""

from src.eval.embedders.bge_small import BgeEmbedder

__all__ = ["BgeEmbedder"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_eval_embedder_bge.py -v
```
Expected: all PASS. First run downloads ~120MB of model weights; subsequent runs use the local HF cache.

- [ ] **Step 5: Commit**

```bash
git add src/eval/embedders/__init__.py src/eval/embedders/bge_small.py \
        tests/test_eval_embedder_bge.py
git commit -m "feat(eval): add BgeEmbedder as Chroma EmbeddingFunction adapter"
```

---

## Task 4: `BM25HybridRetriever` — RRF fusion of sparse + dense

**Files:**
- Create: `src/eval/retrievers/__init__.py`
- Create: `src/eval/retrievers/bm25_hybrid.py`
- Create: `tests/test_eval_retriever_bm25_hybrid.py`
- Modify: `requirements.txt` (+ `rank-bm25`)

- [ ] **Step 1: Add the dep**

Append to `requirements.txt`:

```
rank-bm25==0.2.2
```

Install locally:

```bash
pip install rank-bm25==0.2.2
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_eval_retriever_bm25_hybrid.py`:

```python
"""Tests for BM25HybridRetriever — RRF fusion of BM25 (sparse) + dense (Chroma)."""

from __future__ import annotations

from src.vector_store import SearchResult


def _sr(chunk_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, score=score, metadata={})


def test_rrf_fusion_asymmetric_inputs():
    """RRF on A=[a,b,c,d], B=[d,a] with rrf_k=60 yields fused order a, d, b, c."""
    from src.eval.retrievers.bm25_hybrid import reciprocal_rank_fusion
    A = ["a", "b", "c", "d"]
    B = ["d", "a"]
    fused = reciprocal_rank_fusion([A, B], rrf_k=60)
    assert fused == ["a", "d", "b", "c"]


def test_hybrid_retrieve_returns_top_k():
    """End-to-end: hybrid retriever combines BM25 and Chroma results into top-K."""
    import chromadb
    from src.eval.retrievers.bm25_hybrid import BM25HybridRetriever
    from src.vector_store import ChromaVectorStore

    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="test_hybrid", metadata={"hnsw:space": "cosine"},
    )
    coll.upsert(
        ids=["d1", "d2", "d3", "d4"],
        documents=[
            "Cats are small carnivorous mammals often kept as pets.",
            "Reciprocal rank fusion is a standard sparse-dense combination.",
            "Hybrid search blends BM25 and dense retrieval signals.",
            "Airplanes have fixed wings.",
        ],
    )
    vs = ChromaVectorStore(collection=coll)
    retriever = BM25HybridRetriever(
        vector_store=vs,
        documents={"d1": coll.get(ids=["d1"])["documents"][0],
                    "d2": coll.get(ids=["d2"])["documents"][0],
                    "d3": coll.get(ids=["d3"])["documents"][0],
                    "d4": coll.get(ids=["d4"])["documents"][0]},
        bm25_top_k=3,
        dense_top_k=3,
        rrf_k=60,
    )
    out = retriever.retrieve("hybrid sparse dense fusion", top_k=2)
    assert len(out) == 2
    assert all(isinstance(r, SearchResult) for r in out)
    # Top result should be one of d2 or d3 (both directly relevant).
    assert out[0].chunk_id in {"d2", "d3"}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_eval_retriever_bm25_hybrid.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 4: Implement `BM25HybridRetriever`**

Create `src/eval/retrievers/bm25_hybrid.py`:

```python
"""BM25HybridRetriever — Reciprocal Rank Fusion of sparse (BM25) + dense (Chroma) retrieval.

Pipeline position:
    query → [BM25 + Dense → RRF] → top-K SearchResult → Reranker / Generator

Phase 2 lever 2c. The retriever keeps two parallel ranked lists (BM25 over
documents, dense over Chroma vectors), then fuses them with RRF:

    score(d) = sum over r in {dense, sparse} of 1 / (rrf_k + rank_r(d))

Why RRF over weighted-sum: RRF is parameter-light (one constant), robust to
score-scale differences across the two retrievers, and the literature shows
it consistently matches or beats tuned weighted-sum on benchmarks like BEIR.
"""

from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from src.vector_store import ChromaVectorStore, SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse multiple ranked ID lists into one via Reciprocal Rank Fusion.

    Args:
        rankings: Iterable of ranked ID sequences. Each sequence is one
            retriever's ranking, most-relevant first.
        rrf_k: RRF constant (60 is the textbook default; smaller emphasizes
            top-rank items more, larger flattens contributions).

    Returns:
        Fused ranking, IDs ordered by descending fused score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


class BM25HybridRetriever:
    """Retriever that fuses BM25 and dense Chroma rankings.

    The BM25 index is built once at construction time over a `documents` mapping.
    Each retrieve() call queries both BM25 and the vector store, then RRF-fuses
    the two rankings before truncating to the requested top-K.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        documents: dict[str, str],
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """Build the BM25 index and store retrieval parameters.

        Args:
            vector_store: Dense retriever (Chroma collection wrapper).
            documents: Mapping of chunk_id → raw document text. BM25 needs
                tokenized text; this dict is the authoritative corpus.
            bm25_top_k: Number of candidates BM25 returns per query.
            dense_top_k: Number of candidates the dense retriever returns.
            rrf_k: RRF fusion constant.
        """
        self._vector_store = vector_store
        self._chunk_ids = list(documents.keys())
        # WHY simple split: rank-bm25 expects pre-tokenized inputs. A whitespace
        # split is good enough for English RAG corpora; nltk stems/stopwords
        # would help marginally but add a runtime dep we don't want here.
        tokenized = [documents[i].lower().split() for i in self._chunk_ids]
        self._bm25 = BM25Okapi(tokenized)
        self._documents = documents
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Run BM25 + dense in parallel, RRF-fuse, return top-K SearchResults.

        Args:
            query: Natural-language query.
            top_k: Number of fused results to return.

        Returns:
            Top-K SearchResult ordered by fused score descending. Score on each
            result is the dense similarity (BM25 ranks aren't directly comparable;
            keeping dense score lets downstream rerankers/refusal-handlers reuse
            it as a confidence proxy).
        """
        # --- Sparse side -------------------------------------------------------
        sparse_scores = self._bm25.get_scores(query.lower().split())
        sparse_ranked = sorted(
            range(len(self._chunk_ids)),
            key=lambda i: sparse_scores[i],
            reverse=True,
        )[: self._bm25_top_k]
        sparse_ids = [self._chunk_ids[i] for i in sparse_ranked]

        # --- Dense side --------------------------------------------------------
        dense_results = self._vector_store.query(
            query_text=query, top_k=self._dense_top_k,
        )
        dense_ids = [r.chunk_id for r in dense_results]
        dense_score_by_id = {r.chunk_id: r.score for r in dense_results}

        # --- Fusion ------------------------------------------------------------
        fused_ids = reciprocal_rank_fusion(
            [sparse_ids, dense_ids], rrf_k=self._rrf_k,
        )[:top_k]

        return [
            SearchResult(
                chunk_id=cid,
                content=self._documents[cid],
                score=dense_score_by_id.get(cid, 0.0),
                metadata={},
            )
            for cid in fused_ids
        ]
```

Create `src/eval/retrievers/__init__.py`:

```python
"""Phase 2 retriever package — hybrid sparse/dense retrieval and reranking."""

from src.eval.retrievers.bm25_hybrid import BM25HybridRetriever

__all__ = ["BM25HybridRetriever"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_eval_retriever_bm25_hybrid.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/eval/retrievers/__init__.py \
        src/eval/retrievers/bm25_hybrid.py tests/test_eval_retriever_bm25_hybrid.py
git commit -m "feat(eval): add BM25HybridRetriever with RRF fusion"
```

---

## Task 5: `CrossEncoderReranker` — ms-marco-MiniLM rerank top-N → top-K

**Files:**
- Create: `src/eval/retrievers/reranker.py`
- Modify: `src/eval/retrievers/__init__.py`
- Create: `tests/test_eval_retriever_reranker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_retriever_reranker.py`:

```python
"""Tests for CrossEncoderReranker — re-scores candidates with a cross-encoder model."""

from __future__ import annotations

import pytest

from src.vector_store import SearchResult


@pytest.fixture(scope="module")
def reranker():
    from src.eval.retrievers.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


def test_obvious_match_ranks_first(reranker):
    """Given five candidates with one obviously-relevant doc, it ranks first after rerank."""
    candidates = [
        SearchResult(chunk_id="d1", content="Pyramids of Giza were built around 2500 BC.",
                     score=0.5, metadata={}),
        SearchResult(chunk_id="d2", content="Cats are small carnivorous mammals.",
                     score=0.6, metadata={}),
        SearchResult(chunk_id="d3", content="What is the capital of France? Paris is the capital.",
                     score=0.4, metadata={}),
        SearchResult(chunk_id="d4", content="Airplanes have fixed wings.",
                     score=0.3, metadata={}),
        SearchResult(chunk_id="d5", content="Dogs are domesticated.", score=0.2, metadata={}),
    ]
    out = reranker.rerank("What is the capital of France?", candidates, final_top_k=3)
    assert len(out) == 3
    assert out[0].chunk_id == "d3"


def test_rerank_preserves_search_result_shape(reranker):
    candidates = [
        SearchResult(chunk_id="d1", content="hello", score=0.5, metadata={"k": "v"}),
        SearchResult(chunk_id="d2", content="world", score=0.4, metadata={}),
    ]
    out = reranker.rerank("greeting", candidates, final_top_k=2)
    assert all(isinstance(r, SearchResult) for r in out)
    # Original score and metadata should round-trip.
    found = {r.chunk_id: r for r in out}
    assert found["d1"].metadata == {"k": "v"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_retriever_reranker.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `CrossEncoderReranker`**

Create `src/eval/retrievers/reranker.py`:

```python
"""CrossEncoderReranker — re-scores retrieval candidates with a cross-encoder model.

Pipeline position:
    Retriever top-N → [CrossEncoderReranker] → top-K → Refusal / Generator

Phase 2 lever 2d. Cross-encoders (single-tower models that consume both
the query and a candidate together) typically outperform bi-encoder retrieval
in precision at the cost of latency. We use ms-marco-MiniLM-L-6-v2 — small
enough to run on CPU in milliseconds per pair, trained on MS MARCO so the
ranking signal transfers well to general-domain QA.
"""

from __future__ import annotations

from src.vector_store import SearchResult


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder to re-score retrieval candidates."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.MODEL_NAME)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        final_top_k: int,
    ) -> list[SearchResult]:
        """Re-score candidates against the query and return top-K reranked.

        Args:
            query: Original user query.
            candidates: Pre-retrieved chunks (typically top-N from a base retriever).
            final_top_k: How many to keep after reranking.

        Returns:
            Top-K SearchResult ordered by descending cross-encoder score. The
            original `score` field is *replaced* with the cross-encoder score so
            downstream consumers reading `result.score` get the more precise signal.
        """
        if not candidates:
            return []
        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)
        # Pair each candidate with its new score, sort, truncate.
        scored = sorted(
            zip(candidates, scores), key=lambda t: t[1], reverse=True,
        )[:final_top_k]
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=float(s),
                metadata=c.metadata,
            )
            for c, s in scored
        ]
```

Update `src/eval/retrievers/__init__.py`:

```python
"""Phase 2 retriever package — hybrid sparse/dense retrieval and reranking."""

from src.eval.retrievers.bm25_hybrid import BM25HybridRetriever
from src.eval.retrievers.reranker import CrossEncoderReranker

__all__ = ["BM25HybridRetriever", "CrossEncoderReranker"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_eval_retriever_reranker.py -v
```
Expected: PASS. First call downloads ~80MB of model weights.

- [ ] **Step 5: Commit**

```bash
git add src/eval/retrievers/reranker.py src/eval/retrievers/__init__.py \
        tests/test_eval_retriever_reranker.py
git commit -m "feat(eval): add CrossEncoderReranker (ms-marco-MiniLM)"
```

---

## Task 6: `QueryRewriter` — LLM-based query expansion (with cost capture)

**Files:**
- Create: `src/eval/transforms/__init__.py`
- Create: `src/eval/transforms/query_rewriter.py`
- Create: `tests/test_eval_transform_rewriter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_transform_rewriter.py`:

```python
"""Tests for QueryRewriter — LLM query expansion with cost capture."""

from __future__ import annotations

from typing import Any


class _StubLLM:
    """Records calls and returns canned responses + token counts."""

    def __init__(self, response: str, prompt_tokens: int = 50, completion_tokens: int = 30):
        self._response = response
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[tuple[str, str | None]] = []

    def generate_with_usage(self, prompt: str, system_prompt: str | None = None
                             ) -> tuple[str, int, int]:
        self.calls.append((prompt, system_prompt))
        return self._response, self._prompt_tokens, self._completion_tokens


def test_no_model_passthrough():
    """When model is None, expand returns [query] unchanged with zero cost."""
    from src.eval.transforms import QueryRewriter
    rw = QueryRewriter(model=None, max_expansions=3, llm=None)
    queries, cost, p_t, c_t = rw.expand("What is RAG?")
    assert queries == ["What is RAG?"]
    assert cost == 0.0
    assert p_t == 0
    assert c_t == 0


def test_expansion_returns_dedup_list_and_cost():
    """With a real model name and stub LLM, expand returns deduped expansions + cost."""
    from src.eval.transforms import QueryRewriter
    stub = _StubLLM(
        response='["What does RAG stand for?", "Define retrieval augmented generation", '
                 '"What is RAG?"]',
        prompt_tokens=80, completion_tokens=40,
    )
    rw = QueryRewriter(model="gpt-4.1-nano", max_expansions=3, llm=stub)
    queries, cost, p_t, c_t = rw.expand("What is RAG?")
    # Original query is always first; duplicate dropped; max_expansions=3 cap respected.
    assert queries[0] == "What is RAG?"
    assert "What does RAG stand for?" in queries
    assert "Define retrieval augmented generation" in queries
    assert len(queries) == len(set(queries))  # no duplicates
    assert len(queries) <= 4  # original + at most max_expansions
    # Cost was computed from the stub's token counts at gpt-4.1-nano price.
    assert cost > 0.0
    assert p_t == 80
    assert c_t == 40


def test_malformed_llm_response_falls_back_to_passthrough():
    """If the LLM returns non-JSON, expand returns [query] and logs a warning."""
    from src.eval.transforms import QueryRewriter
    stub = _StubLLM(response="not json at all", prompt_tokens=50, completion_tokens=10)
    rw = QueryRewriter(model="gpt-4.1-nano", max_expansions=3, llm=stub)
    queries, cost, _, _ = rw.expand("What is RAG?")
    assert queries == ["What is RAG?"]
    # Cost is still charged because the call did happen.
    assert cost > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_transform_rewriter.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `QueryRewriter`**

Create `src/eval/transforms/query_rewriter.py`:

```python
"""QueryRewriter — LLM-based query expansion with token/cost capture.

Pipeline position:
    user query → [QueryRewriter] → {q, q', q''} → Retriever → ...

Phase 2 lever 2e. Expansion gives the retriever multiple lexical/semantic
formulations of the same intent, which raises recall on questions where the
original phrasing diverges from the corpus phrasing. We use a tiny model
(gpt-4.1-nano) because the task is cheap and we don't want this lever to
dominate the cost ledger.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from src.eval import pricing

logger = logging.getLogger(__name__)


class _LLMHandler(Protocol):
    """Structural type for any object exposing generate_with_usage."""
    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None,
    ) -> tuple[str, int, int]: ...


class QueryRewriter:
    """Expands one user query into up to N alternative phrasings via an LLM."""

    SYSTEM_PROMPT = (
        "You rewrite user search queries into alternative phrasings that preserve "
        "the original intent but vary surface form. Respond ONLY with a JSON "
        "array of strings — no prose, no code fences."
    )

    def __init__(
        self,
        model: str | None,
        max_expansions: int,
        llm: _LLMHandler | None,
    ) -> None:
        """Configure the rewriter.

        Args:
            model: LLM model name. None disables rewriting (pass-through).
            max_expansions: Cap on the number of alternative phrasings to return.
            llm: Object exposing generate_with_usage(prompt, system_prompt). Required
                if model is not None.
        """
        self._model = model
        self._max_expansions = max_expansions
        self._llm = llm

    def expand(self, query: str) -> tuple[list[str], float, int, int]:
        """Expand `query` into up to N+1 unique phrasings.

        Returns:
            (queries, cost_usd, prompt_tokens, completion_tokens). The original
            query is always the first element. When `model is None`, returns
            ([query], 0.0, 0, 0) and skips the LLM call.
        """
        if self._model is None:
            return [query], 0.0, 0, 0
        if self._llm is None:
            raise ValueError("QueryRewriter has model set but no llm handler provided.")

        user_prompt = (
            f'Original query: "{query}"\n\n'
            f"Return a JSON array of up to {self._max_expansions} alternative "
            f"phrasings of this query. Do NOT include the original."
        )
        raw, p_t, c_t = self._llm.generate_with_usage(
            user_prompt, system_prompt=self.SYSTEM_PROMPT,
        )
        cost = pricing.cost_usd(self._model, p_t, c_t)

        expansions = self._parse_expansions(raw)
        # Always lead with original; dedupe; cap at original + max_expansions.
        ordered: list[str] = [query]
        for alt in expansions:
            if alt and alt not in ordered:
                ordered.append(alt)
            if len(ordered) >= self._max_expansions + 1:
                break
        return ordered, cost, p_t, c_t

    @staticmethod
    def _parse_expansions(raw: str) -> list[str]:
        """Strip code fences and parse the JSON array; return [] on failure."""
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("QueryRewriter got non-JSON response — falling back to [query] only.")
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if isinstance(item, str)]
```

Create `src/eval/transforms/__init__.py`:

```python
"""Phase 2 transforms — pre/post pipeline hooks (rewriter, refusal handler)."""

from src.eval.transforms.query_rewriter import QueryRewriter

__all__ = ["QueryRewriter"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_eval_transform_rewriter.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/transforms/__init__.py src/eval/transforms/query_rewriter.py \
        tests/test_eval_transform_rewriter.py
git commit -m "feat(eval): add QueryRewriter for LLM-based expansion with cost capture"
```

---

## Task 7: `RefusalHandler` — similarity gate for unanswerable questions

**Files:**
- Create: `src/eval/transforms/refusal_handler.py`
- Modify: `src/eval/transforms/__init__.py`
- Create: `tests/test_eval_transform_refusal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_transform_refusal.py`:

```python
"""Tests for RefusalHandler — pure-logic similarity gate."""

from __future__ import annotations

from src.vector_store import SearchResult


def _sr(score: float, chunk_id: str = "d1") -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content="x", score=score, metadata={})


def test_refuses_when_top1_below_threshold():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.20), _sr(0.10)]) is True


def test_does_not_refuse_when_top1_above_threshold():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.50), _sr(0.10)]) is False


def test_refuses_on_empty_candidates():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([]) is True


def test_disabled_handler_never_refuses():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=False, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.0)]) is False
    assert h.should_refuse([]) is False


def test_refuse_response_returns_text_and_no_chunks():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I cannot answer.")
    chunks, answer = h.refuse_response()
    assert chunks == []
    assert answer == "I cannot answer."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_transform_refusal.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `RefusalHandler`**

Create `src/eval/transforms/refusal_handler.py`:

```python
"""RefusalHandler — answerability gate based on top-1 retrieval similarity.

Pipeline position:
    Retriever (post-rerank) candidates → [RefusalHandler] → answer or refusal text

Phase 2 lever 2g. SQuAD v2 includes 'unanswerable' questions whose gold
answer is the empty string. Phase 1's pipeline always tries to answer,
which means it scores poorly on `refusal_correctness`. RefusalHandler is
a deterministic short-circuit: when no candidate clears the similarity
threshold, return a fixed no-answer text instead of calling the LLM.
"""

from __future__ import annotations

from src.vector_store import SearchResult


class RefusalHandler:
    """Deterministic answerability gate driven by top-1 similarity score."""

    def __init__(
        self,
        enabled: bool,
        similarity_threshold: float,
        no_answer_text: str,
    ) -> None:
        """Configure the gate.

        Args:
            enabled: When False, should_refuse always returns False.
            similarity_threshold: Top-1 score must be >= this to NOT refuse.
            no_answer_text: Text returned in place of an LLM answer on refusal.
        """
        self._enabled = enabled
        self._threshold = similarity_threshold
        self._no_answer_text = no_answer_text

    def should_refuse(self, candidates: list[SearchResult]) -> bool:
        """Return True if the pipeline should short-circuit to no-answer text."""
        if not self._enabled:
            return False
        if not candidates:
            return True
        return candidates[0].score < self._threshold

    def refuse_response(self) -> tuple[list[SearchResult], str]:
        """Return ([], no_answer_text) — used when should_refuse is True."""
        return [], self._no_answer_text
```

Update `src/eval/transforms/__init__.py`:

```python
"""Phase 2 transforms — pre/post pipeline hooks (rewriter, refusal handler)."""

from src.eval.transforms.query_rewriter import QueryRewriter
from src.eval.transforms.refusal_handler import RefusalHandler

__all__ = ["QueryRewriter", "RefusalHandler"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_eval_transform_refusal.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/transforms/refusal_handler.py src/eval/transforms/__init__.py \
        tests/test_eval_transform_refusal.py
git commit -m "feat(eval): add RefusalHandler with similarity gate"
```

---

## Task 8: Wire all Phase 2 levers into `build_pipeline` + `EvalPipeline.query`

**Files:**
- Modify: `src/eval/pipeline_factory.py`
- Create: `tests/test_eval_pipeline_factory_phase2.py`
- Modify: `tests/test_eval_pipeline_factory.py` (existing — verify still passes)
- Modify: `tests/test_eval_smoke.py` (existing — verify still passes)
- Create: `tests/fixtures/phase2_corpus/d1.txt`, `d2.txt`, `d3.txt`

- [ ] **Step 1: Add fixture corpus**

Create `tests/fixtures/phase2_corpus/d1.txt`:

```
Reciprocal rank fusion combines two ranked lists by summing 1/(k+rank) for each item.
```

Create `tests/fixtures/phase2_corpus/d2.txt`:

```
Cross-encoders score query-document pairs jointly and improve retrieval precision.
```

Create `tests/fixtures/phase2_corpus/d3.txt`:

```
Airplanes have fixed wings and powered engines.
```

- [ ] **Step 2: Write the failing factory tests**

Create `tests/test_eval_pipeline_factory_phase2.py`:

```python
"""Phase 2 factory tests — every tier YAML produces a pipeline whose attributes match."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.config import load_config
from src.eval.pipeline_factory import build_pipeline


PHASE2_DIR = Path("configs/eval/phase2")


@pytest.fixture
def stub_llm():
    class _S:
        def generate(self, prompt, system_prompt=None):
            return "stub answer"
        def generate_with_usage(self, prompt, system_prompt=None):
            return "stub answer", 10, 5
    return _S()


@pytest.mark.parametrize("yaml_name,expects", [
    ("phase2_baseline.yaml", {"rewriter": False, "reranker": False, "refusal": False, "hybrid": False, "embedder": "chroma_default"}),
    ("phase2b_embedder.yaml", {"rewriter": False, "reranker": False, "refusal": False, "hybrid": False, "embedder": "bge_small_en_v1_5"}),
    ("phase2c_hybrid.yaml",   {"rewriter": False, "reranker": False, "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2d_rerank.yaml",   {"rewriter": False, "reranker": True,  "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2e_rewrite.yaml",  {"rewriter": True,  "reranker": True,  "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2g_refusal.yaml",  {"rewriter": True,  "reranker": True,  "refusal": True,  "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
])
def test_phase2_yaml_builds_pipeline_with_expected_attrs(yaml_name, expects, stub_llm):
    cfg = load_config(PHASE2_DIR / yaml_name)
    pipeline = build_pipeline(
        cfg, dataset_name="squad_v2_dev_200",
        llm_override=stub_llm, judge_llm_override=stub_llm,
    )
    try:
        assert (pipeline.rewriter is not None) == expects["rewriter"]
        assert (pipeline.reranker is not None) == expects["reranker"]
        assert (pipeline.refusal_handler is not None) == expects["refusal"]
        assert (pipeline.hybrid_retriever is not None) == expects["hybrid"]
        assert cfg.pipeline.embedder.name == expects["embedder"]
    finally:
        pipeline.teardown()


def test_phase2_query_with_refusal_short_circuits(stub_llm, tmp_path):
    """End-to-end smoke: refusal handler short-circuits when top-1 < threshold."""
    cfg = load_config(PHASE2_DIR / "phase2g_refusal.yaml")
    pipeline = build_pipeline(
        cfg, dataset_name="squad_v2_dev_200",
        llm_override=stub_llm, judge_llm_override=stub_llm,
    )
    try:
        # Empty index → top-1 score is 0 → handler refuses.
        chunks, answer, telemetry = pipeline.query("what is x?")
        assert chunks == []
        assert answer == cfg.pipeline.refusal_handler.no_answer_text
        assert "refusal_check" in telemetry["timings_ms"]
    finally:
        pipeline.teardown()
```

(Note: this test references YAMLs created in Task 9. Run the parametrized test only after Task 9 lands, OR write the YAMLs as part of this task. Per the spec, configs land in commit 10; for TDD we'll author them inline here as fixtures and move them to `configs/eval/phase2/` in commit 10. To keep the commit graph clean, this task creates the YAMLs in `tests/fixtures/phase2_configs/` and Task 9 moves them.)

Adjust `PHASE2_DIR` in the test to `Path("tests/fixtures/phase2_configs")` and create those YAMLs as fixtures here. Task 9 then moves them.

Create `tests/fixtures/phase2_configs/phase2_baseline.yaml`:

```yaml
name: "phase2_baseline"
description: "Baseline anchor for the Phase 2 matrix — identical to baseline_squad_only."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `tests/fixtures/phase2_configs/phase2b_embedder.yaml`:

```yaml
name: "phase2b_embedder"
description: "Phase 2 tier 2b — swap default embedder for BAAI/bge-small-en-v1.5."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `tests/fixtures/phase2_configs/phase2c_hybrid.yaml`:

```yaml
name: "phase2c_hybrid"
description: "Phase 2 tier 2c — add BM25 hybrid on top of BGE."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `tests/fixtures/phase2_configs/phase2d_rerank.yaml`:

```yaml
name: "phase2d_rerank"
description: "Phase 2 tier 2d — add cross-encoder rerank on top of hybrid."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `tests/fixtures/phase2_configs/phase2e_rewrite.yaml`:

```yaml
name: "phase2e_rewrite"
description: "Phase 2 tier 2e — add LLM query rewriting on top of rerank."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  query_rewriter: {model: gpt-4.1-nano, max_expansions: 3}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `tests/fixtures/phase2_configs/phase2g_refusal.yaml`:

```yaml
name: "phase2g_refusal"
description: "Phase 2 tier 2g — add refusal handler on top of the full upgraded retrieval stack."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  query_rewriter: {model: gpt-4.1-nano, max_expansions: 3}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
  refusal_handler: {enabled: true, similarity_threshold: 0.35,
                     no_answer_text: "I don't have enough information to answer that."}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_eval_pipeline_factory_phase2.py -v
```
Expected: FAIL — `pipeline.rewriter` etc. don't exist; `EvalPipeline` doesn't accept the new fields.

- [ ] **Step 4: Extend `EvalPipeline`**

In `src/eval/pipeline_factory.py`, modify the `@dataclass class EvalPipeline` to add the new optional fields after `judge_llm`:

```python
@dataclass
class EvalPipeline:
    # ... existing fields unchanged ...
    chunker: TextChunker
    vector_store: ChromaVectorStore
    llm: object
    judge_llm: object
    config: EvalConfig
    dataset_name: str
    # Phase 2 additions — None when the corresponding lever is off.
    hybrid_retriever: object | None = None        # BM25HybridRetriever or None
    reranker: object | None = None                 # CrossEncoderReranker or None
    rewriter: object | None = None                 # QueryRewriter or None
    refusal_handler: object | None = None          # RefusalHandler or None
    _client: object = field(repr=False, default=None)
    _collection_name: str = field(repr=False, default="")
```

- [ ] **Step 5: Extend `EvalPipeline.query`**

Replace the existing `EvalPipeline.query` body with the layered version. Keep the dummy/null cases as no-ops so Phase 1 baselines route through identical control flow.

```python
    def query(self, question: str) -> tuple[list[SearchResult], str, dict]:
        """Run the configured pipeline; return (chunks, answer, telemetry).

        Telemetry keys:
            timings_ms: subset of {"rewrite", "retrieve", "rerank", "refusal_check", "generate"}
            tokens:     {"prompt": int, "completion": int} for the generator call
            cost_usd:   generator-side cost (judge + rewriter accounted separately)
            rewriter_cost_usd: rewriter spend (0.0 when disabled)
        """
        from src.eval import pricing
        timings: dict[str, float] = {}
        rewriter_cost = 0.0

        # ---- Rewrite (lever 2e) -----------------------------------------------
        t = time.perf_counter()
        if self.rewriter is not None:
            queries, rewriter_cost, _, _ = self.rewriter.expand(question)
        else:
            queries = [question]
        timings["rewrite"] = (time.perf_counter() - t) * 1000.0

        # ---- Retrieve ---------------------------------------------------------
        top_k_initial = (
            self.config.pipeline.reranker.rerank_top_n
            if self.reranker is not None else self.config.pipeline.retriever.top_k
        )
        t = time.perf_counter()
        if self.hybrid_retriever is not None:
            # Hybrid: retrieve top-N for each rewritten query, dedup by chunk_id.
            seen: dict[str, SearchResult] = {}
            for q in queries:
                for r in self.hybrid_retriever.retrieve(q, top_k=top_k_initial):
                    if r.chunk_id not in seen:
                        seen[r.chunk_id] = r
            results = list(seen.values())
        else:
            seen = {}
            for q in queries:
                for r in self.vector_store.query(query_text=q, top_k=top_k_initial):
                    if r.chunk_id not in seen:
                        seen[r.chunk_id] = r
            results = list(seen.values())
        timings["retrieve"] = (time.perf_counter() - t) * 1000.0

        # ---- Rerank (lever 2d) ------------------------------------------------
        t = time.perf_counter()
        if self.reranker is not None:
            results = self.reranker.rerank(
                question, results,
                final_top_k=self.config.pipeline.reranker.final_top_k,
            )
        else:
            results = results[: self.config.pipeline.retriever.top_k]
        timings["rerank"] = (time.perf_counter() - t) * 1000.0

        # ---- Refusal gate (lever 2g) ------------------------------------------
        t = time.perf_counter()
        if self.refusal_handler is not None and self.refusal_handler.should_refuse(results):
            chunks, answer = self.refusal_handler.refuse_response()
            timings["refusal_check"] = (time.perf_counter() - t) * 1000.0
            telemetry = {
                "timings_ms": timings,
                "tokens": {"prompt": 0, "completion": 0},
                "cost_usd": 0.0,
                "rewriter_cost_usd": rewriter_cost,
            }
            return chunks, answer, telemetry
        timings["refusal_check"] = (time.perf_counter() - t) * 1000.0

        # ---- Generate ---------------------------------------------------------
        context = "\n\n".join(r.content for r in results)
        system_prompt = (
            "You are a helpful assistant. Answer the question based solely on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        full_prompt_text = system_prompt + "\n" + user_prompt
        model = self.config.pipeline.generator.model

        t = time.perf_counter()
        answer = self.llm.generate(user_prompt, system_prompt=system_prompt)
        timings["generate"] = (time.perf_counter() - t) * 1000.0

        prompt_tokens = count_tokens(full_prompt_text, model)
        completion_tokens = count_tokens(answer, model)
        cost = pricing.cost_usd(model, prompt_tokens, completion_tokens)

        telemetry = {
            "timings_ms": timings,
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
            "cost_usd": cost,
            "rewriter_cost_usd": rewriter_cost,
        }
        return results, answer, telemetry
```

- [ ] **Step 6: Extend `build_pipeline`**

After the existing chunker block in `build_pipeline`, replace the embedder/collection setup with:

```python
    # ---- Embedder (lever 2b) -------------------------------------------------
    embedder_cfg = config.pipeline.embedder
    embedding_function = _build_embedding_function(embedder_cfg)

    collection_name = f"eval_{config.name}_{dataset_name}_{uuid.uuid4().hex[:6]}"
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(collection=collection)
```

Add the helper builders below `build_pipeline`:

```python
def _build_embedding_function(cfg) -> object:
    if cfg.name == "chroma_default":
        from chromadb.utils import embedding_functions
        return embedding_functions.DefaultEmbeddingFunction()
    if cfg.name == "bge_small_en_v1_5":
        from src.eval.embedders import BgeEmbedder
        return BgeEmbedder()
    raise ValueError(f"Unknown embedder name: {cfg.name}")


def _build_hybrid_retriever(cfg, vector_store, documents):
    if not cfg.enabled:
        return None
    from src.eval.retrievers import BM25HybridRetriever
    return BM25HybridRetriever(
        vector_store=vector_store, documents=documents,
        bm25_top_k=cfg.bm25_top_k, dense_top_k=cfg.dense_top_k, rrf_k=cfg.rrf_k,
    )


def _build_reranker(cfg):
    if cfg.model is None:
        return None
    from src.eval.retrievers import CrossEncoderReranker
    return CrossEncoderReranker()


def _build_rewriter(cfg, llm):
    if cfg.model is None:
        return None
    from src.eval.transforms import QueryRewriter
    return QueryRewriter(model=cfg.model, max_expansions=cfg.max_expansions, llm=llm)


def _build_refusal(cfg):
    if not cfg.enabled:
        return None
    from src.eval.transforms import RefusalHandler
    return RefusalHandler(
        enabled=True, similarity_threshold=cfg.similarity_threshold,
        no_answer_text=cfg.no_answer_text,
    )
```

Update the bottom of `build_pipeline` to instantiate the new fields:

```python
    # ---- LLM handlers (existing) ---------------------------------------------
    llm = llm_override if llm_override is not None else LLMHandler(config.pipeline.generator.model)
    judge_llm = (
        judge_llm_override if judge_llm_override is not None
        else LLMHandler(config.eval.judge_model)
    )
    # NOTE: hybrid_retriever is built lazily — it needs the {chunk_id: text} map
    # which is only available after ingest(). Set hybrid_cfg here; build_pipeline
    # caller wires the hybrid retriever inside ingest() once chunks are available.

    return EvalPipeline(
        chunker=chunker,
        vector_store=vector_store,
        llm=llm,
        judge_llm=judge_llm,
        config=config,
        dataset_name=dataset_name,
        hybrid_retriever=None,  # populated post-ingest in ingest() or by caller
        reranker=_build_reranker(config.pipeline.reranker),
        rewriter=_build_rewriter(config.pipeline.query_rewriter, llm=llm),
        refusal_handler=_build_refusal(config.pipeline.refusal_handler),
        _client=client,
        _collection_name=collection_name,
    )
```

In `EvalPipeline.ingest`, at the end of `_ingest_squad` (and `_ingest_ml_papers`), add:

```python
        # Phase 2: build the hybrid retriever now that chunks/contexts are upserted.
        if self.config.pipeline.hybrid.enabled:
            from src.eval.pipeline_factory import _build_hybrid_retriever
            documents_map = dict(zip(ids, documents))
            self.hybrid_retriever = _build_hybrid_retriever(
                self.config.pipeline.hybrid, self.vector_store, documents_map,
            )
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_eval_pipeline_factory_phase2.py \
       tests/test_eval_pipeline_factory.py \
       tests/test_eval_smoke.py -v
```
Expected: PASS. Existing Phase 1 factory tests must still pass (the new fields are all optional and None by default).

- [ ] **Step 8: Commit**

```bash
git add src/eval/pipeline_factory.py tests/test_eval_pipeline_factory_phase2.py \
        tests/fixtures/phase2_configs/ tests/fixtures/phase2_corpus/
git commit -m "feat(eval): wire Phase 2 levers into build_pipeline + EvalPipeline.query"
```

---

## Task 9: Add `archive` subcommand to `cli`

**Files:**
- Modify: `src/eval/cli.py`
- Create: `tests/test_eval_cli_archive.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_cli_archive.py`:

```python
"""Tests for `python -m src.eval.cli archive` — copies small artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.cli import _cmd_archive


def test_archive_copies_four_artifacts(tmp_path):
    # Build a fake run dir with the expected artifacts plus an irrelevant large file.
    src = tmp_path / "eval_runs" / "fake_run"
    src.mkdir(parents=True)
    (src / "metrics.json").write_text("[]")
    (src / "cost.json").write_text("{}")
    (src / "metadata.json").write_text('{"run_id": "fake_run"}')
    (src / "config.yaml").write_text("name: fake")
    (src / "questions.jsonl").write_text("\n".join(["{}"] * 200))  # large

    dst = tmp_path / "docs" / "phase2" / "runs" / "fake_run"
    rc = _cmd_archive(_FakeArgs(run_id="fake_run", to=str(dst), runs_root=str(tmp_path / "eval_runs")))
    assert rc == 0
    assert (dst / "metrics.json").exists()
    assert (dst / "cost.json").exists()
    assert (dst / "metadata.json").exists()
    assert (dst / "config.yaml").exists()
    # questions.jsonl is NOT copied (large), but its SHA should be in metadata.json
    assert not (dst / "questions.jsonl").exists()
    md = json.loads((dst / "metadata.json").read_text())
    assert "questions_jsonl_sha256" in md


class _FakeArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_eval_cli_archive.py -v
```
Expected: FAIL — `_cmd_archive` doesn't exist.

- [ ] **Step 3: Implement `_cmd_archive` in `src/eval/cli.py`**

Append to `src/eval/cli.py` (after the existing command functions, before `main`):

```python
def _cmd_archive(args: argparse.Namespace) -> int:
    """Copy small artifacts of a run from eval_runs/ to a tracked location.

    Files copied: metrics.json, cost.json, metadata.json, config.yaml.
    NOT copied: questions.jsonl (large). Its SHA-256 is recorded in metadata.json
    under `questions_jsonl_sha256` so reviewers can verify against a re-run.
    """
    import hashlib
    import json
    import shutil
    from pathlib import Path

    runs_root = Path(getattr(args, "runs_root", None) or "eval_runs")
    src = runs_root / args.run_id
    if not src.exists():
        print(f"Run not found: {src}")
        return 1

    dst = Path(args.to)
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("metrics.json", "cost.json", "config.yaml"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    # Record questions.jsonl SHA in metadata.json before copying it
    metadata = json.loads((src / "metadata.json").read_text())
    questions_path = src / "questions.jsonl"
    if questions_path.exists():
        h = hashlib.sha256()
        h.update(questions_path.read_bytes())
        metadata["questions_jsonl_sha256"] = h.hexdigest()
    (dst / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Archived run {args.run_id} → {dst}")
    return 0
```

In `main`, register the subparser before `args = parser.parse_args(argv)`:

```python
    p_archive = subparsers.add_parser(
        "archive",
        help="Copy small run artifacts (metrics/cost/metadata/config) to a tracked path.",
    )
    p_archive.add_argument("run_id", help="Run id to archive (must exist under eval_runs/).")
    p_archive.add_argument("--to", required=True, help="Destination directory.")
    p_archive.add_argument(
        "--runs-root", default="eval_runs",
        help="Root directory holding run subdirectories (default: eval_runs).",
    )
    p_archive.set_defaults(func=_cmd_archive)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_eval_cli_archive.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/cli.py tests/test_eval_cli_archive.py
git commit -m "feat(eval): add cli archive subcommand to copy small run artifacts to a tracked path"
```

---

## Task 10: Promote Phase 2 tier configs to `configs/eval/phase2/`

**Files:**
- Move test fixtures into the canonical config directory.
- Add three answer-model variants for tier 2f.

- [ ] **Step 1: Move tier YAMLs from fixtures to canonical location**

```bash
mkdir -p configs/eval/phase2
git mv tests/fixtures/phase2_configs/phase2_baseline.yaml   configs/eval/phase2/phase2_baseline.yaml
git mv tests/fixtures/phase2_configs/phase2b_embedder.yaml  configs/eval/phase2/phase2b_embedder.yaml
git mv tests/fixtures/phase2_configs/phase2c_hybrid.yaml    configs/eval/phase2/phase2c_hybrid.yaml
git mv tests/fixtures/phase2_configs/phase2d_rerank.yaml    configs/eval/phase2/phase2d_rerank.yaml
git mv tests/fixtures/phase2_configs/phase2e_rewrite.yaml   configs/eval/phase2/phase2e_rewrite.yaml
git mv tests/fixtures/phase2_configs/phase2g_refusal.yaml   configs/eval/phase2/phase2g_refusal.yaml
```

- [ ] **Step 2: Update the factory test path**

In `tests/test_eval_pipeline_factory_phase2.py`, change `PHASE2_DIR = Path("tests/fixtures/phase2_configs")` to:

```python
PHASE2_DIR = Path("configs/eval/phase2")
```

- [ ] **Step 3: Author the three 2f answer-model variants**

Create `configs/eval/phase2/phase2f_models_gpt5mini.yaml`:

```yaml
name: "phase2f_models_gpt5mini"
description: "Phase 2 tier 2f — answer-model comparison on the full 2g stack: gpt-5-mini variant.
              This config is identical to phase2g_refusal.yaml; PR-B re-uses the 2g artifact rather
              than re-running."
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  embedder: {name: bge_small_en_v1_5}
  retriever: {top_k: 5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  query_rewriter: {model: gpt-4.1-nano, max_expansions: 3}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
  refusal_handler: {enabled: true, similarity_threshold: 0.35,
                     no_answer_text: "I don't have enough information to answer that."}
eval:
  datasets: [squad_v2_dev_200]
  judge_model: gpt-4.1-mini
  bootstrap_n: 1000
  permutation_n: 10000
  seed: 42
  spend_ceiling_usd: 1.5
```

Create `configs/eval/phase2/phase2f_models_gpt41mini.yaml`: same as above with `generator.model: gpt-4.1-mini` and `name: "phase2f_models_gpt41mini"`.

Create `configs/eval/phase2/phase2f_models_haiku.yaml`: same with `generator.model: claude-haiku-4-5` and `name: "phase2f_models_haiku"`.

- [ ] **Step 4: Add a smoke test loading every Phase 2 YAML**

Append to `tests/test_eval_pipeline_factory_phase2.py`:

```python
def test_every_phase2_yaml_loads():
    """Every YAML under configs/eval/phase2/ must validate against EvalConfig."""
    for path in sorted(PHASE2_DIR.glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.name == path.stem
```

- [ ] **Step 5: Run the test to verify all 9 YAMLs validate**

```bash
pytest tests/test_eval_pipeline_factory_phase2.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/eval/phase2/ tests/test_eval_pipeline_factory_phase2.py
git commit -m "chore(eval): add Phase 2 tier configs under configs/eval/phase2/"
```

---

## End of PR-A — Push and open PR

- [ ] **Run the full test suite once before pushing**

```bash
pytest -x -q
```
Expected: zero failures.

- [ ] **Push the branch and open the PR**

```bash
git push -u origin feature/phase2-pipeline-extensions
gh pr create \
  --base feature/eval-harness-1d \
  --head feature/phase2-pipeline-extensions \
  --title "feat(eval): pipeline extensions for Phase 2 RAG quality matrix" \
  --body "$(cat <<'EOF'
## Summary

PR-A of Phase 2: pipeline extensions only. PR-B follows with the experiment runs and writeup.

- 5 new pipeline modules: BgeEmbedder, BM25HybridRetriever, CrossEncoderReranker, QueryRewriter, RefusalHandler.
- Schema extension to PipelineCfg with 5 sub-configs + EvalCfg.spend_ceiling_usd; backward compatible (existing baseline configs unchanged).
- Cost ledger refactor: EvalResult.cost_breakdown covers generator + judge + rewriter; aggregator surfaces per-bucket totals.
- New CLI: python -m src.eval.cli archive <run_id> --to <path>.
- 9 Phase 2 tier YAMLs under configs/eval/phase2/.
- New dep: rank-bm25.

Stacked on #4 (Phase 1 PR-D). Will retarget to main once Phase 1 merges.

## Test plan

- [x] pytest -x -q passes locally
- [x] Existing Phase 1 baseline configs still load and produce the same pipeline shape
- [x] Each Phase 2 YAML builds a pipeline with the expected lever activations
- [ ] Reviewer: pip install -r requirements.txt adds exactly one entry (rank-bm25)
- [ ] Reviewer: python -m src.eval.cli run --config configs/eval/phase2/phase2_baseline.yaml succeeds with stubbed-LLM mode
EOF
)"
```

---

# PR-B — Experiments + Writeup

PR-B is data-driven. Tasks below assume PR-A has merged or is at least in a state where its CLI works end-to-end against the real OpenAI API.

> **Branching for PR-B:**
> ```bash
> git checkout feature/phase2-pipeline-extensions
> git pull
> git checkout -b feature/phase2-experiments-writeup
> ```

## Task 11: Run the baseline anchor + archive

**Files:**
- Create: `docs/phase2/runs/<short_id>_phase2_baseline/{metrics,cost,metadata,config}.json`

- [ ] **Step 1: Run the baseline**

```bash
set -a; source .env; set +a
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2_baseline.yaml
```
Expected: end-of-output line `Run complete: phase2_baseline n=200 errors=0 <run_id>`. Capture `<run_id>`.

- [ ] **Step 2: Archive small artifacts to `docs/phase2/runs/`**

```bash
RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"   # date+config prefix
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2_baseline"
```

- [ ] **Step 3: Verify the four artifacts landed**

```bash
ls "docs/phase2/runs/${SHORT_ID}_phase2_baseline"
```
Expected: exactly `metrics.json`, `cost.json`, `metadata.json`, `config.yaml`. No `questions.jsonl`.

- [ ] **Step 4: Commit**

```bash
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 baseline run"
```

---

## Task 12: Run tier 2b (BGE embedder) + archive

- [ ] **Step 1: Run**

```bash
set -a; source .env; set +a
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2b_embedder.yaml
```

- [ ] **Step 2: Archive**

```bash
RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2b_embedder"
```

- [ ] **Step 3: Spot-check that recall@5 is reported in metrics.json**

```bash
python3 -c "import json; m=json.load(open('docs/phase2/runs/${SHORT_ID}_phase2b_embedder/metrics.json')); \
[print(r) for r in m if r['metric_name']=='recall_at_5']"
```

- [ ] **Step 4: Commit**

```bash
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2b — BGE embedder"
```

---

## Task 13: Run tier 2c (BM25 hybrid) + archive

- [ ] **Step 1: Run**

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2c_hybrid.yaml
```

- [ ] **Step 2: Archive + commit**

```bash
RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2c_hybrid"
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2c — BM25 hybrid retrieval"
```

---

## Task 14: Run tier 2d (cross-encoder rerank) + archive

- [ ] **Step 1–3: Same pattern as Task 13 with `phase2d_rerank.yaml`**

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2d_rerank.yaml

RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2d_rerank"
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2d — cross-encoder rerank"
```

---

## Task 15: Run tier 2e (query rewriting) + archive

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2e_rewrite.yaml

RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2e_rewrite"
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2e — query rewriting"
```

---

## Task 16: Run tier 2g (refusal handler) + archive

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2g_refusal.yaml

RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2g_refusal"
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2g — refusal handler"
```

---

## Task 17: Run tier 2f (answer-model comparison: gpt-4.1-mini and claude-haiku-4-5)

The gpt-5-mini variant is identical to tier 2g — re-use that artifact. Only run the two non-baseline answer models.

- [ ] **Step 1: Run gpt-4.1-mini variant**

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2f_models_gpt41mini.yaml

RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2f_models_gpt41mini"
```

- [ ] **Step 2: Run claude-haiku-4-5 variant**

```bash
conda run -n rag-qa --no-capture-output \
  python -m src.eval.cli run --config configs/eval/phase2/phase2f_models_haiku.yaml

RUN_ID="<paste run_id>"
SHORT_ID="${RUN_ID:0:23}"
python -m src.eval.cli archive "$RUN_ID" \
  --to "docs/phase2/runs/${SHORT_ID}_phase2f_models_haiku"
```

- [ ] **Step 3: Symlink the gpt-5-mini artifact for completeness**

```bash
G2G_DIR=$(ls -d docs/phase2/runs/*_phase2g_refusal | head -1)
G2G_BASENAME=$(basename "$G2G_DIR")
ln -s "../$G2G_BASENAME" "docs/phase2/runs/${G2G_BASENAME%_phase2g_refusal}_phase2f_models_gpt5mini"
```

- [ ] **Step 4: Commit**

```bash
git add docs/phase2/runs/
git commit -m "chore(eval): execute Phase 2 tier 2f — answer-model comparison (gpt-4.1-mini, claude-haiku-4-5)"
```

---

## Task 18: Generate pairwise compare reports

5 chain comparisons + 2 cross-model comparisons = 7 reports under `docs/phase2/compare/`.

- [ ] **Step 1: Helper script**

Create `scripts/phase2_compare.sh`:

```bash
#!/usr/bin/env bash
# Run pairwise compare for all Phase 2 chain + model comparisons.
set -euo pipefail

mkdir -p docs/phase2/compare

# Resolve the run_id for each tier from the archived metadata.
get_run_id() {
    local tier_suffix="$1"
    local meta=$(ls docs/phase2/runs/*_${tier_suffix}/metadata.json | head -1)
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$meta"
}

BASELINE=$(get_run_id phase2_baseline)
B2B=$(get_run_id phase2b_embedder)
B2C=$(get_run_id phase2c_hybrid)
B2D=$(get_run_id phase2d_rerank)
B2E=$(get_run_id phase2e_rewrite)
B2G=$(get_run_id phase2g_refusal)
M_GPT5=$B2G
M_GPT41=$(get_run_id phase2f_models_gpt41mini)
M_HAIKU=$(get_run_id phase2f_models_haiku)

# Chain comparisons
python -m src.eval.cli compare "$BASELINE" "$B2B" --html docs/phase2/compare/2b_vs_baseline.html
python -m src.eval.cli compare "$B2B" "$B2C"     --html docs/phase2/compare/2c_vs_2b.html
python -m src.eval.cli compare "$B2C" "$B2D"     --html docs/phase2/compare/2d_vs_2c.html
python -m src.eval.cli compare "$B2D" "$B2E"     --html docs/phase2/compare/2e_vs_2d.html
python -m src.eval.cli compare "$B2E" "$B2G"     --html docs/phase2/compare/2g_vs_2e.html

# Cross-model comparisons (all on the 2g stack)
python -m src.eval.cli compare "$M_GPT5" "$M_GPT41" --html docs/phase2/compare/gpt41mini_vs_gpt5mini.html
python -m src.eval.cli compare "$M_GPT5" "$M_HAIKU" --html docs/phase2/compare/haiku_vs_gpt5mini.html

echo "Wrote 7 compare reports to docs/phase2/compare/"
```

```bash
chmod +x scripts/phase2_compare.sh
```

- [ ] **Step 2: Run it**

```bash
./scripts/phase2_compare.sh
```

- [ ] **Step 3: Verify 7 HTML files**

```bash
ls docs/phase2/compare/*.html | wc -l
```
Expected: `7`.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase2_compare.sh docs/phase2/compare/
git commit -m "feat(eval): pairwise compare reports for the Phase 2 matrix"
```

---

## Task 19: Write `docs/PHASE2_RESULTS.md`

**Files:**
- Create: `docs/PHASE2_RESULTS.md`
- Create: `docs/phase2/chart.png` (rendered from compare data)

- [ ] **Step 1: Generate the per-tier metric chart**

Create `scripts/phase2_chart.py`:

```python
"""Render per-tier metric chart from the archived Phase 2 run dirs.

Usage:
    python scripts/phase2_chart.py --out docs/phase2/chart.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

TIER_ORDER = [
    "phase2_baseline", "phase2b_embedder", "phase2c_hybrid",
    "phase2d_rerank", "phase2e_rewrite", "phase2g_refusal",
]
METRICS = [
    "answer_correctness", "judge_faithfulness",
    "recall_at_5", "refusal_correctness",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="docs/phase2/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    tier_to_metrics: dict[str, dict[str, tuple[float, float, float]]] = {}
    for tier in TIER_ORDER:
        match = list(runs_dir.glob(f"*_{tier}/metrics.json"))
        if not match:
            print(f"missing: {tier}")
            continue
        rows = json.loads(match[0].read_text())
        tier_to_metrics[tier] = {
            r["metric_name"]: (r["mean"], r["ci_low"], r["ci_high"])
            for r in rows if r["dataset"] == "(all)" and r["metric_name"] in METRICS
        }

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4), sharey=True)
    for ax, metric in zip(axes, METRICS):
        means, lows, highs = [], [], []
        for tier in TIER_ORDER:
            m = tier_to_metrics.get(tier, {}).get(metric, (0, 0, 0))
            means.append(m[0])
            lows.append(m[0] - m[1])
            highs.append(m[2] - m[0])
        ax.bar(range(len(TIER_ORDER)), means, yerr=[lows, highs], capsize=4)
        ax.set_title(metric)
        ax.set_xticks(range(len(TIER_ORDER)))
        ax.set_xticklabels([t.replace("phase2", "") for t in TIER_ORDER], rotation=30, ha="right")
        ax.set_ylim(0, 1)
    fig.suptitle("Phase 2 — RAG Quality Matrix (per-tier means with 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

```bash
python scripts/phase2_chart.py --out docs/phase2/chart.png
```

- [ ] **Step 2: Author the writeup**

Create `docs/PHASE2_RESULTS.md`:

```markdown
# Phase 2 — RAG Quality Matrix Results

> Layered ablation of seven RAG architectural levers on the Phase 1 SQuAD-200 baseline.
> Spec: [`docs/superpowers/specs/2026-04-27-phase2-rag-quality-matrix-design.md`](superpowers/specs/2026-04-27-phase2-rag-quality-matrix-design.md)
> Plan: [`docs/superpowers/plans/2026-04-27-phase2-rag-quality-matrix.md`](superpowers/plans/2026-04-27-phase2-rag-quality-matrix.md)

## Methodology

Phase 1 shipped an end-to-end evaluation harness with paired permutation tests and bootstrap CIs (PR #1–#4). Phase 2 uses that harness to measure five architectural levers in a layered chain — each tier inheriting from the previous tier and toggling exactly one variable — plus a parallel three-way answer-model comparison on top of the final 2g stack. The eval set is 200 SQuAD v2 dev questions; ML papers are deferred to Phase 3.

Comparisons reported in this document use:
- **Bootstrap percentile CI** (n=1000) for per-tier means.
- **Paired permutation test** (n=10000) for between-tier deltas.
- A delta is "significant" at p < 0.05.

## The matrix at a glance

![Per-tier metric chart](phase2/chart.png)

## Per-tier results

| Tier | Lever toggled | answer_correctness Δ | judge_faithfulness Δ | refusal_correctness Δ | recall@5 Δ | p (paired) |
|------|----------------|------:|------:|------:|------:|------:|
| baseline | (none) | — | — | — | — | — |
| 2b | BGE embedder | <fill from compare/2b_vs_baseline.html> | | | | |
| 2c | BM25 hybrid | | | | | |
| 2d | cross-encoder rerank | | | | | |
| 2e | query rewriting | | | | | |
| 2g | refusal handler | | | | | |

## Answer-model comparison (on the 2g stack)

| Generator model | answer_correctness | judge_faithfulness | refusal_correctness | total cost |
|-----------------|-------------------:|-------------------:|--------------------:|----------:|
| gpt-5-mini      | <fill from 2g run>      | | | |
| gpt-4.1-mini    | | | | |
| claude-haiku-4-5| | | | |

## Findings

For each lever, fill in 2-3 sentences referencing the paired-significance result. Examples:

- **2b BGE embedder.** <fill: did the swap raise recall@5 with a significant p-value? If not, was it neutral or did it regress?>
- **2c BM25 hybrid.** <fill: SQuAD's gold contexts are short; BM25's contribution was either dominant or noisy. State which.>
- **2d Cross-encoder rerank.** <fill>
- **2e Query rewriting.** <fill: cost vs lift trade-off; the cost ledger should show how much rewriting added.>
- **2g Refusal handler.** <fill: did refusal_correctness go up? Did answer_correctness regress? By how much?>
- **2f Answer-model comparison.** <fill: which model paired best with the 2g stack on cost-quality trade-off.>

## Winning stack

<fill: list the configs whose deltas were positive and significant; describe what the "winning stack" is and roughly how to reproduce it via configs/eval/phase2/<config>.yaml>

## Cost ledger

| Run | Generator | Judge | Rewriter | Total |
|-----|----------:|------:|---------:|------:|
| baseline | <from cost.json> | | | |
| 2b | | | | |
| 2c | | | | |
| 2d | | | | |
| 2e | | | | |
| 2g | | | | |
| 2f gpt-4.1-mini | | | | |
| 2f claude-haiku-4-5 | | | | |
| **Total** | | | | <≤ $5> |

## Reproducibility

```bash
# install (Phase 1 + Phase 2)
pip install -r requirements.txt

# run any tier
python -m src.eval.cli run --config configs/eval/phase2/<tier>.yaml

# regenerate compare reports
./scripts/phase2_compare.sh

# regenerate chart
python scripts/phase2_chart.py --out docs/phase2/chart.png
```

## Implementation references

- Schema: `src/eval/config.py` (PipelineCfg, EmbedderCfg, HybridCfg, RerankerCfg, QueryRewriterCfg, RefusalHandlerCfg)
- Pipeline modules: `src/eval/embedders/`, `src/eval/retrievers/`, `src/eval/transforms/`
- Factory: `src/eval/pipeline_factory.py::build_pipeline`
- Cost ledger: `EvalResult.cost_breakdown`, `aggregate_costs` in `src/eval/metrics/operational.py`
```

- [ ] **Step 3: Fill in the placeholders from the actual run data**

Open each archived `metrics.json` and `cost.json`, copy real numbers into the tables. Open each `docs/phase2/compare/*.html` and copy the paired-permutation p-values into the per-tier table. Replace every `<fill: ...>` block with concrete prose grounded in the data.

- [ ] **Step 4: Commit**

```bash
git add docs/PHASE2_RESULTS.md docs/phase2/chart.png scripts/phase2_chart.py
git commit -m "docs(eval): Phase 2 results — methodology, chart, findings"
```

---

## Task 20: Link Phase 2 results from the main README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a one-liner near the existing eval section**

Find the existing "Evaluation" section in `README.md`. Append:

```markdown
### Phase 2 — Quality matrix

Phase 2 measured 7 architectural levers (BGE embedder, BM25 hybrid, cross-encoder rerank, LLM query rewriting, answer-model sweep, refusal handler) layered on top of the Phase 1 baseline against the same 200-question SQuAD v2 dev set. See [`docs/PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md) for the chart, paired-significance results, winning stack, and cost ledger.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): link Phase 2 results from main README"
```

---

## End of PR-B — Push and open PR

- [ ] **Push and open the PR**

```bash
git push -u origin feature/phase2-experiments-writeup
gh pr create \
  --base feature/phase2-pipeline-extensions \
  --head feature/phase2-experiments-writeup \
  --title "docs(eval): Phase 2 RAG quality matrix — results + findings" \
  --body "$(cat <<'EOF'
## Summary

PR-B of Phase 2: experiment runs and writeup. Stacked on PR-A (#<PR-A number>).

- 8 distinct eval runs against SQuAD-200, archived under docs/phase2/runs/.
- 7 pairwise compare reports under docs/phase2/compare/ (5 chain + 2 model).
- docs/PHASE2_RESULTS.md: methodology, chart, paired-significance table, winning stack, per-lever findings, cost ledger.
- README updated with a link to the writeup.

Live eval_runs/ directories stay local (gitignored). Each archived run dir contains only metrics.json, cost.json, metadata.json, config.yaml. The questions.jsonl SHA is recorded in metadata.json.

## Test plan

- [x] All 8 runs landed with errors=0
- [x] All 7 compare HTML reports render and show non-empty significance numbers
- [x] Total cost across all runs is ≤ $5 (see ledger in writeup)
EOF
)"
```

---

# Self-Review

The plan author runs this checklist after writing the plan:

1. **Spec coverage:** Each spec section maps to at least one task — ✅
   - §1 goal + scope → Tasks 11–20 (data) + 1–10 (extensions)
   - §2 architecture → Tasks 3–8
   - §3.1 schema → Task 1
   - §3.2 factory → Task 8
   - §3.3 matrix → Tasks 9, 10–17
   - §4.1 unit tests → embedded in Tasks 3–7
   - §4.2 factory tests → Task 8
   - §4.3 smoke → Task 8 (`test_phase2_query_with_refusal_short_circuits`)
   - §4.6 cost ledger → Task 2
   - §5.1 PR-A commit list → Tasks 1–10
   - §5.2 PR-B commit list → Tasks 11–20
2. **Placeholder scan:** No `TBD`, `TODO`, `fill in details`, or steps that describe without showing. The writeup task (19) intentionally leaves `<fill: ...>` markers because it requires reading the live data; the steps explicitly require replacing them. ✅
3. **Type consistency:**
   - `EmbedderCfg.name` is `Literal["chroma_default", "bge_small_en_v1_5"]` everywhere.
   - `RerankerCfg.model` is `Literal["ms_marco_minilm_l6_v2"] | None` everywhere.
   - `QueryRewriter.expand` returns `tuple[list[str], float, int, int]` in the impl and the tests.
   - `RefusalHandler.refuse_response` returns `tuple[list[SearchResult], str]` in impl and tests.
   - `LLMHandler.generate_with_usage` returns `tuple[str, int, int]` in impl, tests, and the rewriter contract. ✅

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-27-phase2-rag-quality-matrix.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (Tasks 1–10 are TDD code tasks; Tasks 11–20 are run-and-archive). Review between tasks, fast iteration.

2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

**Which approach?**
