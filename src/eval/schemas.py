"""
Pydantic schemas for the RAG eval harness.

Eval Harness Position:
  EvalRunner → Pipeline → Metrics → AggregatedMetric → Storage
                  ^^^         ^^^         ^^^^^^^^^^^^^^^^^^^^
  These types are the contracts that tie the layers together. Every
  layer reads/writes one of these models — no untyped dicts cross
  module boundaries.

Design decisions:
  - Pydantic v2 over plain dataclasses: free JSON round-trip (we
    persist results as JSON Lines) and field validation at boundaries.
  - frozen=True on EvalQuestion to prevent accidental mutation after
    a labeled gold set is loaded — ground truth is immutable.
  - Modern T | None syntax (Python 3.10+).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalQuestion(BaseModel):
    """A single labeled question from a gold eval set.

    Teaches: immutable value objects via frozen=True. The gold set is
    ground truth — no runtime code should mutate it after load.

    Pipeline role: INPUT to EvalRunner; defines the question + expected
    answer that all metrics are computed against.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    gold_answer: str | None = None
    gold_chunk_ids: list[str] = Field(default_factory=list)
    is_unanswerable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Output record for a single question evaluation run.

    Teaches: JSON Lines persistence pattern — one record per question,
    serialized with model_dump_json() and stored line-by-line so large
    eval runs can be streamed without loading the full set into memory.

    Pipeline role: OUTPUT of EvalRunner per question; INPUT to the
    aggregation layer that computes AggregatedMetric summaries.
    """

    question_id: str
    dataset: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[str]
    generated_answer: str
    metrics: dict[str, float]
    metric_details: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float]
    tokens: dict[str, int]
    cost_usd: float
    # Phase 2: per-bucket breakdown. Defaults to generator-only when absent so
    # Phase 1 records continue to round-trip through model_validate.
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def _backfill_cost_breakdown(self) -> "EvalResult":
        if not self.cost_breakdown:
            self.cost_breakdown = {
                "generator": self.cost_usd,
                "judge": 0.0,
                "rewriter": 0.0,
            }
        return self


class AggregatedMetric(BaseModel):
    """Aggregate statistics for one metric across a dataset (or all datasets).

    Teaches: confidence intervals as first-class citizens — never report
    a mean without CI bounds. n is explicit so downstream callers can
    detect low-sample results and weight them appropriately.

    Pipeline role: OUTPUT of the aggregation layer; INPUT to the
    reporting and comparison layers.
    """

    metric_name: str
    dataset: str | None = Field(default=None, description="None = combined across datasets")
    mean: float
    ci_low: float
    ci_high: float
    n: int


class RunMetadata(BaseModel):
    """Provenance record for a complete eval run.

    Teaches: reproducibility by design — every run captures the git SHA,
    config path, and env hash so results can be traced back to exact
    code + config + environment. This is the audit trail.

    Pipeline role: Written once per run; attached to CompareResult so
    A/B comparisons always carry full provenance for both runs.
    """

    run_id: str
    config_name: str
    config_path: str
    git_sha: str
    started_at: datetime
    finished_at: datetime
    env_hash: str
    eval_set_versions: dict[str, str]
    n_questions: int
    n_errors: int
    warnings: list[str] = Field(default_factory=list)


class MetricDelta(BaseModel):
    """Statistical comparison of one metric between two runs.

    Teaches: significance testing as a schema concern — delta alone is
    misleading without a p-value and CI bounds. Consumers should gate
    decisions on significant=True, not raw delta magnitude.

    Pipeline role: Element of CompareResult.deltas; one per
    (metric, dataset) pair being compared.
    """

    metric_name: str
    dataset: str | None = None
    a_mean: float
    a_ci: tuple[float, float]
    b_mean: float
    b_ci: tuple[float, float]
    delta: float
    p_value: float
    significant: bool


class CompareResult(BaseModel):
    """Full A/B comparison between two eval runs.

    Teaches: structured comparison output — embedding run provenance
    (RunMetadata) directly in the result means the comparison is
    self-contained and doesn't require external lookups to interpret.

    Pipeline role: Terminal output of the comparison layer; consumed
    by reporting tools and the CI gate that blocks regressions.
    """

    run_a: RunMetadata
    run_b: RunMetadata
    deltas: list[MetricDelta]
    per_question_diff: list[dict[str, Any]] = Field(default_factory=list)
