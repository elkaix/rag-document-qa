"""
Pydantic schemas for the RAG evaluation harness.

RAG Pipeline Position:
  Document → Chunks → Embeddings → Vector Store → Retrieval → Generation
                                                                    |
                                                             [EVALUATION]
                                                                 ^^^
  This module defines the data contracts for every evaluation artifact:
  questions, per-question results, aggregated metrics, run metadata,
  and cross-run comparisons.

What concept it teaches:
  Pydantic v2 data modelling with validation constraints. Separating
  *data contracts* from *computation logic* keeps both layers testable
  in isolation and makes serialisation (JSON, YAML) trivial.

Why Pydantic over plain dataclasses:
  - Built-in validation with descriptive errors at schema boundaries.
  - JSON round-trip via model_dump() / model_validate() with zero extra code.
  - `Field(...)` constraints (ge, le, gt) express domain rules as schema,
    not scattered if-statements.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# EvalQuestion — a single question/answer pair in the evaluation dataset       #
# --------------------------------------------------------------------------- #

class EvalQuestion(BaseModel):
    """One question from the evaluation dataset.

    PATTERN: Separating the *question spec* from the *result* means the same
    dataset can be replayed across many runs without mutation.
    """

    question: str = Field(..., min_length=1, description="The natural-language question.")
    expected_answer: str = Field(..., min_length=1, description="Ground-truth reference answer.")

    # WHY list not set: order matters for display; duplicates are harmless.
    doc_ids: list[str] = Field(
        default_factory=list,
        description="Optional document IDs that contain the answer (for retrieval eval).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra fields (topic, difficulty, source, …).",
    )


# --------------------------------------------------------------------------- #
# EvalResult — one model answer scored against one EvalQuestion               #
# --------------------------------------------------------------------------- #

class EvalResult(BaseModel):
    """Scored output for a single question in an evaluation run.

    TRADE-OFF: Storing both question and answer in one record (rather than a
    foreign-key join) makes each result self-contained for logging and replay.
    The redundancy is acceptable at evaluation scale (hundreds, not millions).
    """

    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    generated_answer: str = Field(..., description="The model's actual output.")
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks returned by the retriever for this question.",
    )

    # WHY float range [0, 1]: normalised scores simplify aggregation and
    # comparison across metrics with different natural scales.
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Metric name → score in [0.0, 1.0].",
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="End-to-end latency in milliseconds (optional).",
    )
    error: str | None = Field(
        default=None,
        description="Error message if this question failed during evaluation.",
    )

    @field_validator("scores")
    @classmethod
    def scores_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject any score outside [0.0, 1.0].

        WHY validator not Field constraint: dicts of floats can't use ge/le
        directly; a field_validator lets us inspect every value.
        """
        for name, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Score '{name}' = {score!r} is outside [0.0, 1.0]."
                )
        return v


# --------------------------------------------------------------------------- #
# AggregatedMetric — summary statistics for one metric across a run           #
# --------------------------------------------------------------------------- #

class AggregatedMetric(BaseModel):
    """Descriptive statistics for a single metric across an evaluation run.

    PATTERN: Keeping mean/std/n together (rather than separate fields) means
    you can pass one object to a comparison function without decomposing it.
    """

    name: str = Field(..., min_length=1, description="Metric name, e.g. 'faithfulness'.")
    mean: float = Field(..., ge=0.0, le=1.0, description="Arithmetic mean across all questions.")
    std: float = Field(..., ge=0.0, description="Standard deviation (0 when n == 1).")
    n: int = Field(..., gt=0, description="Number of questions contributing to this aggregate.")


# --------------------------------------------------------------------------- #
# RunMetadata — identity and context for one evaluation run                   #
# --------------------------------------------------------------------------- #

class RunMetadata(BaseModel):
    """Identifies a single evaluation run.

    WHY a dedicated schema: run_id + dataset are the primary key for comparing
    runs. Keeping them in a first-class model lets CompareResult reference two
    runs cleanly without repeating string fields everywhere.
    """

    run_id: str = Field(..., min_length=1, description="Unique identifier for this run.")
    dataset: str = Field(..., min_length=1, description="Dataset name / version used.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of run creation.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary labels (e.g. model version, experiment branch).",
    )


# --------------------------------------------------------------------------- #
# MetricDelta — change in one metric between baseline and candidate           #
# --------------------------------------------------------------------------- #

class MetricDelta(BaseModel):
    """Absolute difference for one metric between two evaluation runs.

    WHY computed field for delta: storing it derived (not raw) prevents
    baseline/candidate/delta from going out of sync if a record is patched.
    """

    name: str = Field(..., min_length=1)
    baseline: float = Field(..., description="Baseline run's mean for this metric.")
    candidate: float = Field(..., description="Candidate run's mean for this metric.")

    # PATTERN: computed fields — delta and improved are always derived from
    # baseline/candidate, never set independently.
    delta: float = Field(
        default=0.0,
        description="candidate − baseline (positive = improvement).",
    )
    improved: bool = Field(
        default=False,
        description="True when delta > 0.",
    )

    @model_validator(mode="after")
    def compute_delta(self) -> MetricDelta:
        """Derive delta and improved from baseline/candidate after construction.

        WHY model_validator not default: defaults are evaluated before the
        other fields are set, so we need a post-init hook to compute derived
        values that depend on sibling fields.
        """
        self.delta = self.candidate - self.baseline
        self.improved = self.delta > 0.0
        return self


# --------------------------------------------------------------------------- #
# CompareResult — full comparison between a baseline and a candidate run      #
# --------------------------------------------------------------------------- #

class CompareResult(BaseModel):
    """Side-by-side comparison of two evaluation runs.

    RAG Pipeline Position:
      This is the OUTPUT of the evaluation pipeline — it aggregates per-run
      results and deltas so a caller can decide whether a new model/config
      is an improvement over the current production baseline.
    """

    baseline: RunMetadata = Field(..., description="The reference / production run.")
    candidate: RunMetadata = Field(..., description="The new run being evaluated.")
    deltas: list[MetricDelta] = Field(
        default_factory=list,
        description="Per-metric deltas (candidate − baseline).",
    )
