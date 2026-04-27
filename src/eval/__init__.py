"""RAG eval harness — schemas, metrics, statistics, datasets.

See docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md
for the full design. This package exposes the pure-Python foundation;
the runner, CLI, and API live in separate modules introduced by later
sub-plans.
"""

from __future__ import annotations

from src.eval.pricing import MODEL_PRICES, ModelPrice, cost_usd
from src.eval.schemas import (
    AggregatedMetric,
    CompareResult,
    EvalQuestion,
    EvalResult,
    MetricDelta,
    RunMetadata,
)
from src.eval.statistics import bootstrap_ci, paired_permutation_test

__all__ = [
    "AggregatedMetric",
    "CompareResult",
    "EvalQuestion",
    "EvalResult",
    "MetricDelta",
    "MODEL_PRICES",
    "ModelPrice",
    "RunMetadata",
    "bootstrap_ci",
    "cost_usd",
    "paired_permutation_test",
]
