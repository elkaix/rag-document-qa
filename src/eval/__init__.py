"""RAG eval harness — schemas, metrics, statistics, datasets, runner, storage, compare.

See docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md
for the full design. This package exposes the pure-Python foundation
plus the runner/storage/compare layer; the API and frontend live in
separate modules introduced by Sub-plan 1C.
"""

from __future__ import annotations

from src.eval.compare import compare_runs
from src.eval.config import EvalConfig, load_config
from src.eval.runner import EvalRunner
from src.eval.schemas import (
    AggregatedMetric,
    CompareResult,
    EvalQuestion,
    EvalResult,
    MetricDelta,
    RunMetadata,
)
from src.eval.statistics import bootstrap_ci, paired_permutation_test
from src.eval.storage import list_runs, load_run, save_run
# Pricing moved to the core telemetry package; re-exported here so the eval
# package's public API (`from src.eval import cost_usd`) is preserved.
from src.telemetry.pricing import MODEL_PRICES, ModelPrice, cost_usd

__all__ = [
    # 1A — schemas, pricing, statistics
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
    # 1B — config, runner, storage, compare
    "EvalConfig",
    "EvalRunner",
    "compare_runs",
    "list_runs",
    "load_config",
    "load_run",
    "save_run",
]
