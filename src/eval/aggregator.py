"""
Metric aggregator — turns per-question EvalResult rows into AggregatedMetric
rows with bootstrap confidence intervals.

Eval Harness Position:
  list[EvalResult] → [AGGREGATOR] → list[AggregatedMetric] (per-dataset + combined)
                                  → list[warnings] (skipped low-N combos)

Design decisions:
  - Per-dataset rows let the UI compare squad_v2 vs ml_papers performance.
  - Combined rows give a single headline number across all datasets.
  - Skip combos with <3 samples — bootstrap CIs are meaningless on n<3.
  - Use config.eval.bootstrap_n and config.eval.seed so all runs with the
    same config yield reproducible CIs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.eval.config import EvalConfig
from src.eval.schemas import AggregatedMetric, EvalResult
from src.eval.statistics import bootstrap_ci

MIN_SAMPLES = 3


def aggregate(
    results: list[EvalResult],
    config: EvalConfig,
) -> tuple[list[AggregatedMetric], list[str]]:
    """For every (metric_name, dataset) combo present in results, compute
    bootstrap CIs and return list[AggregatedMetric] + list[warnings].

    Behavior:
      - Per-dataset rows: emit one AggregatedMetric per (metric, dataset).
      - Combined row: emit one AggregatedMetric per metric with dataset=None.
      - Skip metric/dataset combos with <3 non-NaN samples; append a
        warning string (e.g. "Skipped recall_at_5 on ml_papers_v1: only 2 samples")
        to the warnings list.
      - NaN values are dropped per metric per question by bootstrap_ci.
      - Errored results (r.error is not None) are excluded.

    Args:
        results: Per-question evaluation results from the runner.
        config: Eval run configuration (provides bootstrap_n and seed).

    Returns:
        A two-tuple of (aggregated_metrics, warnings) where aggregated_metrics
        is a list of AggregatedMetric (per-dataset and combined) and warnings
        is a list of strings describing skipped low-N combos.
    """
    # WHY: Filter errored results first so downstream grouping never sees them.
    #      An errored result has undefined metric values — including it would
    #      silently skew aggregates if the metrics dict happens to be non-empty.
    valid = [r for r in results if r.error is None]

    # PATTERN: Collect raw score lists keyed by (metric_name, dataset).
    #          defaultdict(list) avoids repeated "if key not in" guards.
    per_dataset: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in valid:
        for metric_name, score in r.metrics.items():
            per_dataset[(metric_name, r.dataset)].append(score)

    # Build the combined (across all datasets) view keyed by metric_name alone.
    # WHY: Combine after grouping so we don't double-count results — iterate
    #      the already-filtered per_dataset dict rather than valid again.
    combined: dict[str, list[float]] = defaultdict(list)
    for (metric_name, _dataset), scores in per_dataset.items():
        combined[metric_name].extend(scores)

    aggregated: list[AggregatedMetric] = []
    warnings: list[str] = []

    bootstrap_n = config.eval.bootstrap_n
    seed = config.eval.seed

    # --- Per-dataset rows ---
    for (metric_name, dataset), scores in per_dataset.items():
        n = len(scores)
        if n < MIN_SAMPLES:
            warnings.append(
                f"Skipped {metric_name} on {dataset}: only {n} samples"
            )
            continue

        mean, ci_low, ci_high = bootstrap_ci(scores, n_resamples=bootstrap_n, seed=seed)
        aggregated.append(AggregatedMetric(
            metric_name=metric_name,
            dataset=dataset,
            mean=mean,
            ci_low=ci_low,
            ci_high=ci_high,
            n=n,
        ))

    # --- Combined rows (dataset=None) ---
    for metric_name, scores in combined.items():
        n = len(scores)
        if n < MIN_SAMPLES:
            warnings.append(
                f"Skipped {metric_name} combined: only {n} samples"
            )
            continue

        mean, ci_low, ci_high = bootstrap_ci(scores, n_resamples=bootstrap_n, seed=seed)
        aggregated.append(AggregatedMetric(
            metric_name=metric_name,
            dataset=None,
            mean=mean,
            ci_low=ci_low,
            ci_high=ci_high,
            n=n,
        ))

    return aggregated, warnings
