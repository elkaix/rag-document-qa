"""
Two-run comparison — diff aggregated metrics between two eval runs with
paired significance tests, plus per-question regressions/wins.

Eval Harness Position:
  load_run(A) ─┐
                ├─→ [COMPARE] → CompareResult (deltas + per_question_diff)
  load_run(B) ─┘

Design decisions:
  - Validate eval_set_versions match before comparing — different versions
    mean different questions, and a delta is meaningless across them.
  - Paired permutation test (vs unpaired t-test) because the same question
    appears in both runs; the pairing reduces noise from question variance.
  - Headline metric chosen as recall_at_5 if present, else alphabetic first
    — predictable and reproducible per_question_diff selection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.eval.schemas import (
    AggregatedMetric,
    CompareResult,
    EvalResult,
    MetricDelta,
    RunMetadata,
)
from src.eval.statistics import paired_permutation_test
from src.eval.storage import load_run

logger = logging.getLogger(__name__)


def _score_index(
    results: list[EvalResult],
) -> dict[tuple[str, str, str], float]:
    """Build a lookup table from (question_id, dataset, metric) → score.

    WHY: Flat dict keyed by 3-tuple lets us O(1)-look up any specific
    (question, dataset, metric) combo when pairing across runs A and B.

    Args:
        results: Per-question eval outputs for one run.

    Returns:
        Dict mapping (question_id, dataset, metric_name) to float score.
    """
    index: dict[tuple[str, str, str], float] = {}
    for r in results:
        for metric, score in r.metrics.items():
            index[(r.question_id, r.dataset, metric)] = score
    return index


def _agg_lookup(
    aggregated: list[AggregatedMetric],
) -> dict[tuple[str, str | None], AggregatedMetric]:
    """Index aggregated metrics by (metric_name, dataset).

    Args:
        aggregated: Aggregated metric list from a run.

    Returns:
        Dict for O(1) lookup by (metric_name, dataset).
    """
    return {(a.metric_name, a.dataset): a for a in aggregated}


def _paired_values(
    scores_a: dict[tuple[str, str, str], float],
    scores_b: dict[tuple[str, str, str], float],
    metric: str,
    dataset: str,
) -> tuple[list[float], list[float], list[str]]:
    """Extract paired per-question scores for a given (metric, dataset).

    Only includes question IDs that appear in both runs with non-NaN scores.

    Args:
        scores_a: Score index for run A.
        scores_b: Score index for run B.
        metric: Metric name to filter on.
        dataset: Dataset name to filter on.

    Returns:
        Three-tuple (a_values, b_values, question_ids) where all three lists
        are aligned by position.
    """
    # Gather question_ids that have a score in run A for this (metric, dataset).
    candidates = {
        qid
        for (qid, ds, m) in scores_a
        if ds == dataset and m == metric
    }
    a_vals: list[float] = []
    b_vals: list[float] = []
    qids: list[str] = []
    for qid in sorted(candidates):  # sorted for determinism
        a_score = scores_a.get((qid, dataset, metric))
        b_score = scores_b.get((qid, dataset, metric))
        if a_score is None or b_score is None:
            continue
        import math
        if math.isnan(a_score) or math.isnan(b_score):
            continue
        a_vals.append(a_score)
        b_vals.append(b_score)
        qids.append(qid)
    return a_vals, b_vals, qids


def compare_runs(id_a: str, id_b: str) -> CompareResult:
    """Load two runs, compute per-metric deltas with paired permutation
    significance tests, and pick the top per-question diffs.

    Raises:
        FileNotFoundError: If either run is missing.
        ValueError: If the two runs used different eval_set_versions.
    """
    # WHY: We import load_run at call time (not module level) so that
    # monkeypatched EVAL_RUNS_DIR is resolved after the test fixture reloads
    # src.eval.storage. The import at module level is fine because the function
    # itself references EVAL_RUNS_DIR only at call time inside storage.py.
    import src.eval.storage as _storage

    run_a = _storage.load_run(id_a)
    run_b = _storage.load_run(id_b)

    meta_a: RunMetadata = run_a["metadata"]
    meta_b: RunMetadata = run_b["metadata"]

    # Step 2: Validate eval_set_versions match.
    # TRADE-OFF: Strict equality — even a subset mismatch means the question
    # pools differ, making per-question deltas misleading.
    if meta_a.eval_set_versions != meta_b.eval_set_versions:
        raise ValueError("eval set version mismatch between runs")

    results_a: list[EvalResult] = run_a["results"]
    results_b: list[EvalResult] = run_b["results"]
    agg_a: list[AggregatedMetric] = run_a["aggregated"]
    agg_b: list[AggregatedMetric] = run_b["aggregated"]

    # Step 3: Build per-question score indexes.
    scores_a = _score_index(results_a)
    scores_b = _score_index(results_b)

    # Step 4: Index aggregated metrics and find combos present in both runs.
    lookup_a = _agg_lookup(agg_a)
    lookup_b = _agg_lookup(agg_b)
    shared_combos = set(lookup_a.keys()) & set(lookup_b.keys())

    deltas: list[MetricDelta] = []

    # WHY: sort key coerces None dataset to "" so None and str are comparable.
    for metric_name, dataset in sorted(shared_combos, key=lambda t: (t[0], t[1] or "")):
        # dataset=None entries represent cross-dataset rollups; per-question
        # scores always carry a real dataset name so we can't pair them.
        # Skip None-dataset combos: there are no EvalResult rows with dataset=None.
        if dataset is None:
            logger.debug(
                "Skipping dataset=None combo for metric '%s' (no per-question pairing possible)",
                metric_name,
            )
            continue

        a_vals, b_vals, qids = _paired_values(scores_a, scores_b, metric_name, dataset)

        if len(a_vals) < 3:
            logger.debug(
                "Skipping (%s, %s): only %d paired questions (need ≥ 3)",
                metric_name,
                dataset,
                len(a_vals),
            )
            continue

        delta, p_value = paired_permutation_test(a_vals, b_vals, n_resamples=10000, seed=42)
        significant = p_value < 0.05

        agg_entry_a = lookup_a[(metric_name, dataset)]
        agg_entry_b = lookup_b[(metric_name, dataset)]

        deltas.append(
            MetricDelta(
                metric_name=metric_name,
                dataset=dataset,
                a_mean=agg_entry_a.mean,
                a_ci=(agg_entry_a.ci_low, agg_entry_a.ci_high),
                b_mean=agg_entry_b.mean,
                b_ci=(agg_entry_b.ci_low, agg_entry_b.ci_high),
                delta=delta,
                p_value=p_value,
                significant=significant,
            )
        )

    # Step 5: Pick headline metric for per-question diff.
    # PATTERN: Prefer recall_at_5 for consistency across evals;
    # fall back to alphabetically-first metric for reproducibility.
    all_metrics = sorted({m for (m, _) in shared_combos})
    headline = "recall_at_5" if "recall_at_5" in all_metrics else (all_metrics[0] if all_metrics else None)

    # Step 6: Compute per-question diffs for the headline metric.
    # Collect across all real datasets (exclude None) where headline metric appears.
    per_question_rows: list[dict[str, Any]] = []

    if headline is not None:
        # Gather all (dataset) combos that use the headline metric (excluding None).
        headline_datasets = sorted(
            {ds for (m, ds) in shared_combos if m == headline and ds is not None}
        )
        for ds in headline_datasets:
            a_vals, b_vals, qids = _paired_values(scores_a, scores_b, headline, ds)
            for qid, a_score, b_score in zip(qids, a_vals, b_vals):
                raw_delta = b_score - a_score
                per_question_rows.append({
                    "question_id": qid,
                    "dataset": ds,
                    "a_score": a_score,
                    "b_score": b_score,
                    "delta": raw_delta,
                })

    # Sort by absolute delta descending, then cap at top 10.
    per_question_rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    top10 = per_question_rows[:10]

    return CompareResult(
        run_a=meta_a,
        run_b=meta_b,
        deltas=deltas,
        per_question_diff=top10,
    )
