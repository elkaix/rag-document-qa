"""
Operational aggregators — latency, cost, token totals over a run.

Eval Harness Position:
  list[EvalResult] → [OPERATIONAL] → run-level summary in metrics.json
                      ^^^^^^^^^^^^
  Pure aggregations. Errored questions are skipped (their timings/
  costs are not representative of healthy pipeline behavior).

Design decisions:
  - p50/p95/p99 via numpy.percentile with default 'linear' interpolation
    — the standard convention; matches how tools like Datadog report
    percentiles. Don't switch to 'nearest' without good reason.
  - Errored results excluded from all aggregates: a 30-second timeout
    on a broken question would skew p99 misleadingly. The error count
    lives in RunMetadata.n_errors.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.eval.schemas import EvalResult


def _healthy(results: Iterable[EvalResult]) -> list[EvalResult]:
    """Return only results where the pipeline did not raise."""
    return [r for r in results if r.error is None]


def aggregate_timings(results: Iterable[EvalResult]) -> dict[str, dict[str, float]]:
    """Compute p50/p95/p99 latency percentiles per pipeline stage.

    Stage names are auto-discovered from the healthy results so the
    function works with any timings_ms schema without configuration.
    Results where a given stage is absent are skipped for that stage
    (not treated as zero — a missing stage means the step didn't run).

    Args:
        results: Iterable of EvalResult instances from a run.

    Returns:
        Mapping of stage_name → {"p50": ms, "p95": ms, "p99": ms}.
        Empty dict if there are no healthy results.
    """
    healthy = _healthy(results)
    if not healthy:
        return {}

    # WHY: Discover stages dynamically — avoids hardcoding stage names
    # and naturally handles runs with different pipeline configurations.
    stage_names: set[str] = {stage for r in healthy for stage in r.timings_ms}

    output: dict[str, dict[str, float]] = {}
    for stage in stage_names:
        # Collect values only where the stage is present in that result.
        values = [r.timings_ms[stage] for r in healthy if stage in r.timings_ms]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        output[stage] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    return output


def aggregate_costs(results: Iterable[EvalResult]) -> dict[str, float]:
    """Compute total and mean cost in USD across healthy results, with per-bucket breakdown.

    Args:
        results: Iterable of EvalResult instances from a run.

    Returns:
        Dict with keys "total_usd", "mean_usd_per_query", "generator_total_usd",
        "judge_total_usd", "rewriter_total_usd". All are 0.0 when there are no
        healthy results.
    """
    healthy = _healthy(results)
    if not healthy:
        return {
            "total_usd": 0.0,
            "mean_usd_per_query": 0.0,
            "generator_total_usd": 0.0,
            "judge_total_usd": 0.0,
            "rewriter_total_usd": 0.0,
        }

    costs = [r.cost_usd for r in healthy]
    total = float(sum(costs))
    mean = total / len(costs)
    generator_total = sum(r.cost_breakdown.get("generator", 0.0) for r in healthy)
    judge_total = sum(r.cost_breakdown.get("judge", 0.0) for r in healthy)
    rewriter_total = sum(r.cost_breakdown.get("rewriter", 0.0) for r in healthy)
    return {
        "total_usd": total,
        "mean_usd_per_query": mean,
        "generator_total_usd": generator_total,
        "judge_total_usd": judge_total,
        "rewriter_total_usd": rewriter_total,
    }


def aggregate_tokens(results: Iterable[EvalResult]) -> dict[str, float | int]:
    """Compute total and mean prompt/completion token counts.

    Args:
        results: Iterable of EvalResult instances from a run.

    Returns:
        Dict with keys "total_prompt", "total_completion",
        "mean_prompt", "mean_completion". All are 0 when there are
        no healthy results.
    """
    healthy = _healthy(results)
    if not healthy:
        return {
            "total_prompt": 0,
            "total_completion": 0,
            "mean_prompt": 0.0,
            "mean_completion": 0.0,
        }

    prompt_counts = [r.tokens.get("prompt", 0) for r in healthy]
    completion_counts = [r.tokens.get("completion", 0) for r in healthy]

    total_prompt = sum(prompt_counts)
    total_completion = sum(completion_counts)
    n = len(healthy)

    return {
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "mean_prompt": total_prompt / n,
        "mean_completion": total_completion / n,
    }
