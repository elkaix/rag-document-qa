"""
Statistical wrappers — bootstrap confidence intervals and paired
permutation tests for eval-run comparison.

Eval Harness Position:
  per-question scores → [STATISTICS] → AggregatedMetric (with CIs)
                                     → MetricDelta      (with p-values)

Design decisions:
  - Bootstrap percentile method (not BCa) — simpler, well-understood,
    sufficient for the precision we report. With n=200 samples and
    n_resamples=1000 the CI is stable to ~1%.
  - NaN values are dropped per-metric per-call: a question that didn't
    have a defined metric (e.g. recall_at_k with empty gold) doesn't
    poison the aggregate.
  - All randomness is seeded; identical inputs produce identical CIs.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Compute the sample mean and a 95% bootstrap percentile confidence interval.

    Bootstrap resampling draws ``n_resamples`` samples of size ``n`` with
    replacement from ``arr``, computes the mean of each resample, and then
    takes the 2.5th and 97.5th percentiles of those means as the CI bounds.

    Args:
        values: Raw per-question metric scores. May contain NaN — they are
            dropped before resampling.
        n_resamples: Number of bootstrap resamples. Higher values reduce
            Monte-Carlo noise in the CI bounds but cost more compute.
            1000 is stable to ~1% for n ≥ 50.
        seed: RNG seed for reproducibility. Identical seed + values produce
            identical output.

    Returns:
        A three-tuple ``(mean, ci_low, ci_high)`` where:
          - ``mean``    is the sample mean of the non-NaN values,
          - ``ci_low``  is the 2.5th percentile of resampled means (lower CI),
          - ``ci_high`` is the 97.5th percentile of resampled means (upper CI).

    Raises:
        ValueError: If all values are NaN and no valid observations remain.
    """
    # WHY: Convert to float64 array first so NaN detection via np.isnan works
    #      uniformly regardless of the input list's original dtype.
    arr = np.array(values, dtype=np.float64)

    # Drop NaN entries: a missing metric on one question shouldn't skew the
    # aggregate or cause all-NaN resamples.
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap CI on all-NaN input")

    # PATTERN: Short-circuit for single values — resampling a length-1 array
    #          always produces the same mean, so skip the loop.
    if arr.size == 1:
        v = float(arr[0])
        return (v, v, v)

    n = arr.size

    # WHY: default_rng (PCG64) is the recommended NumPy RNG since 1.17.
    #      It is faster and statistically superior to np.random.seed().
    rng = np.random.default_rng(seed)

    # Generate all resample index matrices in one shot: shape (n_resamples, n).
    # This vectorised form avoids a Python-level loop and is ~10–50× faster
    # than calling rng.choice() inside a loop.
    indices = rng.integers(0, n, size=(n_resamples, n))

    # TRADE-OFF: arr[indices] materialises an (n_resamples, n) float64 array
    # in memory. For n=200 and n_resamples=1000 that's 200 KB — negligible.
    # For n=100k you would want chunked computation; not needed here.
    resampled_means = arr[indices].mean(axis=1)

    mean = float(arr.mean())
    ci_low = float(np.percentile(resampled_means, 2.5))
    ci_high = float(np.percentile(resampled_means, 97.5))

    return (mean, ci_low, ci_high)
