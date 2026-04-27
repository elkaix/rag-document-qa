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


def paired_permutation_test(
    a: list[float],
    b: list[float],
    n_resamples: int = 10000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Two-sided paired permutation test on the mean difference.

    The null hypothesis is that the labels "a" and "b" are exchangeable
    on a per-pair basis (i.e. swapping ``a[i]`` and ``b[i]`` for any
    subset of indices doesn't change the joint distribution). We
    estimate the p-value by randomly sign-flipping each paired
    difference and counting how often the resampled mean difference
    is at least as extreme as the observed one.

    Args:
        a: First sample (e.g. per-question metric for run A).
        b: Second sample, paired one-to-one with ``a``.
        n_resamples: Number of permutation draws.
        seed: PRNG seed for reproducibility.

    Returns:
        ``(observed_delta, p_value)`` where ``observed_delta`` is the
        mean of ``b[i] - a[i]`` over surviving pairs and ``p_value``
        is two-sided.

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths or no
            pairs survive NaN-drop.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Paired samples must have equal length: len(a)={len(a)}, len(b)={len(b)}"
        )
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    # Drop pairs where either side is NaN.
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    diffs = arr_b[mask] - arr_a[mask]
    if diffs.size == 0:
        raise ValueError("No surviving pairs after NaN-drop")

    observed = float(diffs.mean())

    # WHY sign-flips: under the exchangeability null, swapping a[i] and
    #     b[i] negates the i-th difference. Random sign-flips draw from
    #     the exact null distribution of the mean difference.
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diffs.size))
    resampled_means = (signs * diffs).mean(axis=1)

    # Two-sided p-value with Phipson & Smyth (2010) correction (+1/+1)
    # to prevent meaningless p == 0.0.
    extreme = np.sum(np.abs(resampled_means) >= abs(observed))
    p_value = float((extreme + 1) / (n_resamples + 1))

    return observed, p_value
