"""Tests for src.eval.statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.statistics import bootstrap_ci


SEED = 12345


class TestBootstrapCI:
    def test_known_distribution_brackets_true_mean(self):
        """The 95% CI on a normal sample should usually contain the true mean."""
        rng = np.random.default_rng(SEED)
        true_mean = 5.0
        sample = rng.normal(loc=true_mean, scale=1.0, size=200)

        mean, low, high = bootstrap_ci(sample.tolist(), n_resamples=1000, seed=SEED)
        assert low < true_mean < high
        assert mean == pytest.approx(float(np.mean(sample)))

    def test_seed_reproducibility(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        a = bootstrap_ci(values, n_resamples=500, seed=SEED)
        b = bootstrap_ci(values, n_resamples=500, seed=SEED)
        assert a == b

    def test_drops_nan_values(self):
        values = [1.0, 2.0, float("nan"), 3.0, float("nan"), 4.0]
        mean, _, _ = bootstrap_ci(values, n_resamples=200, seed=SEED)
        assert mean == pytest.approx(2.5)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([float("nan"), float("nan")], n_resamples=100, seed=SEED)

    def test_single_value_yields_zero_width_ci(self):
        mean, low, high = bootstrap_ci([7.0], n_resamples=100, seed=SEED)
        assert mean == 7.0
        assert low == 7.0
        assert high == 7.0
