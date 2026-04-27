"""Tests for src.eval.metrics.retrieval — Recall@k, MRR@k, nDCG@k."""

from __future__ import annotations

import math

import pytest

from src.eval.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c", "x", "y"], 5) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "x", "y", "z", "w"], 5) == pytest.approx(1 / 3)

    def test_zero_recall(self):
        assert recall_at_k(["a", "b", "c"], ["x", "y", "z"], 3) == pytest.approx(0.0)

    def test_truncation_misses(self):
        # k=2 cuts before a and b appear
        assert recall_at_k(["a", "b"], ["x", "y", "a", "b"], 2) == pytest.approx(0.0)

    def test_truncation_hits(self):
        # k=4 includes both
        assert recall_at_k(["a", "b"], ["x", "y", "a", "b"], 4) == pytest.approx(1.0)

    def test_empty_gold_is_nan(self):
        assert math.isnan(recall_at_k([], ["a", "b"], 5))

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], ["a"], 10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MRR@k
# ---------------------------------------------------------------------------


class TestMrrAtK:
    def test_rank_1(self):
        assert mrr_at_k(["a"], ["a", "b", "c"], 5) == pytest.approx(1.0)

    def test_rank_2(self):
        assert mrr_at_k(["a"], ["x", "a", "b"], 5) == pytest.approx(0.5)

    def test_no_hit(self):
        assert mrr_at_k(["a"], ["x", "y", "z"], 3) == pytest.approx(0.0)

    def test_multi_gold_first_hit(self):
        # b is at rank 2, a is at rank 3 — MRR uses FIRST hit (b, rank 2 → 1/2)
        assert mrr_at_k(["a", "b"], ["x", "b", "a"], 5) == pytest.approx(0.5)

    def test_truncation_misses(self):
        # a is at position 4 (0-indexed 3), k=3 cuts it off
        assert mrr_at_k(["a"], ["x", "y", "z", "a"], 3) == pytest.approx(0.0)

    def test_truncation_hits(self):
        # k=4 includes a at rank 4 → 1/4
        assert mrr_at_k(["a"], ["x", "y", "z", "a"], 4) == pytest.approx(0.25)

    def test_empty_gold_is_nan(self):
        assert math.isnan(mrr_at_k([], ["a"], 5))


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["a"], ["x", "y", "z"], 3) == pytest.approx(0.0)

    def test_single_hit_rank_1(self):
        # DCG = 1/log2(2) = 1, IDCG = 1 → nDCG = 1.0
        assert ndcg_at_k(["a"], ["a", "x"], 2) == pytest.approx(1.0)

    def test_single_hit_rank_2(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1 → nDCG = 1/log2(3)
        assert ndcg_at_k(["a"], ["x", "a"], 2) == pytest.approx(1.0 / math.log2(3))

    def test_partial_hit(self):
        # gold=["a","b"], retrieved=["x","a"], k=2
        # DCG = 1/log2(3) (a at rank 2, b not present)
        # IDCG = 1/log2(2) + 1/log2(3) (ideal: both hits at ranks 1 and 2)
        dcg = 1.0 / math.log2(3)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(["a", "b"], ["x", "a"], 2) == pytest.approx(dcg / idcg)

    def test_empty_gold_is_nan(self):
        assert math.isnan(ndcg_at_k([], ["a"], 5))
