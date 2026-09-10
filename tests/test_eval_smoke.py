"""End-of-sub-plan-1A smoke test: verifies the package surface is
importable and the metric pieces compose end-to-end without a runner."""

from __future__ import annotations

import math


def test_top_level_imports():
    from src.eval import (
        MODEL_PRICES,
        AggregatedMetric,
        CompareResult,
        EvalQuestion,
        EvalResult,
        MetricDelta,
        RunMetadata,
        bootstrap_ci,
        cost_usd,
        paired_permutation_test,
    )

    assert callable(bootstrap_ci)
    assert callable(paired_permutation_test)
    assert callable(cost_usd)
    assert MODEL_PRICES
    for cls in (
        AggregatedMetric, CompareResult, EvalQuestion,
        EvalResult, MetricDelta, RunMetadata,
    ):
        assert isinstance(cls, type)


def test_compose_retrieval_then_aggregate():
    """Running retrieval metrics across a synthetic dev-set composes correctly."""
    from src.eval import EvalQuestion, bootstrap_ci
    from src.eval.metrics.retrieval import recall_at_k

    questions = [
        EvalQuestion(id=str(i), question="Q?", gold_chunk_ids=["c1"])
        for i in range(50)
    ]
    retrieved_per_q = [["c1", "x"] if i % 5 != 0 else ["x", "y"] for i in range(50)]
    recalls = [
        recall_at_k(q.gold_chunk_ids, ret, k=5)
        for q, ret in zip(questions, retrieved_per_q, strict=False)
    ]
    assert sum(recalls) / len(recalls) == 0.8

    mean, low, high = bootstrap_ci(recalls, n_resamples=200, seed=42)
    assert mean == 0.8
    assert low < 0.8 < high or math.isclose(low, 0.8) or math.isclose(high, 0.8)
