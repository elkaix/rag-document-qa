"""Tests for src.eval.metrics.operational."""

from __future__ import annotations

import pytest

from src.eval.metrics.operational import (
    aggregate_costs,
    aggregate_timings,
    aggregate_tokens,
)
from src.eval.schemas import EvalResult


def _make_result(
    *,
    retrieve_ms: float = 100.0,
    generate_ms: float = 1000.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost: float = 0.001,
    error: str | None = None,
) -> EvalResult:
    return EvalResult(
        question_id="q",
        dataset="d",
        retrieved_chunk_ids=[],
        retrieved_chunks=[],
        generated_answer="",
        metrics={},
        timings_ms={"retrieve": retrieve_ms, "generate": generate_ms},
        tokens={"prompt": prompt_tokens, "completion": completion_tokens},
        cost_usd=cost,
        error=error,
    )


class TestAggregateTimings:
    def test_basic_percentiles(self):
        results = [_make_result(retrieve_ms=float(i)) for i in range(1, 11)]
        agg = aggregate_timings(results)
        assert agg["retrieve"]["p50"] == pytest.approx(5.5)
        assert agg["retrieve"]["p95"] == pytest.approx(9.55)
        assert agg["retrieve"]["p99"] == pytest.approx(9.91)

    def test_skips_errored_results(self):
        results = [
            _make_result(retrieve_ms=10.0),
            _make_result(retrieve_ms=99999.0, error="boom"),
        ]
        agg = aggregate_timings(results)
        assert agg["retrieve"]["p50"] == pytest.approx(10.0)

    def test_empty_input_returns_empty_dict(self):
        assert aggregate_timings([]) == {}


class TestAggregateCosts:
    def test_total_and_mean(self):
        results = [_make_result(cost=0.01), _make_result(cost=0.03)]
        agg = aggregate_costs(results)
        assert agg["total_usd"] == pytest.approx(0.04)
        assert agg["mean_usd_per_query"] == pytest.approx(0.02)

    def test_skips_errored(self):
        results = [
            _make_result(cost=0.01),
            _make_result(cost=999.0, error="boom"),
        ]
        agg = aggregate_costs(results)
        assert agg["total_usd"] == pytest.approx(0.01)


class TestAggregateTokens:
    def test_totals_and_means(self):
        results = [
            _make_result(prompt_tokens=100, completion_tokens=50),
            _make_result(prompt_tokens=200, completion_tokens=100),
        ]
        agg = aggregate_tokens(results)
        assert agg["total_prompt"] == 300
        assert agg["total_completion"] == 150
        assert agg["mean_prompt"] == pytest.approx(150.0)
        assert agg["mean_completion"] == pytest.approx(75.0)
