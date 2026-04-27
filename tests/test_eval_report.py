"""Tests for src.eval.report."""

from __future__ import annotations

from datetime import datetime, timezone

from src.eval.report import render_compare_html, render_run_html
from src.eval.schemas import (
    AggregatedMetric,
    CompareResult,
    EvalResult,
    MetricDelta,
    RunMetadata,
)


def _meta(run_id: str = "test") -> RunMetadata:
    now = datetime.now(timezone.utc)
    return RunMetadata(
        run_id=run_id, config_name="baseline",
        config_path="x.yaml", git_sha="abc1234",
        started_at=now, finished_at=now, env_hash="h",
        eval_set_versions={"squad_v2_dev_200": "v1"},
        n_questions=2, n_errors=0,
    )


def _result(qid: str) -> EvalResult:
    return EvalResult(
        question_id=qid, dataset="squad_v2_dev_200",
        retrieved_chunk_ids=[], retrieved_chunks=[],
        generated_answer="ans", metrics={"recall_at_5": 1.0},
        timings_ms={}, tokens={"prompt": 10, "completion": 5}, cost_usd=0.0001,
    )


class TestRenderRunHtml:
    def test_basic_render(self):
        run = {
            "metadata": _meta("run-1"),
            "results": [_result("q1"), _result("q2")],
            "aggregated": [
                AggregatedMetric(metric_name="recall_at_5", mean=1.0,
                                 ci_low=1.0, ci_high=1.0, n=2),
            ],
            "cost": {"total_usd": 0.0002, "mean_usd_per_query": 0.0001,
                     "total_prompt": 20, "total_completion": 10},
        }
        html = render_run_html(run)
        assert "<table" in html
        assert "run-1" in html
        assert "recall_at_5" in html
        assert "q1" in html and "q2" in html


class TestRenderCompareHtml:
    def test_basic_render_with_significant_marker(self):
        result = CompareResult(
            run_a=_meta("A"),
            run_b=_meta("B"),
            deltas=[
                MetricDelta(
                    metric_name="recall_at_5",
                    dataset="squad_v2_dev_200",
                    a_mean=0.5, a_ci=(0.45, 0.55),
                    b_mean=0.7, b_ci=(0.65, 0.75),
                    delta=0.2, p_value=0.001, significant=True,
                ),
                MetricDelta(
                    metric_name="faithfulness",
                    dataset=None,
                    a_mean=0.9, a_ci=(0.85, 0.95),
                    b_mean=0.91, b_ci=(0.86, 0.96),
                    delta=0.01, p_value=0.5, significant=False,
                ),
            ],
            per_question_diff=[
                {"question_id": "q1", "dataset": "squad_v2_dev_200",
                 "a_score": 0.0, "b_score": 1.0, "delta": 1.0},
            ],
        )
        html = render_compare_html(result)
        assert "<table" in html
        assert "A" in html and "B" in html
        # Significant delta gets the star marker
        assert "★" in html
