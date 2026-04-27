"""Tests for the eval harness Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.eval.schemas import (
    AggregatedMetric,
    CompareResult,
    EvalQuestion,
    EvalResult,
    MetricDelta,
    RunMetadata,
)


class TestEvalQuestion:
    def test_minimal_construction(self):
        q = EvalQuestion(id="abc", question="Why?")
        assert q.gold_answer is None
        assert q.gold_chunk_ids == []
        assert q.is_unanswerable is False
        assert q.metadata == {}

    def test_full_construction(self):
        q = EvalQuestion(
            id="abc",
            question="Why?",
            gold_answer="Because.",
            gold_chunk_ids=["c1", "c2"],
            is_unanswerable=False,
            metadata={"difficulty": "hard"},
        )
        assert q.gold_answer == "Because."
        assert q.metadata["difficulty"] == "hard"

    def test_frozen_blocks_mutation(self):
        q = EvalQuestion(id="abc", question="Why?")
        with pytest.raises(ValidationError):
            q.question = "What?"  # type: ignore[misc]


class TestEvalResult:
    def test_round_trip_json(self):
        r = EvalResult(
            question_id="abc",
            dataset="squad_v2_dev_200",
            retrieved_chunk_ids=["c1"],
            retrieved_chunks=["text"],
            generated_answer="ans",
            metrics={"recall_at_5": 1.0},
            timings_ms={"retrieve": 12.0, "generate": 100.0},
            tokens={"prompt": 100, "completion": 50},
            cost_usd=0.001,
        )
        encoded = r.model_dump_json()
        decoded = EvalResult.model_validate_json(encoded)
        assert decoded == r

    def test_error_default_none(self):
        r = EvalResult(
            question_id="abc",
            dataset="ml_papers_v1",
            retrieved_chunk_ids=[],
            retrieved_chunks=[],
            generated_answer="",
            metrics={},
            timings_ms={},
            tokens={"prompt": 0, "completion": 0},
            cost_usd=0.0,
        )
        assert r.error is None


class TestAggregatedMetric:
    def test_construction(self):
        m = AggregatedMetric(
            metric_name="recall_at_5",
            mean=0.84,
            ci_low=0.81,
            ci_high=0.87,
            n=200,
        )
        assert m.dataset is None

    def test_with_dataset(self):
        m = AggregatedMetric(
            metric_name="faithfulness",
            dataset="ml_papers_v1",
            mean=0.91,
            ci_low=0.85,
            ci_high=0.96,
            n=50,
        )
        assert m.dataset == "ml_papers_v1"


class TestRunMetadata:
    def test_construction(self):
        now = datetime.now(timezone.utc)
        meta = RunMetadata(
            run_id="2026-04-26_143022_baseline_a3f9c1",
            config_name="baseline",
            config_path="configs/eval/baseline.yaml",
            git_sha="a3f9c1",
            started_at=now,
            finished_at=now,
            env_hash="deadbeef",
            eval_set_versions={"squad_v2_dev_200": "abc123"},
            n_questions=200,
            n_errors=0,
        )
        assert meta.warnings == []


class TestMetricDelta:
    def test_construction(self):
        d = MetricDelta(
            metric_name="recall_at_5",
            a_mean=0.80,
            a_ci=(0.77, 0.83),
            b_mean=0.85,
            b_ci=(0.82, 0.88),
            delta=0.05,
            p_value=0.001,
            significant=True,
        )
        assert d.significant is True
        assert d.delta == pytest.approx(0.05)


class TestCompareResult:
    def test_construction(self):
        now = datetime.now(timezone.utc)
        meta_a = RunMetadata(
            run_id="A", config_name="a", config_path="a.yaml", git_sha="x",
            started_at=now, finished_at=now, env_hash="h",
            eval_set_versions={}, n_questions=10, n_errors=0,
        )
        meta_b = meta_a.model_copy(update={"run_id": "B"})
        result = CompareResult(run_a=meta_a, run_b=meta_b, deltas=[])
        assert result.per_question_diff == []
