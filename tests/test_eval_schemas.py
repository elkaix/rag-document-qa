"""Tests for the eval harness Pydantic schemas.

Coverage: EvalQuestion, EvalResult, AggregatedMetric, RunMetadata,
          MetricDelta, CompareResult — validation, defaults, and edge cases.
"""

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


# --------------------------------------------------------------------------- #
# EvalQuestion                                                                  #
# --------------------------------------------------------------------------- #

class TestEvalQuestion:
    def test_minimal_valid(self):
        """Only required fields — optional ones get defaults."""
        q = EvalQuestion(question="What is RAG?", expected_answer="A retrieval technique.")
        assert q.question == "What is RAG?"
        assert q.expected_answer == "A retrieval technique."
        assert q.doc_ids == []
        assert q.metadata == {}

    def test_full_fields(self):
        q = EvalQuestion(
            question="What is RAG?",
            expected_answer="Retrieval-Augmented Generation.",
            doc_ids=["doc1", "doc2"],
            metadata={"difficulty": "easy", "topic": "rag"},
        )
        assert q.doc_ids == ["doc1", "doc2"]
        assert q.metadata["topic"] == "rag"

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            EvalQuestion(expected_answer="An answer.")

    def test_missing_expected_answer_raises(self):
        with pytest.raises(ValidationError):
            EvalQuestion(question="What?")

    def test_empty_question_raises(self):
        """Blank question string should be rejected."""
        with pytest.raises(ValidationError):
            EvalQuestion(question="", expected_answer="Something.")

    def test_empty_expected_answer_raises(self):
        with pytest.raises(ValidationError):
            EvalQuestion(question="What?", expected_answer="")


# --------------------------------------------------------------------------- #
# EvalResult                                                                    #
# --------------------------------------------------------------------------- #

class TestEvalResult:
    def _base(self, **overrides) -> dict:
        base = {
            "question": "What is RAG?",
            "expected_answer": "A retrieval technique.",
            "generated_answer": "RAG stands for Retrieval-Augmented Generation.",
            "retrieved_doc_ids": ["doc1"],
            "scores": {"faithfulness": 0.9, "relevance": 0.85},
        }
        base.update(overrides)
        return base

    def test_minimal_valid(self):
        r = EvalResult(**self._base())
        assert r.question == "What is RAG?"
        assert r.scores["faithfulness"] == pytest.approx(0.9)
        assert r.latency_ms is None
        assert r.error is None

    def test_latency_and_error(self):
        r = EvalResult(**self._base(latency_ms=123.4, error="timeout"))
        assert r.latency_ms == pytest.approx(123.4)
        assert r.error == "timeout"

    def test_scores_empty_allowed(self):
        """An empty scores dict is valid — metrics are computed separately."""
        r = EvalResult(**self._base(scores={}))
        assert r.scores == {}

    def test_score_out_of_range_raises(self):
        """Score values must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            EvalResult(**self._base(scores={"faithfulness": 1.5}))

    def test_score_negative_raises(self):
        with pytest.raises(ValidationError):
            EvalResult(**self._base(scores={"relevance": -0.1}))

    def test_missing_required_field_raises(self):
        data = self._base()
        del data["generated_answer"]
        with pytest.raises(ValidationError):
            EvalResult(**data)


# --------------------------------------------------------------------------- #
# AggregatedMetric                                                              #
# --------------------------------------------------------------------------- #

class TestAggregatedMetric:
    def test_valid(self):
        m = AggregatedMetric(name="faithfulness", mean=0.85, std=0.05, n=20)
        assert m.name == "faithfulness"
        assert m.mean == pytest.approx(0.85)
        assert m.n == 20

    def test_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            AggregatedMetric(name="x", mean=0.5, std=0.1, n=0)

    def test_std_non_negative(self):
        with pytest.raises(ValidationError):
            AggregatedMetric(name="x", mean=0.5, std=-0.01, n=5)

    def test_mean_bounds(self):
        with pytest.raises(ValidationError):
            AggregatedMetric(name="x", mean=1.1, std=0.0, n=1)

    def test_mean_zero_valid(self):
        m = AggregatedMetric(name="x", mean=0.0, std=0.0, n=1)
        assert m.mean == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# RunMetadata                                                                   #
# --------------------------------------------------------------------------- #

class TestRunMetadata:
    def test_defaults(self):
        m = RunMetadata(run_id="run-abc", dataset="golden_50")
        assert m.run_id == "run-abc"
        assert m.dataset == "golden_50"
        assert isinstance(m.created_at, datetime)
        assert m.created_at.tzinfo is not None  # must be tz-aware
        assert m.tags == []

    def test_custom_timestamp(self):
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        m = RunMetadata(run_id="r", dataset="d", created_at=ts)
        assert m.created_at == ts

    def test_tags(self):
        m = RunMetadata(run_id="r", dataset="d", tags=["v2", "nightly"])
        assert "nightly" in m.tags

    def test_missing_run_id_raises(self):
        with pytest.raises(ValidationError):
            RunMetadata(dataset="d")

    def test_missing_dataset_raises(self):
        with pytest.raises(ValidationError):
            RunMetadata(run_id="r")


# --------------------------------------------------------------------------- #
# MetricDelta                                                                   #
# --------------------------------------------------------------------------- #

class TestMetricDelta:
    def test_positive_delta(self):
        d = MetricDelta(name="faithfulness", baseline=0.7, candidate=0.85)
        assert d.delta == pytest.approx(0.15)
        assert d.improved is True

    def test_negative_delta(self):
        d = MetricDelta(name="relevance", baseline=0.9, candidate=0.8)
        assert d.delta == pytest.approx(-0.1)
        assert d.improved is False

    def test_zero_delta(self):
        d = MetricDelta(name="x", baseline=0.5, candidate=0.5)
        assert d.delta == pytest.approx(0.0)
        assert d.improved is False

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            MetricDelta(baseline=0.5, candidate=0.6)


# --------------------------------------------------------------------------- #
# CompareResult                                                                 #
# --------------------------------------------------------------------------- #

class TestCompareResult:
    def _make_meta(self, run_id: str) -> RunMetadata:
        return RunMetadata(run_id=run_id, dataset="golden_50")

    def _make_deltas(self) -> list[MetricDelta]:
        return [
            MetricDelta(name="faithfulness", baseline=0.7, candidate=0.85),
            MetricDelta(name="relevance", baseline=0.9, candidate=0.88),
        ]

    def test_valid(self):
        cr = CompareResult(
            baseline=self._make_meta("run-A"),
            candidate=self._make_meta("run-B"),
            deltas=self._make_deltas(),
        )
        assert cr.baseline.run_id == "run-A"
        assert cr.candidate.run_id == "run-B"
        assert len(cr.deltas) == 2

    def test_empty_deltas_allowed(self):
        cr = CompareResult(
            baseline=self._make_meta("A"),
            candidate=self._make_meta("B"),
            deltas=[],
        )
        assert cr.deltas == []

    def test_missing_candidate_raises(self):
        with pytest.raises(ValidationError):
            CompareResult(baseline=self._make_meta("A"), deltas=[])

    def test_missing_baseline_raises(self):
        with pytest.raises(ValidationError):
            CompareResult(candidate=self._make_meta("B"), deltas=[])
