"""Tests for src.eval.compare."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.compare import compare_runs
from src.eval.schemas import (
    AggregatedMetric,
    EvalResult,
    MetricDelta,
    RunMetadata,
)


def _make_metadata(run_id: str, versions: dict[str, str] | None = None) -> RunMetadata:
    now = datetime.now(timezone.utc)
    return RunMetadata(
        run_id=run_id, config_name=run_id, config_path=f"{run_id}.yaml",
        git_sha="x" * 7, started_at=now, finished_at=now, env_hash="h",
        eval_set_versions=versions or {"squad_v2_dev_200": "v1"},
        n_questions=10, n_errors=0,
    )


def _r(qid: str, dataset: str, score: float) -> EvalResult:
    return EvalResult(
        question_id=qid, dataset=dataset,
        retrieved_chunk_ids=[], retrieved_chunks=[],
        generated_answer="", metrics={"recall_at_5": score},
        timings_ms={}, tokens={"prompt": 0, "completion": 0}, cost_usd=0.0,
    )


def _agg(metric_name: str, dataset: str | None, mean: float, n: int = 10) -> AggregatedMetric:
    return AggregatedMetric(
        metric_name=metric_name, dataset=dataset,
        mean=mean, ci_low=mean - 0.05, ci_high=mean + 0.05, n=n,
    )


@pytest.fixture
def tmp_eval_runs(tmp_path, monkeypatch):
    runs = tmp_path / "eval_runs"
    runs.mkdir()
    monkeypatch.setenv("EVAL_RUNS_DIR", str(runs))
    import importlib
    import src.eval.storage
    importlib.reload(src.eval.storage)
    yield src.eval.storage
    monkeypatch.delenv("EVAL_RUNS_DIR", raising=False)
    importlib.reload(src.eval.storage)


def _save_synthetic_run(storage, run_id: str, results: list[EvalResult],
                        aggregated: list[AggregatedMetric],
                        versions: dict[str, str] | None = None) -> None:
    meta = _make_metadata(run_id, versions=versions)
    storage.save_run(
        storage.EVAL_RUNS_DIR / run_id, meta, results, aggregated,
        {"total_usd": 0.0, "mean_usd_per_query": 0.0},
        f"name: {run_id}\n",
    )


class TestCompareRuns:
    def test_constant_shift_significant(self, tmp_eval_runs):
        # Run A: scores 0.5 for all 10 questions; Run B: scores 0.6 for all.
        results_a = [_r(f"q{i}", "squad_v2_dev_200", 0.5) for i in range(10)]
        results_b = [_r(f"q{i}", "squad_v2_dev_200", 0.6) for i in range(10)]
        agg_a = [_agg("recall_at_5", "squad_v2_dev_200", 0.5),
                 _agg("recall_at_5", None, 0.5)]
        agg_b = [_agg("recall_at_5", "squad_v2_dev_200", 0.6),
                 _agg("recall_at_5", None, 0.6)]
        _save_synthetic_run(tmp_eval_runs, "A", results_a, agg_a)
        _save_synthetic_run(tmp_eval_runs, "B", results_b, agg_b)

        result = compare_runs("A", "B")
        assert result.run_a.run_id == "A"
        assert result.run_b.run_id == "B"
        # All deltas ≈ +0.1
        for d in result.deltas:
            assert d.delta == pytest.approx(0.1, abs=0.01)
            assert d.significant is True

    def test_version_mismatch_raises(self, tmp_eval_runs):
        results_a = [_r("q1", "squad_v2_dev_200", 0.5)]
        results_b = [_r("q1", "squad_v2_dev_200", 0.6)]
        agg = [_agg("recall_at_5", "squad_v2_dev_200", 0.5)]
        _save_synthetic_run(tmp_eval_runs, "A", results_a, agg,
                            versions={"squad_v2_dev_200": "v1"})
        _save_synthetic_run(tmp_eval_runs, "B", results_b, agg,
                            versions={"squad_v2_dev_200": "v2"})

        with pytest.raises(ValueError, match="eval set version mismatch"):
            compare_runs("A", "B")

    def test_per_question_diff_sorted_and_capped(self, tmp_eval_runs):
        # 12 questions; 5 with delta=+0.5, 5 with delta=-0.5, 2 with delta=0
        results_a = []
        results_b = []
        for i in range(5):
            results_a.append(_r(f"big_pos_{i}", "squad_v2_dev_200", 0.0))
            results_b.append(_r(f"big_pos_{i}", "squad_v2_dev_200", 0.5))
        for i in range(5):
            results_a.append(_r(f"big_neg_{i}", "squad_v2_dev_200", 0.5))
            results_b.append(_r(f"big_neg_{i}", "squad_v2_dev_200", 0.0))
        for i in range(2):
            results_a.append(_r(f"flat_{i}", "squad_v2_dev_200", 0.5))
            results_b.append(_r(f"flat_{i}", "squad_v2_dev_200", 0.5))
        agg = [_agg("recall_at_5", "squad_v2_dev_200", 0.4),
               _agg("recall_at_5", None, 0.4)]
        _save_synthetic_run(tmp_eval_runs, "A", results_a, agg)
        _save_synthetic_run(tmp_eval_runs, "B", results_b, agg)

        result = compare_runs("A", "B")
        # At most 10 entries
        assert len(result.per_question_diff) <= 10
        # Sorted by absolute delta descending
        deltas = [abs(d["delta"]) for d in result.per_question_diff]
        assert deltas == sorted(deltas, reverse=True)
        # Flat questions should be excluded (or at least not at the top)
        flat_in_top = sum(1 for d in result.per_question_diff if d["question_id"].startswith("flat_"))
        assert flat_in_top == 0
