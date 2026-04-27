"""Tests for src.eval.storage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.schemas import (
    AggregatedMetric,
    EvalResult,
    RunMetadata,
)


@pytest.fixture
def tmp_eval_runs(tmp_path: Path, monkeypatch):
    """Set EVAL_RUNS_DIR to a temp dir and re-import storage to pick it up."""
    runs_dir = tmp_path / "eval_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("EVAL_RUNS_DIR", str(runs_dir))
    # Force re-evaluation of EVAL_RUNS_DIR by re-importing.
    import importlib
    import src.eval.storage
    importlib.reload(src.eval.storage)
    yield src.eval.storage
    # cleanup: reload back to default for other tests
    monkeypatch.delenv("EVAL_RUNS_DIR", raising=False)
    importlib.reload(src.eval.storage)


def _make_metadata(run_id: str = "test-run") -> RunMetadata:
    now = datetime.now(timezone.utc)
    return RunMetadata(
        run_id=run_id,
        config_name="baseline",
        config_path="configs/eval/baseline.yaml",
        git_sha="abc1234",
        started_at=now,
        finished_at=now,
        env_hash="deadbeef",
        eval_set_versions={"squad_v2_dev_200": "v1"},
        n_questions=2,
        n_errors=0,
    )


def _make_result(qid: str = "q1") -> EvalResult:
    return EvalResult(
        question_id=qid, dataset="squad_v2_dev_200",
        retrieved_chunk_ids=["c1"], retrieved_chunks=["text"],
        generated_answer="ans", metrics={"recall_at_5": 1.0},
        timings_ms={"retrieve": 12.0, "generate": 100.0},
        tokens={"prompt": 50, "completion": 25}, cost_usd=0.001,
    )


class TestComputeRunId:
    def test_format(self, tmp_eval_runs):
        ts = datetime(2026, 4, 26, 14, 30, 22, tzinfo=timezone.utc)
        rid = tmp_eval_runs.compute_run_id("baseline", ts, "a3f9c1abcdef")
        assert rid == "2026-04-26_143022_baseline_a3f9c1a"

    def test_deterministic(self, tmp_eval_runs):
        ts = datetime(2026, 4, 26, 14, 30, 22, tzinfo=timezone.utc)
        rid1 = tmp_eval_runs.compute_run_id("x", ts, "abc1234567")
        rid2 = tmp_eval_runs.compute_run_id("x", ts, "abc1234567")
        assert rid1 == rid2


class TestSaveAndLoadRun:
    def test_round_trip(self, tmp_eval_runs):
        meta = _make_metadata("test-run-1")
        results = [_make_result("q1"), _make_result("q2")]
        aggregated = [
            AggregatedMetric(
                metric_name="recall_at_5", mean=1.0,
                ci_low=1.0, ci_high=1.0, n=2,
            )
        ]
        cost = {"total_usd": 0.002, "mean_usd_per_query": 0.001}
        run_dir = tmp_eval_runs.EVAL_RUNS_DIR / meta.run_id
        tmp_eval_runs.save_run(
            run_dir, meta, results, aggregated, cost, "name: test\n"
        )

        loaded = tmp_eval_runs.load_run(meta.run_id)
        assert loaded["metadata"] == meta
        assert loaded["results"] == results
        assert loaded["aggregated"] == aggregated
        assert loaded["cost"] == cost

    def test_files_created(self, tmp_eval_runs):
        meta = _make_metadata("test-run-2")
        run_dir = tmp_eval_runs.EVAL_RUNS_DIR / meta.run_id
        tmp_eval_runs.save_run(run_dir, meta, [], [], {}, "name: test\n")
        for f in ["metadata.json", "questions.jsonl", "metrics.json", "cost.json", "config.yaml"]:
            assert (run_dir / f).exists(), f"Missing {f}"

    def test_load_missing_raises(self, tmp_eval_runs):
        with pytest.raises(FileNotFoundError):
            tmp_eval_runs.load_run("does-not-exist")


class TestListRuns:
    def test_lists_completed_runs_descending(self, tmp_eval_runs):
        # Create two runs with distinct timestamps.
        meta_old = _make_metadata("old-run")
        meta_old = meta_old.model_copy(update={
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        })
        meta_new = _make_metadata("new-run")
        meta_new = meta_new.model_copy(update={
            "started_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
        })
        for m in (meta_old, meta_new):
            run_dir = tmp_eval_runs.EVAL_RUNS_DIR / m.run_id
            tmp_eval_runs.save_run(run_dir, m, [], [], {}, "x: y\n")
        runs = tmp_eval_runs.list_runs()
        assert [r.run_id for r in runs] == ["new-run", "old-run"]

    def test_ignores_dirs_without_metadata(self, tmp_eval_runs):
        (tmp_eval_runs.EVAL_RUNS_DIR / "incomplete-run").mkdir()
        assert tmp_eval_runs.list_runs() == []

    def test_empty_dir_returns_empty(self, tmp_eval_runs):
        assert tmp_eval_runs.list_runs() == []


class TestDeleteRun:
    def test_removes_run_dir(self, tmp_eval_runs):
        meta = _make_metadata("doomed-run")
        run_dir = tmp_eval_runs.EVAL_RUNS_DIR / meta.run_id
        tmp_eval_runs.save_run(run_dir, meta, [], [], {}, "x: y\n")
        assert run_dir.exists()
        tmp_eval_runs.delete_run(meta.run_id)
        assert not run_dir.exists()

    def test_refuses_path_traversal(self, tmp_eval_runs):
        with pytest.raises(ValueError):
            tmp_eval_runs.delete_run("../etc")
        with pytest.raises(ValueError):
            tmp_eval_runs.delete_run("a/b")
