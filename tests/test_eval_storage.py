"""Tests for src.eval.storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.eval import storage
from src.eval.schemas import (
    AggregatedMetric,
    EvalResult,
    RunMetadata,
)


@pytest.fixture
def tmp_eval_runs(tmp_path: Path, monkeypatch) -> Path:
    """Point the eval runs directory at a temp dir for the duration of a test.

    BEFORE: this set EVAL_RUNS_DIR and then `importlib.reload`ed the storage
            module, because the directory was a module-level constant bound at
            import time.
    AFTER:  storage resolves the directory per call, so setting the variable is
            enough — and every storage function also accepts `base_dir=` for
            callers that would rather inject than set an environment variable.
    """
    runs_dir = tmp_path / "eval_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("EVAL_RUNS_DIR", str(runs_dir))
    return runs_dir


def _make_metadata(run_id: str = "test-run") -> RunMetadata:
    now = datetime.now(UTC)
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
        ts = datetime(2026, 4, 26, 14, 30, 22, tzinfo=UTC)
        rid = storage.compute_run_id("baseline", ts, "a3f9c1abcdef")
        assert rid == "2026-04-26_143022_baseline_a3f9c1a"

    def test_deterministic(self, tmp_eval_runs):
        ts = datetime(2026, 4, 26, 14, 30, 22, tzinfo=UTC)
        rid1 = storage.compute_run_id("x", ts, "abc1234567")
        rid2 = storage.compute_run_id("x", ts, "abc1234567")
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
        run_dir = tmp_eval_runs / meta.run_id
        storage.save_run(
            run_dir, meta, results, aggregated, cost, "name: test\n"
        )

        loaded = storage.load_run(meta.run_id)
        assert loaded["metadata"] == meta
        assert loaded["results"] == results
        assert loaded["aggregated"] == aggregated
        assert loaded["cost"] == cost

    def test_files_created(self, tmp_eval_runs):
        meta = _make_metadata("test-run-2")
        run_dir = tmp_eval_runs / meta.run_id
        storage.save_run(run_dir, meta, [], [], {}, "name: test\n")
        for f in ["metadata.json", "questions.jsonl", "metrics.json", "cost.json", "config.yaml"]:
            assert (run_dir / f).exists(), f"Missing {f}"

    def test_load_missing_raises(self, tmp_eval_runs):
        with pytest.raises(FileNotFoundError):
            storage.load_run("does-not-exist")


class TestListRuns:
    def test_lists_completed_runs_descending(self, tmp_eval_runs):
        # Create two runs with distinct timestamps.
        meta_old = _make_metadata("old-run")
        meta_old = meta_old.model_copy(update={
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
        })
        meta_new = _make_metadata("new-run")
        meta_new = meta_new.model_copy(update={
            "started_at": datetime(2026, 4, 1, tzinfo=UTC),
            "finished_at": datetime(2026, 4, 1, tzinfo=UTC),
        })
        for m in (meta_old, meta_new):
            run_dir = tmp_eval_runs / m.run_id
            storage.save_run(run_dir, m, [], [], {}, "x: y\n")
        runs = storage.list_runs()
        assert [r.run_id for r in runs] == ["new-run", "old-run"]

    def test_ignores_dirs_without_metadata(self, tmp_eval_runs):
        (tmp_eval_runs / "incomplete-run").mkdir()
        assert storage.list_runs() == []

    def test_empty_dir_returns_empty(self, tmp_eval_runs):
        assert storage.list_runs() == []


class TestDeleteRun:
    def test_removes_run_dir(self, tmp_eval_runs):
        meta = _make_metadata("doomed-run")
        run_dir = tmp_eval_runs / meta.run_id
        storage.save_run(run_dir, meta, [], [], {}, "x: y\n")
        assert run_dir.exists()
        storage.delete_run(meta.run_id)
        assert not run_dir.exists()

    def test_refuses_path_traversal(self, tmp_eval_runs):
        with pytest.raises(ValueError):
            storage.delete_run("../etc")
        with pytest.raises(ValueError):
            storage.delete_run("a/b")


class TestInjectableRunsDirectory:
    """Callers can pass the runs directory instead of setting an env var.

    The directory used to be a module-level constant, so the only way to
    redirect it was to reassign another module's global — which the CLI did at
    four call sites and which forced tests to re-import the module.
    """

    def test_save_and_load_via_explicit_base_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVAL_RUNS_DIR", raising=False)
        base = tmp_path / "elsewhere"
        base.mkdir()
        meta = _make_metadata("injected-run")
        storage.save_run(base / meta.run_id, meta, [], [], {}, "x: y\n")

        assert storage.load_run(meta.run_id, base_dir=base)["metadata"].run_id == meta.run_id
        assert [m.run_id for m in storage.list_runs(base_dir=base)] == [meta.run_id]

        storage.delete_run(meta.run_id, base_dir=base)
        assert storage.list_runs(base_dir=base) == []

    def test_runs_dir_is_resolved_per_call(self, tmp_path, monkeypatch):
        """No module reload needed — this is what the old workaround existed for."""
        monkeypatch.setenv("EVAL_RUNS_DIR", str(tmp_path / "one"))
        assert storage.runs_dir() == tmp_path / "one"
        monkeypatch.setenv("EVAL_RUNS_DIR", str(tmp_path / "two"))
        assert storage.runs_dir() == tmp_path / "two"

    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("EVAL_RUNS_DIR", raising=False)
        assert storage.runs_dir() == Path(storage.DEFAULT_RUNS_DIRNAME)
