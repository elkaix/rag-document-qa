"""Tests for src.eval.submission — starting a run without going through HTTP.

The orchestration these exercise used to live inside a FastAPI route handler,
so the only way to reach it was an HTTP request. That is why its failure path
had no coverage and why the progress defect survived: nothing could call the
joint between runner and registry directly.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from src.eval.submission import (
    ConfigNotFoundError,
    RunProgressSink,
    reserve_run_id,
    resolve_config,
    submit_run,
)


class RecordingSink:
    """A progress sink that records the lifecycle it was told about."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def register(self, run_id: str, n_total: int) -> None:
        self.events.append(("register", run_id, n_total))

    def update_progress(self, run_id: str, n_completed: int, n_total=None) -> None:
        self.events.append(("progress", run_id, n_completed, n_total))

    def mark_completed(self, run_id: str) -> None:
        self.events.append(("completed", run_id))

    def mark_failed(self, run_id: str, error_message: str) -> None:
        self.events.append(("failed", run_id, error_message))


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    return d


class TestResolveConfig:
    def test_missing_config_raises_a_translatable_error(self, configs_dir):
        with pytest.raises(ConfigNotFoundError, match="nope"):
            resolve_config("nope", configs_dir)

    def test_error_is_a_filenotfound(self, configs_dir):
        """So callers that only care about the broad category still catch it."""
        assert issubclass(ConfigNotFoundError, FileNotFoundError)


class TestReserveRunId:
    def test_is_stable_for_a_fixed_submission_time(self):
        from datetime import datetime

        when = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)
        assert reserve_run_id("baseline", when) == reserve_run_id("baseline", when)

    def test_embeds_the_config_name_and_a_sortable_timestamp(self):
        from datetime import datetime

        when = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)
        run_id = reserve_run_id("baseline", when)
        assert run_id.startswith("2026-09-09_120000_baseline_")


class TestSubmitRunLifecycle:
    def test_a_missing_config_registers_nothing(self, configs_dir):
        """A bad name must not leave an orphan entry the status endpoint reports."""
        sink = RecordingSink()
        with pytest.raises(ConfigNotFoundError):
            submit_run("nope", configs_dir=configs_dir, progress=sink)
        assert sink.events == []

    def test_a_failing_run_is_marked_failed(self, configs_dir, monkeypatch):
        """The failure path had no coverage before submission was extracted."""
        sink = RecordingSink()
        (configs_dir / "boom.yaml").write_text("name: boom\n")

        monkeypatch.setattr(
            "src.eval.submission.load_config", lambda path: object()
        )

        class _Exploding:
            def __init__(self, *a, **kw): ...
            def run(self):
                raise RuntimeError("dataset unavailable")

        monkeypatch.setattr("src.eval.submission.EvalRunner", _Exploding)

        result = submit_run("boom", configs_dir=configs_dir, progress=sink)

        assert ("register", result.run_id, 0) in sink.events
        assert ("failed", result.run_id, "dataset unavailable") in sink.events
        assert not any(e[0] == "completed" for e in sink.events)

    def test_a_successful_run_is_marked_completed_and_forwards_the_total(
        self, configs_dir, monkeypatch
    ):
        sink = RecordingSink()
        (configs_dir / "ok.yaml").write_text("name: ok\n")
        monkeypatch.setattr("src.eval.submission.load_config", lambda path: object())

        class _Runner:
            def __init__(self, *a, on_progress=None, **kw):
                self._on_progress = on_progress

            def run(self):
                self._on_progress(2, 10)

        monkeypatch.setattr("src.eval.submission.EvalRunner", _Runner)

        result = submit_run("ok", configs_dir=configs_dir, progress=sink)

        assert ("progress", result.run_id, 2, 10) in sink.events, "total must reach the sink"
        assert ("completed", result.run_id) in sink.events

    def test_a_reserved_id_is_the_one_the_run_uses(self, configs_dir, monkeypatch):
        """No 'override' reconciling two independent derivations."""
        sink = RecordingSink()
        (configs_dir / "ok.yaml").write_text("name: ok\n")
        monkeypatch.setattr("src.eval.submission.load_config", lambda path: object())

        seen = {}

        class _Runner:
            def __init__(self, *a, run_id=None, **kw):
                seen["run_id"] = run_id

            def run(self): ...

        monkeypatch.setattr("src.eval.submission.EvalRunner", _Runner)

        reserved = reserve_run_id("ok")
        result = submit_run(
            "ok", configs_dir=configs_dir, progress=sink, run_id=reserved
        )
        assert seen["run_id"] == reserved == result.run_id


class TestRegistrySatisfiesTheSink:
    def test_run_registry_is_a_valid_progress_sink(self):
        """The eval package declares what it needs instead of importing the API."""
        from src.api.services.eval_runs import RunRegistry

        registry = RunRegistry()
        assert isinstance(registry, RunProgressSink)
