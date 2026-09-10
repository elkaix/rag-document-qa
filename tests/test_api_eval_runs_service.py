"""Tests for src.api.services.eval_runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from src.api.services.eval_runs import RunRegistry


class TestBasicLifecycle:
    def test_register_then_get(self):
        reg = RunRegistry()
        reg.register("r1", n_total=10)
        s = reg.get("r1")
        assert s is not None
        assert s.run_id == "r1"
        assert s.status == "queued"
        assert s.n_completed == 0
        assert s.n_total == 10

    def test_update_progress_transitions_to_running(self):
        reg = RunRegistry()
        reg.register("r1", n_total=10)
        reg.update_progress("r1", 3)
        s = reg.get("r1")
        assert s.status == "running"
        assert s.n_completed == 3

    def test_mark_completed(self):
        reg = RunRegistry()
        reg.register("r1", n_total=10)
        reg.mark_completed("r1")
        s = reg.get("r1")
        assert s.status == "completed"
        assert s.n_completed == 10
        assert s.completed_at is not None

    def test_mark_failed(self):
        reg = RunRegistry()
        reg.register("r1", n_total=10)
        reg.mark_failed("r1", "boom")
        s = reg.get("r1")
        assert s.status == "failed"
        assert s.error_message == "boom"
        assert s.completed_at is not None

    def test_get_unknown_returns_none(self):
        assert RunRegistry().get("nope") is None


class TestListActive:
    def test_only_returns_queued_or_running(self):
        reg = RunRegistry()
        reg.register("queued", 10)
        reg.register("running", 10)
        reg.update_progress("running", 1)
        reg.register("done", 10)
        reg.mark_completed("done")
        reg.register("err", 10)
        reg.mark_failed("err", "x")

        active_ids = {s.run_id for s in reg.list_active()}
        assert active_ids == {"queued", "running"}


class TestConcurrentUpdates:
    def test_concurrent_progress_updates_remain_consistent(self):
        reg = RunRegistry()
        reg.register("r1", n_total=100)

        def bump(i: int):
            reg.update_progress("r1", i)

        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(bump, range(100)))

        s = reg.get("r1")
        # The final n_completed should be one of the values written;
        # the important invariant is no exceptions and not None.
        assert s is not None
        assert 0 <= s.n_completed <= 100


class TestEviction:
    def test_evicts_completed_after_ttl(self):
        reg = RunRegistry()
        reg.register("old", 10)
        reg.mark_completed("old")
        # Forge an older completed_at to simulate elapsed time.
        s = reg.get("old")
        s.completed_at = datetime.now(UTC) - timedelta(seconds=7200)

        reg.register("new", 10)
        reg.mark_completed("new")

        evicted = reg.evict_old(ttl_seconds=3600.0)
        assert evicted == 1
        assert reg.get("old") is None
        assert reg.get("new") is not None

    def test_does_not_evict_active(self):
        reg = RunRegistry()
        reg.register("active", 10)
        evicted = reg.evict_old(ttl_seconds=0.0)
        assert evicted == 0
        assert reg.get("active") is not None


class TestProgressReporting:
    """Regression tests for the progress defect found in the 2026-09-09 review.

    BEFORE: the route registered every run with n_total=0 and its progress
            callback discarded the runner's `total` argument, so n_total stayed
            0 forever and the polling endpoint reported 0.0 for the whole run
            and then jumped to 1.0. Registry and runner were each unit-tested;
            the joint between them was not.
    """

    def test_update_progress_records_total_when_supplied(self):
        """The runner learns the question count only after loading datasets."""
        reg = RunRegistry()
        reg.register("r1", n_total=0)
        reg.update_progress("r1", 3, n_total=12)
        s = reg.get("r1")
        assert s.n_total == 12
        assert s.n_completed == 3

    def test_update_progress_keeps_known_total_when_omitted(self):
        reg = RunRegistry()
        reg.register("r1", n_total=10)
        reg.update_progress("r1", 4)
        assert reg.get("r1").n_total == 10

    def test_mark_completed_uses_the_learned_total(self):
        reg = RunRegistry()
        reg.register("r1", n_total=0)
        reg.update_progress("r1", 5, n_total=20)
        reg.mark_completed("r1")
        s = reg.get("r1")
        assert s.n_total == 20
        assert s.n_completed == 20


class TestProgressFraction:
    """The fraction the status endpoint reports, as a directly testable rule."""

    def test_reports_fraction_while_running(self):
        from src.api.services.eval_runs import progress_fraction

        reg = RunRegistry()
        reg.register("r1", n_total=0)
        reg.update_progress("r1", 3, n_total=12)
        assert progress_fraction(reg.get("r1")) == pytest.approx(0.25)

    def test_reports_zero_before_the_total_is_known(self):
        from src.api.services.eval_runs import progress_fraction

        reg = RunRegistry()
        reg.register("r1", n_total=0)
        assert progress_fraction(reg.get("r1")) == 0.0

    def test_reports_one_when_completed(self):
        from src.api.services.eval_runs import progress_fraction

        reg = RunRegistry()
        reg.register("r1", n_total=0)
        reg.update_progress("r1", 5, n_total=20)
        reg.mark_completed("r1")
        assert progress_fraction(reg.get("r1")) == 1.0

    def test_failed_run_keeps_its_partial_fraction(self):
        from src.api.services.eval_runs import progress_fraction

        reg = RunRegistry()
        reg.register("r1", n_total=0)
        reg.update_progress("r1", 2, n_total=8)
        reg.mark_failed("r1", "boom")
        assert progress_fraction(reg.get("r1")) == pytest.approx(0.25)
