"""Tests for src.api.services.eval_runs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src.api.services.eval_runs import RunRegistry, RunStatus


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
        s.completed_at = datetime.now(timezone.utc) - timedelta(seconds=7200)

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
