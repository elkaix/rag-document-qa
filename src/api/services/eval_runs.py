"""
In-process registry of in-flight eval runs.

API Layer Position:
  EvalRunner.run() (async dispatch)  ──┐
                                       │
  POST /api/eval/run         ──────────┼──→ [RunRegistry] ←─ GET /api/eval/runs/{id}/status
                                       │
  on_progress callback       ──────────┘

Design decisions:
  - In-memory + threading.Lock — FastAPI's threadpool for sync route
    handlers means concurrent reads/writes are real. Locked dict is
    simple and sufficient for portfolio-scale traffic.
  - Auto-evict completed runs after TTL so the registry doesn't grow
    unbounded; runs are persisted to disk separately by storage.save_run
    so eviction loses no information.
  - Active runs (no completed_at) are NEVER evicted regardless of age —
    a stuck run should be visible, not silently disappear.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


@dataclass
class RunStatus:
    """Snapshot of a single eval run's lifecycle state.

    Concept taught:
        Mutable dataclass as a simple value object — no ORM, no Pydantic,
        just a plain struct the registry owns and callers can read.

    Pipeline position:
        Lives inside RunRegistry._runs dict, keyed by run_id. Created at
        register(), mutated by update_progress / mark_completed / mark_failed.
    """

    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    n_completed: int
    n_total: int
    error_message: str | None = None
    # WHY: completed_at drives TTL-based eviction. Active runs keep this None
    #      so evict_old() can distinguish "still running" from "done long ago".
    completed_at: datetime | None = None


class RunRegistry:
    """Thread-safe in-process registry of eval run states.

    Concept taught:
        The simplest possible shared-state solution for an async web server
        that uses a threadpool for sync handlers. A single threading.Lock
        serialises every read and write — no race conditions, no external
        dependency, trivially correct at portfolio scale.

    Why threading.Lock over asyncio.Lock:
        FastAPI runs sync route handlers in a threadpool (not the event loop),
        so asyncio primitives don't protect against thread-level data races.
        threading.Lock is the right tool here.

    Pipeline position:
        Instantiated once at app startup (singleton on app.state). Shared by
        POST /api/eval/run (write) and GET /api/eval/runs/{id}/status (read).
    """

    def __init__(self) -> None:
        # PATTERN: Single lock guards the entire dict — simple and sufficient.
        self._lock = threading.Lock()
        self._runs: dict[str, RunStatus] = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, run_id: str, n_total: int) -> None:
        """Create a new run entry in the queued state.

        Args:
            run_id: Unique identifier for the run (caller's responsibility).
            n_total: Total number of evaluation items to process.
        """
        with self._lock:
            self._runs[run_id] = RunStatus(
                run_id=run_id,
                status="queued",
                n_completed=0,
                n_total=n_total,
            )

    def update_progress(self, run_id: str, n_completed: int) -> None:
        """Record incremental progress; transitions queued→running on first call.

        Only valid when the run is in queued or running state. Silently ignores
        unknown run IDs to keep progress callbacks fire-and-forget safe.

        Args:
            run_id: The run to update.
            n_completed: Number of items completed so far.
        """
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None or entry.status not in ("queued", "running"):
                return
            # WHY: First progress call transitions queued→running so callers
            #      can distinguish "not started" from "in progress".
            entry.status = "running"
            entry.n_completed = n_completed

    def mark_completed(self, run_id: str) -> None:
        """Finalise a run as successfully completed.

        Sets n_completed = n_total and stamps completed_at with current UTC
        time, enabling TTL-based eviction.

        Args:
            run_id: The run to finalise.
        """
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return
            entry.status = "completed"
            entry.n_completed = entry.n_total
            entry.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, run_id: str, error: str) -> None:
        """Record a run as failed with an error message.

        Also stamps completed_at so the TTL eviction logic treats failed runs
        the same as completed ones — they don't need to live forever either.

        Args:
            run_id: The run that failed.
            error: Human-readable error description.
        """
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return
            entry.status = "failed"
            entry.error_message = error
            entry.completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunStatus | None:
        """Return the RunStatus for a run, or None if not found.

        Args:
            run_id: The run to look up.

        Returns:
            The RunStatus object (mutable — callers may read fields directly)
            or None if the run_id is unknown or was evicted.
        """
        with self._lock:
            return self._runs.get(run_id)

    def list_active(self) -> list[RunStatus]:
        """Return all runs currently in queued or running state.

        Returns:
            List of RunStatus objects for active runs; order is not guaranteed.
        """
        with self._lock:
            # WHY: snapshot under lock so the list is consistent even if
            #      another thread marks a run completed concurrently.
            return [
                s for s in self._runs.values()
                if s.status in ("queued", "running")
            ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def evict_old(self, ttl_seconds: float = 3600.0) -> int:
        """Remove completed/failed runs whose completed_at is older than ttl.

        Active runs (completed_at is None) are NEVER evicted, regardless of
        how long they have been running.

        Args:
            ttl_seconds: Maximum age in seconds for a completed/failed entry.

        Returns:
            Number of entries removed from the registry.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        to_evict: list[str] = []

        with self._lock:
            for run_id, entry in self._runs.items():
                # TRADE-OFF: We only evict entries that have a completed_at
                # timestamp. A stuck/hung active run will never be evicted —
                # it stays visible so operators can notice and investigate.
                if entry.completed_at is not None and entry.completed_at < cutoff:
                    to_evict.append(run_id)
            for run_id in to_evict:
                del self._runs[run_id]

        return len(to_evict)
