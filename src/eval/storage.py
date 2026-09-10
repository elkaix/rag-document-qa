"""
Storage layer — read/write eval run directories.

Eval Harness Position:
  EvalRunner → save_run() → eval_runs/<run_id>/{metadata,questions,metrics,cost,config}
  CLI/API    → list_runs() / load_run() → render

Design decisions:
  - One directory per run, with five well-known files. Plain JSON / JSONL
    so any tool (jq, pandas, the eye) can inspect a run.
  - The runs directory is resolved per call (runs_dir()), and every read/write
    accepts it as an argument, so a caller can point at a temp directory without
    mutating module state.
  - delete_run refuses path traversal — destructive operations get a
    safety check at the boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.schemas import AggregatedMetric, EvalResult, RunMetadata

DEFAULT_RUNS_DIRNAME = "eval_runs"


def runs_dir() -> Path:
    """Resolve the eval runs directory.

    Returns:
        ``$EVAL_RUNS_DIR`` when set, otherwise ``eval_runs`` in the working
        directory.

    BEFORE: this was a module-level constant evaluated at import time, so the
        CLI configured it by *reassigning another module's global*
        (``_storage.EVAL_RUNS_DIR = ...``) and tests had to re-import the module
        after setting the variable. Five separate WHY-comments across three
        files existed to explain that workaround.
    AFTER:  resolution happens per call, and every function takes the directory
        as an argument, so callers inject rather than mutate.
    """
    return Path(os.getenv("EVAL_RUNS_DIR", DEFAULT_RUNS_DIRNAME))


def current_git_sha() -> str:
    """Return the HEAD commit SHA, or ``"unknown"`` outside a git checkout.

    Returns:
        The full SHA, or ``"unknown"`` when git is unavailable — the harness may
        run in a CI container or a zip-extracted deployment, and provenance
        being unknown is not a reason to fail a run.

    WHY here: run-id derivation needs it, and this used to be a
        ``subprocess.check_output`` with a bare ``except`` copied into both
        ``EvalRunner.run`` and the HTTP submit route, which then had to agree on
        the result to land in the same directory.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compute_run_id(config_name: str, started_at: datetime, git_sha: str) -> str:
    """Build a human-readable, sortable run ID.

    Format: YYYY-MM-DD_HHMMSS_<config-name>_<sha7>

    Args:
        config_name: Name of the eval config (e.g. "baseline").
        started_at: UTC datetime the run began.
        git_sha: Full or partial git SHA of the current commit.

    Returns:
        Deterministic string suitable for use as a directory name.

    Teaches:
        Sortable directory names — lexicographic order == chronological
        order because the timestamp is the leading component. This makes
        `ls -1 eval_runs/` an implicit run history without any index file.
    """
    ts = started_at.strftime("%Y-%m-%d_%H%M%S")
    sha7 = git_sha[:7]
    return f"{ts}_{config_name}_{sha7}"


def save_run(
    run_dir: Path,
    metadata: RunMetadata,
    results: list[EvalResult],
    aggregated: list[AggregatedMetric],
    cost: dict[str, Any],
    config_yaml_text: str,
) -> None:
    """Persist all artifacts for one eval run to disk.

    Writes five files into run_dir (creating it and parents if needed):
      - metadata.json   — RunMetadata as pretty JSON
      - questions.jsonl — one EvalResult per line (JSON Lines)
      - metrics.json    — list of AggregatedMetric as pretty JSON
      - cost.json       — cost summary dict as pretty JSON
      - config.yaml     — raw YAML text of the config that drove this run

    Args:
        run_dir: Destination directory (created if absent).
        metadata: Provenance record for the run.
        results: Per-question evaluation outputs.
        aggregated: Metric aggregates across the run.
        cost: Cost summary (total_usd, mean_usd_per_query, etc.).
        config_yaml_text: Raw YAML string of the EvalConfig used.

    Teaches:
        JSON Lines (JSONL) for streaming — questions.jsonl can be read
        one line at a time for arbitrarily large eval runs, unlike a
        monolithic JSON array that must be fully parsed before any record
        is accessible.
    """
    # PATTERN: mkdir(parents=True, exist_ok=True) is the idiomatic way to
    #          ensure a directory exists without racing on creation.
    run_dir.mkdir(parents=True, exist_ok=True)

    # metadata.json — Pydantic's model_dump_json handles datetime serialization.
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))

    # questions.jsonl — one record per line; empty file for zero results.
    with (run_dir / "questions.jsonl").open("w") as fh:
        for result in results:
            fh.write(result.model_dump_json() + "\n")

    # metrics.json — list of AggregatedMetric dicts; default=str handles any
    # non-JSON-native types (e.g. numpy floats) gracefully.
    metrics_data = [am.model_dump() for am in aggregated]
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_data, indent=2, default=str)
    )

    # cost.json — plain dict; default=str for safety.
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2, default=str))

    # config.yaml — raw text, no parsing needed at write time.
    (run_dir / "config.yaml").write_text(config_yaml_text)


def load_run(run_id: str, base_dir: Path | None = None) -> dict:
    """Load all artifacts for a run from disk.

    Args:
        run_id: Directory name under the runs directory.
        base_dir: Runs directory to read from. Defaults to ``runs_dir()``.

    Returns:
        Dict with keys:
          - "metadata"   → RunMetadata
          - "results"    → list[EvalResult]
          - "aggregated" → list[AggregatedMetric]
          - "cost"       → dict

    Raises:
        FileNotFoundError: If the run directory does not exist.

    Teaches:
        model_validate_json vs model_validate — use model_validate_json
        when reading raw JSON strings (avoids an intermediate parse step),
        model_validate when you already have a Python dict/list.
    """
    run_dir = (base_dir or runs_dir()) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run {run_id} not found at {run_dir}")

    metadata = RunMetadata.model_validate_json(
        (run_dir / "metadata.json").read_text()
    )

    # JSONL: skip blank lines to handle trailing newlines robustly.
    results = [
        EvalResult.model_validate_json(line)
        for line in (run_dir / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    aggregated = [
        AggregatedMetric.model_validate(d)
        for d in json.loads((run_dir / "metrics.json").read_text())
    ]

    cost = json.loads((run_dir / "cost.json").read_text())

    return {
        "metadata": metadata,
        "results": results,
        "aggregated": aggregated,
        "cost": cost,
    }


def list_runs(base_dir: Path | None = None) -> list[RunMetadata]:
    """Enumerate all valid eval runs in the runs directory.

    A valid run is a subdirectory containing metadata.json. Directories
    without metadata.json (e.g. incomplete or interrupted runs) are silently
    skipped.

    Args:
        base_dir: Runs directory to scan. Defaults to ``runs_dir()``.

    Returns:
        RunMetadata instances sorted by started_at descending (newest first).

    Teaches:
        Convention over configuration — no index file is needed because
        the filesystem *is* the index. Any directory with metadata.json
        is a valid run; the rest are ignored.
    """
    base = base_dir or runs_dir()
    if not base.exists():
        return []

    runs: list[RunMetadata] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        metadata_file = entry / "metadata.json"
        if not metadata_file.exists():
            # TRADE-OFF: silently skip incomplete runs rather than raising.
            # A corrupted run shouldn't block listing all other runs.
            continue
        runs.append(RunMetadata.model_validate_json(metadata_file.read_text()))

    # Descending by started_at so the most recent run appears first.
    runs.sort(key=lambda r: r.started_at, reverse=True)
    return runs


def delete_run(run_id: str, base_dir: Path | None = None) -> None:
    """Permanently delete a run directory.

    Args:
        run_id: Directory name under the runs directory.
        base_dir: Runs directory to delete from. Defaults to ``runs_dir()``.

    Raises:
        ValueError: If run_id contains path traversal characters ('..' or '/').
        FileNotFoundError: If the run directory does not exist (from shutil.rmtree).

    Teaches:
        SECURITY: validate before act — check for traversal *before* any
        filesystem call. An attacker supplying run_id="../../../etc" must
        be rejected at the boundary, not after the path is constructed.
    """
    # SECURITY: reject path traversal before touching the filesystem.
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        raise ValueError(f"Invalid run_id: {run_id}")

    run_dir = (base_dir or runs_dir()) / run_id
    shutil.rmtree(run_dir)
