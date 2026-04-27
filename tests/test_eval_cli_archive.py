"""Tests for `python -m src.eval.cli archive` — copies small artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.cli import _cmd_archive


class _FakeArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_archive_copies_four_artifacts(tmp_path):
    """Archive copies metrics, cost, metadata, config but NOT questions.jsonl."""
    # Build a fake run dir with the expected artifacts plus a large file.
    src = tmp_path / "eval_runs" / "fake_run"
    src.mkdir(parents=True)
    (src / "metrics.json").write_text("[]")
    (src / "cost.json").write_text("{}")
    (src / "metadata.json").write_text('{"run_id": "fake_run"}')
    (src / "config.yaml").write_text("name: fake")
    (src / "questions.jsonl").write_text("\n".join(["{}"] * 200))  # large — not copied

    dst = tmp_path / "docs" / "phase2" / "runs" / "fake_run"
    rc = _cmd_archive(_FakeArgs(
        run_id="fake_run",
        to=str(dst),
        runs_root=str(tmp_path / "eval_runs"),
    ))
    assert rc == 0
    assert (dst / "metrics.json").exists()
    assert (dst / "cost.json").exists()
    assert (dst / "metadata.json").exists()
    assert (dst / "config.yaml").exists()
    # questions.jsonl is NOT copied (large).
    assert not (dst / "questions.jsonl").exists()
    # Its SHA-256 is recorded in metadata.json.
    md = json.loads((dst / "metadata.json").read_text())
    assert "questions_jsonl_sha256" in md
