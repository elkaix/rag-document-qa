"""Tests for src.api.routes.eval."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_squad(monkeypatch, tmp_path):
    from src.eval.schemas import EvalQuestion
    questions = [
        EvalQuestion(
            id=f"q{i}", question=f"What is fact {i}?",
            gold_answer=f"Fact {i}.", gold_chunk_ids=[f"q{i}"],
            metadata={"context": f"Fact {i} is important.", "title": "t"},
        )
        for i in range(3)
    ]
    path = tmp_path / "squad.jsonl"
    with path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")
    monkeypatch.setattr("src.eval.datasets.squad_v2.DEFAULT_OUTPUT_PATH", path)
    return path


@pytest.fixture
def configs_dir(tmp_path, monkeypatch):
    """Create configs/eval/test.yaml in a temp dir; monkeypatch CONFIGS_DIR."""
    cfg_dir = tmp_path / "configs" / "eval"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "test.yaml"
    cfg_path.write_text("""
name: api-test
description: ""
pipeline:
  chunker: {strategy: "recursive", chunk_size: 256, chunk_overlap: 32}
  retriever: {top_k: 3}
  generator: {model: "gpt-4.1-nano", reasoning_model: null}
eval:
  datasets: ["squad_v2_dev_200"]
  judge_model: "gpt-4.1-nano"
  bootstrap_n: 50
  permutation_n: 50
  seed: 1
""")
    monkeypatch.setattr("src.api.routes.eval.CONFIGS_DIR", cfg_dir)
    return cfg_dir


@pytest.fixture
def tmp_eval_runs(tmp_path, monkeypatch):
    runs = tmp_path / "eval_runs"
    runs.mkdir()
    monkeypatch.setenv("EVAL_RUNS_DIR", str(runs))
    import importlib
    import src.eval.storage
    importlib.reload(src.eval.storage)
    yield runs
    monkeypatch.delenv("EVAL_RUNS_DIR", raising=False)
    importlib.reload(src.eval.storage)


@pytest.fixture
def client_with_dummy_llm(monkeypatch):
    """TestClient where the eval route uses a dummy LLM via env override."""
    monkeypatch.setenv("EVAL_LLM_OVERRIDE_DUMMY", "1")
    from src.api.main import app
    yield TestClient(app)


class TestConfigsEndpoint:
    def test_lists_configs(self, configs_dir, client_with_dummy_llm):
        r = client_with_dummy_llm.get("/api/eval/configs")
        assert r.status_code == 200
        assert "test" in r.json()


class TestRunSubmitAndStatus:
    def test_submit_and_complete(self, configs_dir, tmp_eval_runs,
                                 synthetic_squad, client_with_dummy_llm):
        r = client_with_dummy_llm.post(
            "/api/eval/run", json={"config_name": "test"}
        )
        assert r.status_code == 202, r.text
        body = r.json()
        run_id = body["run_id"]
        assert body["status"] == "queued"

        # Poll status until completed (or fail after 30s)
        for _ in range(60):
            sr = client_with_dummy_llm.get(f"/api/eval/runs/{run_id}/status")
            assert sr.status_code == 200
            if sr.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        assert sr.json()["status"] == "completed", sr.json()

    def test_unknown_config_returns_404(self, configs_dir, client_with_dummy_llm):
        r = client_with_dummy_llm.post(
            "/api/eval/run", json={"config_name": "nope"}
        )
        assert r.status_code == 404


class TestRunsList:
    def test_lists_completed_runs(self, configs_dir, tmp_eval_runs,
                                  synthetic_squad, client_with_dummy_llm):
        # Submit + wait
        r = client_with_dummy_llm.post(
            "/api/eval/run", json={"config_name": "test"}
        )
        run_id = r.json()["run_id"]
        for _ in range(60):
            sr = client_with_dummy_llm.get(f"/api/eval/runs/{run_id}/status")
            if sr.json()["status"] == "completed":
                break
            time.sleep(0.5)

        # List
        list_r = client_with_dummy_llm.get("/api/eval/runs")
        assert list_r.status_code == 200
        runs = list_r.json()
        assert any(r["run_id"] == run_id for r in runs)


class TestRunDetailAndResults:
    def test_get_run_detail(self, configs_dir, tmp_eval_runs,
                            synthetic_squad, client_with_dummy_llm):
        r = client_with_dummy_llm.post(
            "/api/eval/run", json={"config_name": "test"}
        )
        run_id = r.json()["run_id"]
        for _ in range(60):
            sr = client_with_dummy_llm.get(f"/api/eval/runs/{run_id}/status")
            if sr.json()["status"] == "completed":
                break
            time.sleep(0.5)

        dr = client_with_dummy_llm.get(f"/api/eval/runs/{run_id}")
        assert dr.status_code == 200
        d = dr.json()
        assert d["metadata"]["run_id"] == run_id
        assert d["n_results"] == 3

    def test_get_run_results_paginated(self, configs_dir, tmp_eval_runs,
                                       synthetic_squad, client_with_dummy_llm):
        r = client_with_dummy_llm.post(
            "/api/eval/run", json={"config_name": "test"}
        )
        run_id = r.json()["run_id"]
        for _ in range(60):
            sr = client_with_dummy_llm.get(f"/api/eval/runs/{run_id}/status")
            if sr.json()["status"] == "completed":
                break
            time.sleep(0.5)

        rr = client_with_dummy_llm.get(
            f"/api/eval/runs/{run_id}/results?page=1&page_size=2"
        )
        assert rr.status_code == 200
        body = rr.json()
        assert len(body["items"]) == 2
        assert body["total"] == 3

    def test_missing_run_returns_404(self, configs_dir, client_with_dummy_llm):
        r = client_with_dummy_llm.get("/api/eval/runs/nonexistent")
        assert r.status_code == 404
