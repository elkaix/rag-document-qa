"""Tests for src.eval.runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.config import EvalConfig
from src.eval.runner import EvalRunner
from src.eval.schemas import EvalQuestion


class DummyLLM:
    """Returns canned answers / canned JSON for any prompt."""
    def __init__(self, answer: str = "<dummy>", judge_payload: dict | None = None):
        self.answer = answer
        self.judge_payload = judge_payload or {
            "score": 1.0, "claims": [], "chunks": [], "factual_match": 1.0,
            "is_refusal": False, "reasoning": "ok",
        }
        self.calls: list[str] = []
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(prompt)
        # Heuristic: judge prompts request JSON; answer prompts don't.
        if "JSON" in (system_prompt or "") or '"score"' in prompt or '"claims"' in prompt or 'JSON' in prompt:
            return json.dumps(self.judge_payload)
        return self.answer


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "test", "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"],
            "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 100, "permutation_n": 100, "seed": 42,
        },
    })


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


@pytest.fixture
def squad_5(monkeypatch, tmp_path):
    """Override the SQuAD frozen path with a tiny 5-question synthetic set."""
    questions = [
        EvalQuestion(
            id=f"q{i}", question=f"What is fact {i}?",
            gold_answer=f"Fact {i}.",
            gold_chunk_ids=[f"q{i}"],
            metadata={"context": f"Fact {i} is important.", "title": "t"},
        )
        for i in range(5)
    ]
    path = tmp_path / "squad.jsonl"
    with path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")
    monkeypatch.setattr("src.eval.datasets.squad_v2.DEFAULT_OUTPUT_PATH", path)
    return path


class TestEvalRunner:
    def test_end_to_end_squad(self, tmp_eval_runs, squad_5):
        cfg = _baseline_config()
        runner = EvalRunner(
            cfg,
            llm_override=DummyLLM("Fact 0."),
            judge_llm_override=DummyLLM(judge_payload={
                "score": 1.0, "claims": [], "chunks": [], "factual_match": 1.0,
                "is_refusal": False, "reasoning": "ok",
            }),
        )
        meta = runner.run()
        assert meta.n_questions == 5
        assert meta.n_errors == 0
        assert meta.config_name == "test"

        # Verify run dir contains all expected files
        run_dir = tmp_eval_runs.EVAL_RUNS_DIR / meta.run_id
        for f in ["metadata.json", "questions.jsonl", "metrics.json",
                  "cost.json", "config.yaml"]:
            assert (run_dir / f).exists()

        # Reload via storage
        loaded = tmp_eval_runs.load_run(meta.run_id)
        assert len(loaded["results"]) == 5
        assert loaded["aggregated"], "aggregated metrics should be non-empty"

    def test_progress_callback(self, tmp_eval_runs, squad_5):
        cfg = _baseline_config()
        progress_calls = []
        runner = EvalRunner(
            cfg,
            llm_override=DummyLLM("answer"),
            judge_llm_override=DummyLLM(),
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )
        runner.run()
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)
