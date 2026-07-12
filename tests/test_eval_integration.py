"""End-to-end integration test for Sub-plan 1B: full eval lifecycle.

Runs two evals back-to-back with different top_k values, then compares
them. Uses DummyLLM + 5-question synthetic SQuAD slice so no network
calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import EvalConfig, EvalRunner, compare_runs, list_runs, load_run
from src.eval.schemas import EvalQuestion


class DummyLLM:
    """Returns canned data — JSON for judges, plain for generation."""
    model = "gpt-4.1-nano"  # engine reads .model for spans + cost pricing

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if "JSON" in (system_prompt or "") or '"score"' in prompt or '"is_refusal"' in prompt:
            return json.dumps({
                "score": 1.0, "claims": [], "chunks": [],
                "factual_match": 1.0, "is_refusal": False, "reasoning": "ok",
            })
        return "<dummy answer>"

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int, int]:
        text = self.generate(prompt, system_prompt)
        return text, max(1, len(prompt.split())), len(text.split())


def _make_config(name: str, top_k: int) -> EvalConfig:
    return EvalConfig.model_validate({
        "name": name, "description": f"top_k={top_k}",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": top_k},
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
def synthetic_squad(monkeypatch, tmp_path):
    questions = [
        EvalQuestion(
            id=f"q{i}", question=f"What is fact {i}?",
            gold_answer=f"Fact {i}.", gold_chunk_ids=[f"q{i}"],
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


class TestFullLifecycle:
    def test_two_runs_then_compare(self, tmp_eval_runs, synthetic_squad):
        # Run 1: top_k=3
        cfg_a = _make_config("topk-3", top_k=3)
        runner_a = EvalRunner(
            cfg_a, llm_override=DummyLLM(), judge_llm_override=DummyLLM(),
        )
        meta_a = runner_a.run()
        assert meta_a.n_questions == 5
        assert meta_a.n_errors == 0

        # Run 2: top_k=1
        cfg_b = _make_config("topk-1", top_k=1)
        runner_b = EvalRunner(
            cfg_b, llm_override=DummyLLM(), judge_llm_override=DummyLLM(),
        )
        meta_b = runner_b.run()
        assert meta_b.n_questions == 5

        # list_runs sees both
        runs = list_runs()
        assert len(runs) == 2

        # load_run round-trips both
        loaded_a = load_run(meta_a.run_id)
        loaded_b = load_run(meta_b.run_id)
        assert loaded_a["metadata"].config_name == "topk-3"
        assert loaded_b["metadata"].config_name == "topk-1"

        # compare_runs produces a CompareResult (eval_set_versions match
        # since both runs used the same synthetic SQuAD)
        cmp = compare_runs(meta_a.run_id, meta_b.run_id)
        assert cmp.run_a.run_id == meta_a.run_id
        assert cmp.run_b.run_id == meta_b.run_id
