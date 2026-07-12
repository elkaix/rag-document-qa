"""Tests for src.eval.pipeline_factory."""

from __future__ import annotations

import json

import pytest

from src.eval.config import EvalConfig
from src.eval.pipeline_factory import EvalPipeline, build_pipeline
from src.eval.schemas import EvalQuestion


class DummyLLM:
    """Returns a fixed answer; tracks calls."""
    def __init__(self, answer: str = "<dummy>"):
        self.answer = answer
        self.model = "gpt-4.1-nano"  # engine reads .model for spans + cost pricing
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self.answer

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int, int]:
        self.calls.append((prompt, system_prompt))
        return self.answer, max(1, len(prompt.split())), len(self.answer.split())


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "test",
        "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"],
            "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 100, "permutation_n": 100, "seed": 7,
        },
    })


def _squad_question(qid: str, ctx: str, q: str = "Q?") -> EvalQuestion:
    return EvalQuestion(
        id=qid, question=q, gold_answer="A", gold_chunk_ids=[qid],
        metadata={"context": ctx, "title": "t"},
    )


class TestBuildPipeline:
    def test_returns_pipeline_with_components(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM("answer"),
                           judge_llm_override=DummyLLM("{}"))
        assert isinstance(p, EvalPipeline)
        assert p.config is cfg
        assert p.dataset_name == "squad_v2_dev_200"
        p.teardown()


class TestIngestAndQuery:
    def test_squad_ingest_then_query(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM("Paris"),
                           judge_llm_override=DummyLLM("{}"))
        try:
            qs = [
                _squad_question("q1", "Paris is the capital of France.", "What is the capital of France?"),
                _squad_question("q2", "The Eiffel Tower is in Paris.", "Where is the Eiffel Tower?"),
            ]
            p.ingest(qs)

            chunks, answer, telemetry = p.query("What is the capital of France?")
            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            assert answer == "Paris"
            assert "timings_ms" in telemetry
            assert "tokens" in telemetry
            assert "cost_usd" in telemetry
            assert telemetry["timings_ms"]["retrieve"] >= 0.0
            assert telemetry["timings_ms"]["generate"] >= 0.0
            assert telemetry["tokens"]["prompt"] > 0
            assert telemetry["tokens"]["completion"] >= 0
            assert telemetry["cost_usd"] >= 0.0
        finally:
            p.teardown()


class TestTeardown:
    def test_teardown_does_not_raise(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM(),
                           judge_llm_override=DummyLLM())
        p.teardown()  # should not raise
