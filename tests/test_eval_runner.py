"""Tests for src.eval.runner."""

from __future__ import annotations

import json

import pytest

from src.domain import SearchResult
from src.eval import storage
from src.eval.config import EvalConfig
from src.eval.runner import (
    EvalRunner,
    SpendCeilingExceeded,
    _score_question,
    assert_within_spend_ceiling,
)
from src.eval.schemas import EvalQuestion, EvalResult


class DummyLLM:
    """Returns canned answers / canned JSON for any prompt."""

    def __init__(self, answer: str = "<dummy>", judge_payload: dict | None = None):
        self.answer = answer
        self.model = "gpt-4.1-nano"  # engine reads .model for spans + cost pricing
        self.judge_payload = judge_payload or {
            "score": 1.0,
            "claims": [],
            "chunks": [],
            "factual_match": 1.0,
            "is_refusal": False,
            "reasoning": "ok",
        }
        self.calls: list[str] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(prompt)
        # Heuristic: judge prompts request JSON; answer prompts don't.
        if (
            "JSON" in (system_prompt or "")
            or '"score"' in prompt
            or '"claims"' in prompt
            or "JSON" in prompt
        ):
            return json.dumps(self.judge_payload)
        return self.answer

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int, int]:
        text = self.generate(prompt, system_prompt)
        return text, max(1, len(prompt.split())), len(text.split())


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate(
        {
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
                "bootstrap_n": 100,
                "permutation_n": 100,
                "seed": 42,
            },
        }
    )


@pytest.fixture
def squad_5(monkeypatch, tmp_path):
    """Override the SQuAD frozen path with a tiny 5-question synthetic set."""
    questions = [
        EvalQuestion(
            id=f"q{i}",
            question=f"What is fact {i}?",
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


class TestScoreQuestion:
    """Direct coverage of _score_question's judge-metric wiring.

    WHY this test exists: _score_question previously delegated the
    (score, reasoning[, details_json]) -> (score, details_dict) reshape to
    three wrapper functions in eval.metrics.generation. Those wrappers had
    their own tests but _score_question itself — the only caller — did not.
    This guards the rewire that inlines the reshape here.
    """

    def _question(self) -> EvalQuestion:
        return EvalQuestion(
            id="q1",
            question="What is fact 0?",
            gold_answer="Fact 0.",
            gold_chunk_ids=["c1"],
        )

    def _chunks(self) -> list[SearchResult]:
        return [
            SearchResult(
                content="Fact 0 is important.",
                metadata={"doc_id": "d1"},
                score=0.9,
                doc_id="d1",
                chunk_id="c1",
            )
        ]

    def test_populates_judge_metrics_and_details(self):
        llm = DummyLLM(
            answer="Fact 0.",
            judge_payload={
                "score": 0.8,
                "claims": [{"claim": "x", "supported": True, "evidence": "y"}],
                "chunks": [{"chunk_index": 0, "relevant": True}],
                "factual_match": 1.0,
                "is_refusal": False,
                "reasoning": "matches",
            },
        )

        metrics, details = _score_question(self._question(), self._chunks(), "Fact 0.", llm)

        for key in ("judge_faithfulness", "judge_context_precision", "judge_answer_relevancy"):
            assert metrics[key] == pytest.approx(0.8)

        # answer_correctness is the mean of embedding cosine (real, unmocked
        # embedder) and the judge's factual_match (1.0 here). generated ==
        # gold_answer, so cosine ≈ 1.0 and the mean should be too.
        assert metrics["answer_correctness"] == pytest.approx(1.0, abs=1e-3)

        assert details["judge_faithfulness"]["reasoning"] == "matches"
        assert "claims" in details["judge_faithfulness"]
        assert details["judge_context_precision"]["reasoning"] == "matches"
        assert "chunks" in details["judge_context_precision"]
        assert details["judge_answer_relevancy"]["reasoning"] == "matches"

    def test_skips_judge_metrics_without_gold_chunk_ids(self):
        question = EvalQuestion(id="q1", question="What is fact 0?")
        llm = DummyLLM()

        metrics, details = _score_question(question, self._chunks(), "Fact 0.", llm)

        assert "judge_faithfulness" not in metrics
        assert "judge_context_precision" not in metrics
        assert "judge_answer_relevancy" not in metrics
        assert "context_recall" not in metrics


class TestEvalRunner:
    def test_end_to_end_squad(self, tmp_eval_runs, squad_5):
        cfg = _baseline_config()
        runner = EvalRunner(
            cfg,
            llm_override=DummyLLM("Fact 0."),
            judge_llm_override=DummyLLM(
                judge_payload={
                    "score": 1.0,
                    "claims": [],
                    "chunks": [],
                    "factual_match": 1.0,
                    "is_refusal": False,
                    "reasoning": "ok",
                }
            ),
        )
        meta = runner.run()
        assert meta.n_questions == 5
        assert meta.n_errors == 0
        assert meta.config_name == "test"

        # Verify run dir contains all expected files
        run_dir = tmp_eval_runs / meta.run_id
        for f in ["metadata.json", "questions.jsonl", "metrics.json", "cost.json", "config.yaml"]:
            assert (run_dir / f).exists()

        # Reload via storage
        loaded = storage.load_run(meta.run_id)
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


class TestSpendCeiling:
    """The harness's only guard on real money — previously untested.

    The check was written inline inside the per-question loop, where nothing
    could reach it.
    """

    def _result(self, cost: float) -> EvalResult:
        return EvalResult(
            question_id=f"q{cost}",
            dataset="d",
            retrieved_chunk_ids=[],
            retrieved_chunks=[],
            generated_answer="a",
            metrics={},
            timings_ms={},
            tokens={},
            cost_usd=cost,
        )

    def test_no_ceiling_never_aborts(self):
        # The guard signals by raising, so "does not raise" is the assertion.
        assert_within_spend_ceiling([self._result(1000.0)], None)

    def test_under_the_ceiling_passes(self):
        assert_within_spend_ceiling([self._result(0.4), self._result(0.4)], 1.0)

    def test_exactly_at_the_ceiling_passes(self):
        """Strictly greater aborts, so spending the full budget is allowed."""
        assert_within_spend_ceiling([self._result(1.0)], 1.0)

    def test_over_the_ceiling_aborts(self):
        with pytest.raises(SpendCeilingExceeded):
            assert_within_spend_ceiling([self._result(0.6), self._result(0.6)], 1.0)

    def test_the_message_names_the_amount_and_how_far_it_got(self):
        with pytest.raises(SpendCeilingExceeded, match=r"\$1\.2000 > \$1\.0000"):
            assert_within_spend_ceiling([self._result(0.6), self._result(0.6)], 1.0)
        with pytest.raises(SpendCeilingExceeded, match="after 2 questions"):
            assert_within_spend_ceiling([self._result(0.6), self._result(0.6)], 1.0)

    def test_an_empty_run_never_aborts(self):
        assert_within_spend_ceiling([], 0.0)
