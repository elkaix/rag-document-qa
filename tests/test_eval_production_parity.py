"""Eval<->production parity — the regression guard for prompt/context drift.

The whole point of step 4c (issue #16) is that the eval harness measures the
*shipped* pipeline. Before convergence, the eval pipeline carried its own copy
of the answer prompt (worded differently) and joined context without filename
prefixes, so eval scored a pipeline that was not the one served. This test pins
the eval path to the single shipped prompt + context builders in
`src.query_engine.prompt` — the same ones the production RAGBackend uses. If
either drifts, this fails.
"""

from __future__ import annotations

from src.eval.config import EvalConfig
from src.eval.pipeline_factory import build_pipeline
from src.eval.schemas import EvalQuestion
from src.query_engine.prompt import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_context,
)


class _RecordingLLM:
    """Captures exactly the (system, user) instructions the answer pass receives."""

    model = "gpt-4.1-nano"

    def __init__(self) -> None:
        self.system: str | None = None
        self.user: str | None = None

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "{}"  # judge path — unused for answer capture

    def generate_with_usage(self, prompt, system_prompt=None):
        self.system = system_prompt
        self.user = prompt
        return "captured", 5, 2


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "parity", "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"], "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 100, "permutation_n": 100, "seed": 7,
        },
    })


def test_eval_pipeline_issues_the_shipped_prompt_and_context():
    """Eval sends the production ANSWER_SYSTEM_PROMPT and filename-prefixed context."""
    recorder = _RecordingLLM()
    pipeline = build_pipeline(
        _baseline_config(), "squad_v2_dev_200",
        llm_override=recorder, judge_llm_override=_RecordingLLM(),
    )
    try:
        pipeline.ingest([
            EvalQuestion(
                id="q1", question="What is the capital of France?",
                gold_answer="Paris", gold_chunk_ids=["q1"],
                metadata={"context": "Paris is the capital of France.", "title": "t"},
            )
        ])
        results, _answer, _telemetry = pipeline.query("What is the capital of France?")

        # Eval uses the ONE shipped answer prompt — not a reworded eval copy.
        assert recorder.system == ANSWER_SYSTEM_PROMPT
        # ...and the ONE shipped context + user builders, byte-for-byte. A bare
        # join or a divergent template would make this inequality fail.
        expected_user = build_answer_user_prompt(
            build_context(results), "What is the capital of France?"
        )
        assert recorder.user == expected_user
    finally:
        pipeline.teardown()
