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
    return EvalConfig.model_validate(
        {
            "name": "parity",
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
                "seed": 7,
            },
        }
    )


def test_eval_pipeline_issues_the_shipped_prompt_and_context():
    """Eval sends the production ANSWER_SYSTEM_PROMPT and filename-prefixed context."""
    recorder = _RecordingLLM()
    pipeline = build_pipeline(
        _baseline_config(),
        "squad_v2_dev_200",
        llm_override=recorder,
        judge_llm_override=_RecordingLLM(),
    )
    try:
        pipeline.ingest(
            [
                EvalQuestion(
                    id="q1",
                    question="What is the capital of France?",
                    gold_answer="Paris",
                    gold_chunk_ids=["q1"],
                    metadata={"context": "Paris is the capital of France.", "title": "t"},
                )
            ]
        )
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


class TestCompositionParity:
    """Eval and production must compose retrieval by the same rule.

    ADR 0004 single-sourced the prompt and the context builder. It left the
    *composition* rule in two places: production selected by a strategy string
    and passed top_k flat, while eval selected by lever flags and derived top_k
    from whether reranking was on. The two agreed only because final_top_k and
    TOP_K_RESULTS happened to be the same number — an agreement by coincidence
    that nothing tested and that would break the moment either was tuned.
    """

    def test_reranked_production_and_eval_agree_on_the_effective_top_k(self):
        """The case the original parity test could not reach: reranking on."""
        from src.retrieval.composition import compose_retrieval

        class _Base:
            def retrieve(self, query, top_k):
                return []

        class _Reranker:
            def rerank(self, query, candidates, final_top_k):
                return candidates[:final_top_k]

        production = compose_retrieval(
            base=_Base(), reranker=_Reranker(), top_k=5, rerank_over_fetch_n=20
        )
        evaluation = compose_retrieval(
            base=_Base(),
            reranker=_Reranker(),
            top_k=5,
            rerank_over_fetch_n=20,
            rerank_final_top_k=5,
        )
        assert production.top_k == evaluation.top_k

    def test_tuning_the_shared_constant_moves_both_sides_together(self):
        """The literals are single-sourced, so they cannot drift apart."""
        from src.config import (
            REFUSAL_NO_ANSWER_TEXT,
            REFUSAL_SIMILARITY_THRESHOLD,
            RERANK_OVER_FETCH_N,
            TOP_K_RESULTS,
        )
        from src.eval.config import RefusalHandlerCfg, RerankerCfg

        reranker = RerankerCfg()
        assert reranker.rerank_top_n == RERANK_OVER_FETCH_N
        assert reranker.final_top_k == TOP_K_RESULTS

        refusal = RefusalHandlerCfg()
        assert refusal.similarity_threshold == REFUSAL_SIMILARITY_THRESHOLD
        assert refusal.no_answer_text == REFUSAL_NO_ANSWER_TEXT

    def test_both_paths_use_the_one_composition_function(self):
        """A structural guard: neither caller may stack adapters itself again."""
        import inspect

        from src import backend
        from src.eval import pipeline_factory

        for module in (backend, pipeline_factory):
            source = inspect.getsource(module)
            assert "RerankingRetriever(" not in source, module.__name__
            assert "MultiQueryRetriever(" not in source, module.__name__
