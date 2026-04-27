"""Tests for the Phase 2 cost ledger covering generator + judge + rewriter spend."""

from __future__ import annotations

from src.eval.schemas import EvalResult


def test_eval_result_has_cost_breakdown_field():
    """EvalResult must carry a cost_breakdown dict with per-bucket spend."""
    r = EvalResult(
        question_id="q1",
        dataset="squad_v2_dev_200",
        retrieved_chunk_ids=[],
        retrieved_chunks=[],
        generated_answer="",
        metrics={},
        timings_ms={},
        tokens={},
        cost_usd=0.0,
        cost_breakdown={"generator": 0.0, "judge": 0.0, "rewriter": 0.0},
    )
    assert r.cost_breakdown["generator"] == 0.0
    assert r.cost_breakdown["judge"] == 0.0
    assert r.cost_breakdown["rewriter"] == 0.0


def test_eval_result_cost_breakdown_defaults():
    """cost_breakdown must default to a generator-only dict when omitted."""
    r = EvalResult(
        question_id="q1",
        dataset="squad_v2_dev_200",
        retrieved_chunk_ids=[],
        retrieved_chunks=[],
        generated_answer="",
        metrics={},
        timings_ms={},
        tokens={},
        cost_usd=0.05,
    )
    # back-compat default: existing records read as generator-only
    assert r.cost_breakdown == {"generator": 0.05, "judge": 0.0, "rewriter": 0.0}


def test_aggregator_sums_cost_breakdown_into_totals():
    """aggregate_costs must surface per-bucket totals alongside total_usd."""
    from src.eval.metrics.operational import aggregate_costs

    results = [
        EvalResult(
            question_id="q1", dataset="d", retrieved_chunk_ids=[], retrieved_chunks=[],
            generated_answer="", metrics={}, timings_ms={}, tokens={},
            cost_usd=0.10,
            cost_breakdown={"generator": 0.04, "judge": 0.05, "rewriter": 0.01},
        ),
        EvalResult(
            question_id="q2", dataset="d", retrieved_chunk_ids=[], retrieved_chunks=[],
            generated_answer="", metrics={}, timings_ms={}, tokens={},
            cost_usd=0.20,
            cost_breakdown={"generator": 0.08, "judge": 0.10, "rewriter": 0.02},
        ),
    ]
    summary = aggregate_costs(results)
    assert round(summary["total_usd"], 4) == 0.30
    assert round(summary["generator_total_usd"], 4) == 0.12
    assert round(summary["judge_total_usd"], 4) == 0.15
    assert round(summary["rewriter_total_usd"], 4) == 0.03


def test_llm_handler_generate_with_usage_returns_tokens():
    """LLMHandler.generate_with_usage must return (text, prompt_tokens, completion_tokens)."""
    from src.llm_handler import LLMHandler

    # Use the dummy fallback path — no API key needed.
    handler = LLMHandler("__dummy__")
    text, prompt_tokens, completion_tokens = handler.generate_with_usage(
        prompt="What is RAG?", system_prompt="Be brief."
    )
    assert isinstance(text, str)
    assert isinstance(prompt_tokens, int) and prompt_tokens > 0
    assert isinstance(completion_tokens, int) and completion_tokens >= 0
