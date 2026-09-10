"""Characterization + unit tests for retrieval composition.

The composition rule — which adapters wrap which, in what order, and what the
effective top-k becomes — used to exist twice: once in the production factory
(selected by a strategy string) and once in the eval pipeline (selected by four
boolean levers). These tests pin the rule so the two can share one owner.
"""

from __future__ import annotations

import pytest

from src.domain import SearchResult
from src.retrieval.base import Retriever
from src.retrieval.query_rewriter import MultiQueryRetriever
from src.retrieval.reranker import RerankingRetriever


class FakeRetriever:
    """A Retriever that returns a fixed list and records the top_k it was asked for."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeReranker:
    def rerank(self, query, candidates, final_top_k):
        return list(reversed(candidates))[:final_top_k]


class FakeRewriter:
    def rewrite(self, query: str) -> list[str]:
        return [query, f"{query} (rephrased)"]


def _result(chunk_id: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        content=f"content {chunk_id}",
        metadata={"chunk_index": 0},
        score=score,
        doc_id="doc",
        chunk_id=chunk_id,
    )


class TestCompositionOrder:
    """Reranking wraps rewriting wraps the base — the order eval established."""

    def test_bare_base_is_returned_unwrapped(self):
        from src.retrieval.composition import compose_retrieval

        base = FakeRetriever()
        plan = compose_retrieval(base=base, top_k=5)
        assert plan.retriever is base

    def test_rewriter_wraps_the_base(self):
        from src.retrieval.composition import compose_retrieval

        base = FakeRetriever()
        plan = compose_retrieval(base=base, rewriter=FakeRewriter(), top_k=5)
        assert isinstance(plan.retriever, MultiQueryRetriever)

    def test_reranker_wraps_the_rewriter(self):
        from src.retrieval.composition import compose_retrieval

        base = FakeRetriever()
        plan = compose_retrieval(
            base=base, rewriter=FakeRewriter(), reranker=FakeReranker(), top_k=5
        )
        outer = plan.retriever
        assert isinstance(outer, RerankingRetriever)
        assert isinstance(outer._inner, MultiQueryRetriever)

    def test_every_composition_still_conforms_to_the_seam(self):
        from src.retrieval.composition import compose_retrieval

        for kwargs in (
            {},
            {"rewriter": FakeRewriter()},
            {"reranker": FakeReranker()},
            {"rewriter": FakeRewriter(), "reranker": FakeReranker()},
        ):
            plan = compose_retrieval(base=FakeRetriever(), top_k=5, **kwargs)
            assert isinstance(plan.retriever, Retriever)


class TestEffectiveTopK:
    """The rule that used to exist only on the eval side."""

    def test_without_reranking_top_k_is_the_requested_one(self):
        from src.retrieval.composition import compose_retrieval

        assert compose_retrieval(base=FakeRetriever(), top_k=7).top_k == 7

    def test_with_reranking_the_final_top_k_wins(self):
        from src.retrieval.composition import compose_retrieval

        plan = compose_retrieval(
            base=FakeRetriever(), reranker=FakeReranker(), top_k=7, rerank_final_top_k=3
        )
        assert plan.top_k == 3

    def test_with_reranking_and_no_explicit_final_the_requested_top_k_stands(self):
        from src.retrieval.composition import compose_retrieval

        plan = compose_retrieval(base=FakeRetriever(), reranker=FakeReranker(), top_k=7)
        assert plan.top_k == 7

    def test_reranker_over_fetches_wider_than_the_final_count(self):
        from src.retrieval.composition import compose_retrieval

        base = FakeRetriever([_result(f"c{i}") for i in range(30)])
        plan = compose_retrieval(
            base=base,
            reranker=FakeReranker(),
            top_k=5,
            rerank_over_fetch_n=20,
            rerank_final_top_k=5,
        )
        plan.retriever.retrieve("q", plan.top_k)
        assert base.calls == [("q", 20)], "inner must be asked for the wider set"


class TestStrategyPresets:
    """Production strategy names are presets over the same composition rule."""

    def test_dense_is_the_bare_base(self, populated_vector_store):
        from src.retrieval.composition import build_retrieval_plan
        from src.retrieval.dense import DenseRetriever

        plan = build_retrieval_plan("dense", populated_vector_store)
        assert isinstance(plan.retriever, DenseRetriever)

    def test_unknown_strategy_is_rejected(self, populated_vector_store):
        from src.retrieval.composition import build_retrieval_plan

        with pytest.raises(ValueError, match="Unknown retriever strategy"):
            build_retrieval_plan("nonsense", populated_vector_store)

    def test_hybrid_is_wired_not_deferred(self, populated_vector_store):
        """ADR 0009 turned the two recognised-but-unbuildable names into presets."""
        from src.retrieval.composition import build_retrieval_plan
        from src.retrieval.hybrid import BM25HybridRetriever

        plan = build_retrieval_plan("hybrid", populated_vector_store)
        assert isinstance(plan.retriever, BM25HybridRetriever)
        assert plan.top_k == 5

    def test_multi_query_is_wired_not_deferred(self, populated_vector_store, monkeypatch):
        from src.retrieval.composition import build_retrieval_plan

        monkeypatch.setattr("src.retrieval.composition.QUERY_REWRITER_MODEL", "gpt-4.1-nano")

        class _LLM:
            def generate_with_usage(self, prompt, system_prompt=None):
                return "[]", 0, 0

        plan = build_retrieval_plan("multi_query", populated_vector_store, llm=_LLM())
        assert isinstance(plan.retriever, MultiQueryRetriever)
        assert plan.top_k == 5
