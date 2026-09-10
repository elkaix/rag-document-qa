"""Retrieval composition — turn a set of levers into a Retriever and its top-k.

RAG Pipeline Position:
    levers -> [COMPOSITION] -> Retriever + effective top_k -> QueryEngine
                   ^^^
    This is the single place that knows how retrieval adapters stack.

What concept it teaches:
    Composition as data. The adapters already conform to one seam (ADR 0004);
    what was missing was one owner for the *rule* that says which wraps which
    and what top-k the result should be queried with.

Why this approach over alternatives:
    ADR 0004 promoted the eval-proven levers into the core so production could
    activate them by configuration. It left two composition rules behind:

      - ``build_retriever`` selected by a **strategy string** and could express
        only ``dense`` and ``reranked``;
      - ``EvalPipeline._get_engine`` selected by **four boolean levers**, could
        express hybrid and multi-query, and derived the final top-k from
        whether reranking was on.

    Production had no equivalent of that last rule. The two agreed only because
    ``final_top_k`` and ``TOP_K_RESULTS`` happened to be the same number — an
    agreement by coincidence rather than by construction, which nothing tested
    and which would have broken the moment either was tuned. This module is the
    one owner; the strategy names become presets over it.

Design Decision:
    ``compose_retrieval`` takes an already-built base Retriever rather than a
    vector store, so it composes without touching storage, an embedder, or a
    cross-encoder download. That is what makes the rule unit-testable — the
    reason it had no test before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.config import RERANK_OVER_FETCH_N, TOP_K_RESULTS
from src.retrieval.base import Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.query_rewriter import MultiQueryRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from src.vector_store import ChromaVectorStore

_DEFERRED = {
    "hybrid": "needs a live BM25 corpus synced with ingestion (a new feature)",
    "multi_query": "lands with its rewriter-cost surfacing",
}


class _Reranker(Protocol):
    """What ``RerankingRetriever`` needs of a reranker."""

    def rerank(self, query: str, candidates: list, final_top_k: int) -> list: ...


class _Rewriter(Protocol):
    """What ``MultiQueryRetriever`` needs of a rewriter."""

    def rewrite(self, query: str) -> list[str]: ...


@dataclass(frozen=True)
class RetrievalPlan:
    """A composed Retriever together with the top-k it should be queried with.

    Attributes:
        retriever: The outermost adapter; everything else is inside it.
        top_k: The number of chunks the engine should ask for. This is part of
            the plan rather than a separate constant because reranking changes
            it — the caller cannot compute it without knowing the composition.
    """

    retriever: Retriever
    top_k: int


def compose_retrieval(
    *,
    base: Retriever,
    rewriter: _Rewriter | None = None,
    reranker: _Reranker | None = None,
    top_k: int = TOP_K_RESULTS,
    rerank_over_fetch_n: int = RERANK_OVER_FETCH_N,
    rerank_final_top_k: int | None = None,
) -> RetrievalPlan:
    """Stack the enabled levers over a base Retriever.

    Order is fixed: rewriting wraps the base, reranking wraps that. Rewriting
    widens the candidate pool, so reranking must see the union rather than one
    query's results.

    Args:
        base: The retriever every other lever composes over — dense, or a
            sparse/hybrid retriever once one is available.
        rewriter: Enables multi-query fan-out when supplied.
        reranker: Enables cross-encoder reranking when supplied.
        top_k: Chunks the engine should end up with when reranking is off.
        rerank_over_fetch_n: How many candidates the reranker sees before it
            narrows them. Only meaningful when ``reranker`` is supplied.
        rerank_final_top_k: Chunks to keep after reranking. Defaults to
            ``top_k``.

    Returns:
        The composed retriever and its effective top-k.
    """
    retriever: Retriever = base
    if rewriter is not None:
        retriever = MultiQueryRetriever(inner=retriever, rewriter=rewriter)

    if reranker is None:
        return RetrievalPlan(retriever=retriever, top_k=top_k)

    return RetrievalPlan(
        retriever=RerankingRetriever(
            inner=retriever,
            reranker=reranker,
            over_fetch_n=rerank_over_fetch_n,
        ),
        # WHY the final count changes: the reranker over-fetches a wider set and
        #     then narrows it, so what the engine receives is this number, not
        #     the one the inner retriever was asked for.
        top_k=top_k if rerank_final_top_k is None else rerank_final_top_k,
    )


def build_retrieval_plan(
    strategy: str,
    vector_store: ChromaVectorStore,
    top_k: int = TOP_K_RESULTS,
    rerank_over_fetch_n: int = RERANK_OVER_FETCH_N,
) -> RetrievalPlan:
    """Build the production retrieval plan for a configured strategy name.

    The strategy names are presets over :func:`compose_retrieval`, so production
    and the eval harness share one composition rule.

    Args:
        strategy: ``dense`` or ``reranked`` (wired), or ``hybrid`` /
            ``multi_query`` (recognised but deferred — see ADR 0004).
        vector_store: The dense index every strategy is built over.
        top_k: Chunks the engine should end up with.
        rerank_over_fetch_n: Candidate width the reranked strategy over-fetches.

    Returns:
        The composed retriever and its effective top-k.

    Raises:
        ValueError: If the strategy is unknown, or recognised but not yet wired
            for production.
    """
    dense = DenseRetriever(vector_store)

    if strategy == "dense":
        return compose_retrieval(base=dense, top_k=top_k)

    if strategy == "reranked":
        return compose_retrieval(
            base=dense,
            reranker=CrossEncoderReranker(),
            top_k=top_k,
            rerank_over_fetch_n=rerank_over_fetch_n,
        )

    if strategy in _DEFERRED:
        raise ValueError(
            f"Retriever strategy {strategy!r} is validated in the eval harness but "
            f"not yet wired for production ({_DEFERRED[strategy]}); see ADR 0004."
        )

    raise ValueError(f"Unknown retriever strategy: {strategy!r}")
