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

    ADR 0009 finished the job: ``hybrid`` and ``multi_query``, which ADR 0004
    left recognised-but-unbuildable, are now presets like the other two. Each
    preset activates one lever; stacking several at once is what calling
    ``compose_retrieval`` directly is for, and the eval harness does exactly
    that.

Design Decision:
    ``compose_retrieval`` takes an already-built base Retriever rather than a
    vector store, so it composes without touching storage, an embedder, or a
    cross-encoder download. That is what makes the rule unit-testable — the
    reason it had no test before.
"""

from __future__ import annotations

from dataclasses import dataclass

# WHY the leading-underscore Protocols are imported rather than redeclared:
#      an identical local copy is a *different* type to a type checker, so
#      passing a real QueryRewriter through this module failed to type-check
#      even though it satisfied the contract.
from src.config import (
    HYBRID_BM25_TOP_K,
    HYBRID_DENSE_TOP_K,
    HYBRID_RRF_K,
    MAX_QUERY_EXPANSIONS,
    QUERY_REWRITER_MODEL,
    RERANK_OVER_FETCH_N,
    TOP_K_RESULTS,
)
from src.retrieval.base import Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import BM25HybridRetriever
from src.retrieval.query_rewriter import (
    MultiQueryRetriever,
    QueryRewriter,
    _LLMHandler,
    _Rewriter,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankingRetriever,
    _Reranker,
)
from src.vector_store import ChromaVectorStore


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


def _production_rewriter(llm: _LLMHandler | None) -> QueryRewriter:
    """Build the configured query rewriter, or say precisely why it cannot.

    Args:
        llm: The handler the rewriter calls for expansions.

    Returns:
        A rewriter wired to ``QUERY_REWRITER_MODEL``.

    Raises:
        ValueError: If no rewriter model is configured, or no LLM handler was
            supplied to build one with.

    WHY this raises instead of returning None: ``QueryRewriter(model=None)`` is
        a legal pass-through, so an unconfigured multi_query strategy would
        compose a MultiQueryRetriever that expands nothing and behaves exactly
        like dense retrieval — a deployment believing it had recall it did not
        have, with no error anywhere. Failing at startup is the whole point of
        validating at the boundary.
    """
    if QUERY_REWRITER_MODEL is None:
        raise ValueError(
            "Retriever strategy 'multi_query' needs a rewriter model: set "
            "QUERY_REWRITER_MODEL (see src/config.py). It is unset, and a "
            "rewriter without a model expands nothing."
        )
    if llm is None:
        raise ValueError(
            "Retriever strategy 'multi_query' needs an LLM handler to expand "
            "queries with; build_retrieval_plan was called without one."
        )
    return QueryRewriter(
        model=QUERY_REWRITER_MODEL,
        max_expansions=MAX_QUERY_EXPANSIONS,
        llm=llm,
    )


def build_retrieval_plan(
    strategy: str,
    vector_store: ChromaVectorStore,
    top_k: int = TOP_K_RESULTS,
    rerank_over_fetch_n: int = RERANK_OVER_FETCH_N,
    llm: _LLMHandler | None = None,
) -> RetrievalPlan:
    """Build the production retrieval plan for a configured strategy name.

    The strategy names are presets over :func:`compose_retrieval`, so production
    and the eval harness share one composition rule. Each preset activates a
    single lever over the dense baseline; the eval harness stacks several at
    once by calling :func:`compose_retrieval` directly.

    Args:
        strategy: ``dense``, ``hybrid``, ``reranked``, or ``multi_query``.
        vector_store: The index every strategy is built over.
        top_k: Chunks the engine should end up with.
        rerank_over_fetch_n: Candidate width the reranked strategy over-fetches.
        llm: Handler the ``multi_query`` strategy expands queries with. Unused
            by every other strategy.

    Returns:
        The composed retriever and its effective top-k.

    Raises:
        ValueError: If the strategy is unknown, or is known but its required
            configuration is missing.
    """
    if strategy == "dense":
        return compose_retrieval(base=DenseRetriever(vector_store), top_k=top_k)

    if strategy == "hybrid":
        # WHY the retriever takes the store and not a corpus: it keeps its BM25
        #     index in step with the store's revision, so documents ingested or
        #     deleted after startup are reflected on the next query. That
        #     freshness rule is what ADR 0004 deferred this strategy for; see
        #     ADR 0009.
        return compose_retrieval(
            base=BM25HybridRetriever(
                vector_store,
                bm25_top_k=HYBRID_BM25_TOP_K,
                dense_top_k=HYBRID_DENSE_TOP_K,
                rrf_k=HYBRID_RRF_K,
            ),
            top_k=top_k,
        )

    if strategy == "reranked":
        return compose_retrieval(
            base=DenseRetriever(vector_store),
            reranker=CrossEncoderReranker(),
            top_k=top_k,
            rerank_over_fetch_n=rerank_over_fetch_n,
        )

    if strategy == "multi_query":
        return compose_retrieval(
            base=DenseRetriever(vector_store),
            rewriter=_production_rewriter(llm),
            top_k=top_k,
        )

    raise ValueError(
        f"Unknown retriever strategy: {strategy!r}. "
        "Expected one of: dense, hybrid, reranked, multi_query."
    )
