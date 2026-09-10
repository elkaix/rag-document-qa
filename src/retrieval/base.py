"""Retriever seam — the one interface every retrieval strategy hides behind.

RAG Pipeline Position:
    Query -> [RETRIEVER] -> list[SearchResult] -> QueryEngine -> Answer
              ^^^^^^^^^
    This module defines the *seam*: a single `retrieve(query, top_k)` interface
    that dense, hybrid, reranked, and multi-query retrieval all present. The
    QueryEngine (step 4b) depends only on this Protocol, so a retrieval strategy
    validated offline in the eval harness is promoted to production by
    *configuration*, not by a code change.

Design Decision:
    A `Protocol` (not an ABC) per the project standard — retrieval strategies
    conform structurally without inheriting, and `@runtime_checkable` lets tests
    assert conformance with `isinstance`. `SearchResult` stays the shared result
    type (defined in `vector_store`) so no adapter invents its own shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain import SearchResult


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into ranked chunks.

    Implementations either *conform* directly (a dense store wrapper, the BM25
    hybrid retriever) or *compose* an inner Retriever (reranking, multi-query),
    always presenting this same interface outward.
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to `top_k` chunks most relevant to `query`.

        Args:
            query: Natural-language query.
            top_k: Maximum number of results to return, most relevant first.

        Returns:
            SearchResult list ordered by descending *relevance*, possibly empty.

        The ordering guarantee is deliberately about relevance and not about
        ``SearchResult.score``. Every implementation ranks its results, but each
        one scores them in its own space: ``DenseRetriever`` reports cosine
        similarity, ``RerankingRetriever`` reports a cross-encoder logit, and
        ``BM25HybridRetriever`` orders by fused RRF rank and reports ``0.0`` for
        a sparse-only hit, because no cosine similarity exists for one. So
        ``results[0]`` is the most relevant chunk, but ``results[0].score`` is
        not necessarily the largest score in the list, and scores from two
        different strategies do not compare at all.

        A caller asking "how similar is the best match?" must therefore read
        ``max(r.score for r in results)`` rather than index position 0 — the
        distinction that ``RefusalHandler.should_refuse`` and ``RAGBackend``'s
        confidence metric both had to be corrected for (ADR 0009). Callers that
        only need the ranking can rely on list order as before.
        """
        ...
