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

from src.vector_store import SearchResult


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
            top_k: Maximum number of results to return, best first.

        Returns:
            SearchResult list ordered by descending relevance (possibly empty).
        """
        ...
