"""DenseRetriever — the default adapter: pure dense vector search.

RAG Pipeline Position:
    Query -> [DenseRetriever -> ChromaVectorStore] -> list[SearchResult]

This is the baseline retrieval strategy and the QueryEngine's default. It is a
thin adapter that presents the `Retriever` interface over `ChromaVectorStore`,
whose own `query(query_text=..., top_k=...)` already returns `SearchResult`s.

Design Decision:
    The adapter exists (rather than passing the store directly) so the seam is
    uniform: dense, hybrid, reranked, and multi-query all expose `retrieve()`.
    The store's `query` keyword (`query_text=`) is an implementation detail this
    adapter hides behind the Protocol's positional `query`.
"""

from __future__ import annotations

from src.vector_store import ChromaVectorStore, SearchResult


class DenseRetriever:
    """Dense retrieval over a Chroma collection, behind the Retriever seam."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        """Wrap a vector store.

        Args:
            vector_store: The dense index to query. The adapter never reaches
                past its public `query` method.
        """
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the `top_k` nearest chunks to `query` by cosine similarity.

        Args:
            query: Natural-language query text.
            top_k: Number of results to return.

        Returns:
            SearchResult list ordered by descending similarity (empty if the
            index has no documents).
        """
        return self._vector_store.query(query_text=query, top_k=top_k)
