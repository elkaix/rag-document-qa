"""Cross-encoder reranking — the re-scorer and the Retriever adapter that composes it.

Pipeline position:
    Retriever top-N → [RerankingRetriever → CrossEncoderReranker] → top-K → Generator

Two collaborators live here:

- `CrossEncoderReranker` re-scores a candidate list. Cross-encoders (single-tower
  models that consume the query and a candidate together) typically outperform
  bi-encoder retrieval in precision at the cost of latency. We use
  ms-marco-MiniLM-L-6-v2 — small enough to run on CPU in milliseconds per pair,
  trained on MS MARCO so the ranking signal transfers to general-domain QA.
- `RerankingRetriever` presents the `Retriever` interface by *composing* an inner
  Retriever: it over-fetches a wide candidate set, then narrows via the reranker.
  This is the "compose rather than conform" adapter from ADR 0004.
"""

from __future__ import annotations

from typing import Protocol

from src.retrieval.base import Retriever
from src.vector_store import SearchResult


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder to re-score retrieval candidates."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.MODEL_NAME)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        final_top_k: int,
    ) -> list[SearchResult]:
        """Re-score candidates against the query and return top-K reranked.

        Args:
            query: Original user query.
            candidates: Pre-retrieved chunks (typically top-N from a base retriever).
            final_top_k: How many to keep after reranking.

        Returns:
            Top-K SearchResult ordered by descending cross-encoder score. The
            original `score` field is *replaced* with the cross-encoder score so
            downstream consumers reading `result.score` get the more precise signal.
        """
        if not candidates:
            return []
        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)
        scored = sorted(
            zip(candidates, scores), key=lambda t: t[1], reverse=True,
        )[:final_top_k]
        return [
            SearchResult(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                content=c.content,
                score=float(s),
                metadata=c.metadata,
            )
            for c, s in scored
        ]


class _Reranker(Protocol):
    """Structural type for a candidate re-scorer (the one collaborator we inject)."""

    def rerank(
        self, query: str, candidates: list[SearchResult], final_top_k: int,
    ) -> list[SearchResult]: ...


class RerankingRetriever:
    """Retriever adapter: over-fetch from an inner Retriever, then cross-encode.

    Presents `retrieve(query, top_k)` while delegating candidate generation to an
    inner Retriever and precision re-scoring to a reranker — so reranking is
    interchangeable with any other retrieval strategy behind the same seam.
    """

    def __init__(
        self,
        inner: Retriever,
        reranker: _Reranker,
        over_fetch_n: int,
    ) -> None:
        """Compose an inner Retriever with a candidate re-scorer.

        Args:
            inner: The Retriever that produces the initial candidate set.
            reranker: The cross-encoder re-scorer applied to those candidates.
            over_fetch_n: How many candidates to pull from `inner` before
                reranking. Wider than the final `top_k` so the precise reranker
                has real choice; the eval-tuned default is 20.
        """
        self._inner = inner
        self._reranker = reranker
        self._over_fetch_n = over_fetch_n

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Over-fetch `over_fetch_n` candidates, rerank, return the top `top_k`.

        Args:
            query: Natural-language query.
            top_k: Final number of results after reranking.

        Returns:
            The reranked top-`top_k` SearchResults (empty if the inner retriever
            found nothing).
        """
        candidates = self._inner.retrieve(query, top_k=self._over_fetch_n)
        return self._reranker.rerank(query, candidates, final_top_k=top_k)
