"""CrossEncoderReranker — re-scores retrieval candidates with a cross-encoder model.

Pipeline position:
    Retriever top-N → [CrossEncoderReranker] → top-K → Refusal / Generator

Phase 2 lever 2d. Cross-encoders (single-tower models that consume both
the query and a candidate together) typically outperform bi-encoder retrieval
in precision at the cost of latency. We use ms-marco-MiniLM-L-6-v2 — small
enough to run on CPU in milliseconds per pair, trained on MS MARCO so the
ranking signal transfers well to general-domain QA.
"""

from __future__ import annotations

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
