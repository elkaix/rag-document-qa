"""Reciprocal Rank Fusion — combine several ranked lists into one.

RAG Pipeline Position:
    two or more ranked id lists -> [RRF] -> one ranked id list

What concept it teaches:
    Rank-based fusion. Two retrievers (or two phrasings of one query) produce
    scores in incomparable spaces — a BM25 term-frequency score and a cosine
    similarity have no common scale, and neither do the similarities of two
    different queries. Their *ranks*, however, always compare.

Why this approach over alternatives:
    Weighted-sum fusion needs per-retriever normalisation and a weight to tune
    per corpus. RRF has one constant, is insensitive to score scale, and the
    literature (Cormack et al. 2009, and every BEIR-era comparison since) shows
    it matching or beating tuned weighted-sum.

Design Decision:
    This lives in its own module because it has two callers —
    ``BM25HybridRetriever`` fusing sparse with dense, and
    ``MultiQueryRetriever`` fusing one expansion's ranking with another's. One
    caller would have been a hypothetical seam; two make it a real one.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse multiple ranked ID lists into one via Reciprocal Rank Fusion.

    An id's fused score is the sum over the lists it appears in of
    ``1 / (rrf_k + rank)``, so appearing high in several lists beats appearing
    highest in one.

    Args:
        rankings: Iterable of ranked ID sequences. Each sequence is one
            retriever's (or one query's) ranking, most-relevant first.
        rrf_k: RRF constant (60 is the textbook default; smaller emphasizes
            top-rank items more, larger flattens contributions).

    Returns:
        Fused ranking, IDs ordered by descending fused score. Ties keep the
        order in which the ids were first seen, so the fusion is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
