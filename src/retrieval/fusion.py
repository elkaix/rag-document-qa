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
        Fused ranking, IDs ordered by descending fused score. The result is
        deterministic for a given input, but the resolution of a tie is not a
        documented guarantee — see below.

    Note:
        Two caveats worth knowing before reading a fused ordering as meaningful.

        *Ties resolve toward the list passed first.* Rank *r* in any list earns
        the same ``1 / (rrf_k + r)``, so disjoint lists tie at every rank and
        Python's stable sort breaks each tie for whichever list was passed
        first. ``BM25HybridRetriever`` passes sparse before dense, so a
        sparse-only hit wins the positional coin-flip against a dense-only hit
        at the same rank. At ``top_k=1`` that decides the single result
        returned, and a sparse-only hit carries score 0.0. Swapping the
        arguments would flip the bias, not remove it: with no common scale
        there is no principled tie-break, which is the price of rank fusion.

        *Exactly-tied sums may still order arbitrarily.* Float addition is not
        associative, so ids whose scores are mathematically equal can differ in
        the last bit and sort against insertion order. Deterministic, but not
        first-seen.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        # WHY the per-list dedupe: an id repeated inside one ranking would
        #     otherwise earn a contribution per occurrence, so a retriever that
        #     emitted the same chunk three times could outrank a genuinely
        #     better result on repetition alone. Every current caller passes
        #     unique ids, which is exactly why this would fail silently if one
        #     ever stopped. Only the best (lowest) rank counts.
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
