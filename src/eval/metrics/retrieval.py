"""
Retrieval metrics — Recall@k, MRR@k, nDCG@k.

Eval Harness Position:
  Retriever → retrieved_chunk_ids → [METRICS] ← gold_chunk_ids
                                     ^^^^^^^
  Pure functions over (gold_chunk_ids, retrieved_chunk_ids). No I/O,
  no LLM calls. Fast, deterministic, unit-testable in isolation.

Design decisions:
  - Operate on opaque string IDs, not chunk objects, so the same
    metrics work over Chroma chunk IDs, BM25 doc IDs, or any other
    identifier scheme.
  - Empty gold returns NaN (not 0.0) because the metric is undefined,
    not zero. The aggregator drops NaN values per-metric per-question.
  - nDCG uses binary relevance (gold or not gold) — graded relevance
    would require richer labels, deferred to a future phase.
"""

from __future__ import annotations

import math
from typing import Sequence

# Sentinel returned for undefined metrics (empty gold set).
# Callers should check math.isnan() and skip these in aggregation.
NAN = float("nan")


def recall_at_k(
    gold_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of gold chunks found in the top-k retrieved results.

    Args:
        gold_chunk_ids: Ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ranked list of retrieved chunk IDs (best first).
        k: Cutoff — only the first k entries of retrieved_chunk_ids count.

    Returns:
        |gold ∩ retrieved[:k]| / |gold|, or NaN if gold is empty.
    """
    if not gold_chunk_ids:
        return NAN

    gold_set = set(gold_chunk_ids)
    top_k = set(retrieved_chunk_ids[:k])
    return len(gold_set & top_k) / len(gold_set)


def mrr_at_k(
    gold_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant chunk in the top-k results.

    MRR captures how early the first relevant result appears. It rewards
    systems that surface at least one correct answer near the top of the list.

    Args:
        gold_chunk_ids: Ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ranked list of retrieved chunk IDs (best first).
        k: Cutoff — only the first k entries of retrieved_chunk_ids count.

    Returns:
        1/rank of the first hit (1-indexed), 0.0 if no hit in top-k,
        or NaN if gold is empty.
    """
    if not gold_chunk_ids:
        return NAN

    gold_set = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    gold_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at rank k (binary relevance).

    nDCG measures both the presence and the rank of relevant results.
    Higher-ranked hits contribute more than lower-ranked ones (logarithmic
    discount). Normalizing by the ideal DCG makes the score comparable
    across queries with different numbers of gold chunks.

    Args:
        gold_chunk_ids: Ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ranked list of retrieved chunk IDs (best first).
        k: Cutoff — only the first k entries of retrieved_chunk_ids count.

    Returns:
        DCG / IDCG in [0.0, 1.0], 0.0 if IDCG is 0, or NaN if gold is empty.
    """
    if not gold_chunk_ids:
        return NAN

    gold_set = set(gold_chunk_ids)

    # Compute actual DCG: sum 1/log2(rank+1) for each hit in retrieved[:k].
    # WHY rank+1 inside log2: standard DCG convention so rank-1 gives
    # log2(2)=1 (a gain of 1.0, not infinity). Without +1, log2(1)=0 → div/0.
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1)
        if chunk_id in gold_set
    )

    # Compute ideal DCG: assume all gold chunks are retrieved at the top ranks.
    # We only credit up to min(|gold|, k) ideal positions.
    ideal_hits = min(len(gold_chunk_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    # TRADE-OFF: returning 0.0 when IDCG=0 instead of NaN because an
    # empty-gold case is already handled above; IDCG=0 here means k=0,
    # which is a caller error, and 0.0 is a safe neutral value.
    if idcg == 0.0:
        return 0.0

    return dcg / idcg
