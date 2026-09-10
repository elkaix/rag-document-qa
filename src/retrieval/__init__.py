"""Retrieval package — the Retriever seam and its adapters (issue #16, step 4).

One interface, `Retriever` (`retrieve(query, top_k) -> list[SearchResult]`), with
adapters that either conform directly or compose an inner Retriever:

- `DenseRetriever` — dense vector search (the default).
- `BM25HybridRetriever` — sparse (BM25) + dense fused by RRF; conforms directly.
  Its corpus is a `ChunkCorpus`, which re-reads the store when its revision moves.
- `RerankingRetriever` — composes an inner Retriever, over-fetches, re-scores with
  a cross-encoder (`CrossEncoderReranker`).
- `MultiQueryRetriever` — composes an inner Retriever, fans out rewritten queries
  (`QueryRewriter`), and fuses the per-query rankings by RRF.
- `reciprocal_rank_fusion` — the rank-based fusion used by `BM25HybridRetriever`
  (sparse vs. dense) and `MultiQueryRetriever` (one ranking per expansion).
  `RerankingRetriever` does not fuse — it re-scores a single ranking.

`RefusalHandler` is not a Retriever — it is an answerability gate the QueryEngine
applies after retrieval. These modules were promoted from `src/eval/` so
production can activate the eval-proven levers by configuration. See
[ADR 0004](../../docs/adr/0004-retriever-seam-and-query-engine.md) for the seam and
[ADR 0009](../../docs/adr/0009-wire-hybrid-and-multi-query.md) for wiring the last two
strategies into production.
"""

from __future__ import annotations

from src.retrieval.base import Retriever
from src.retrieval.composition import (
    RetrievalPlan,
    build_retrieval_plan,
    compose_retrieval,
)
from src.retrieval.corpus import ChunkCorpus, CorpusSnapshot
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import BM25HybridRetriever
from src.retrieval.query_rewriter import MultiQueryRetriever, QueryRewriter
from src.retrieval.refusal_handler import RefusalHandler
from src.retrieval.reranker import CrossEncoderReranker, RerankingRetriever

__all__ = [
    "BM25HybridRetriever",
    "ChunkCorpus",
    "CorpusSnapshot",
    "CrossEncoderReranker",
    "DenseRetriever",
    "MultiQueryRetriever",
    "QueryRewriter",
    "RefusalHandler",
    "RerankingRetriever",
    "RetrievalPlan",
    "Retriever",
    "build_retrieval_plan",
    "compose_retrieval",
    "reciprocal_rank_fusion",
]
