"""Retrieval package — the Retriever seam and its adapters (issue #16, step 4).

One interface, `Retriever` (`retrieve(query, top_k) -> list[SearchResult]`), with
adapters that either conform directly or compose an inner Retriever:

- `DenseRetriever` — dense vector search (the default).
- `BM25HybridRetriever` — sparse (BM25) + dense fused by RRF; conforms directly.
- `RerankingRetriever` — composes an inner Retriever, over-fetches, re-scores with
  a cross-encoder (`CrossEncoderReranker`).
- `MultiQueryRetriever` — composes an inner Retriever, fans out rewritten queries
  (`QueryRewriter`), dedups.

`RefusalHandler` is not a Retriever — it is an answerability gate the QueryEngine
applies after retrieval. These modules were promoted from `src/eval/` so
production can activate the eval-proven levers by configuration. See
[ADR 0004](../../docs/adr/0004-retriever-seam-and-query-engine.md).
"""

from __future__ import annotations

from src.retrieval.base import Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.factory import build_retriever
from src.retrieval.hybrid import BM25HybridRetriever, reciprocal_rank_fusion
from src.retrieval.query_rewriter import MultiQueryRetriever, QueryRewriter
from src.retrieval.refusal_handler import RefusalHandler
from src.retrieval.reranker import CrossEncoderReranker, RerankingRetriever

__all__ = [
    "Retriever",
    "DenseRetriever",
    "build_retriever",
    "BM25HybridRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "RerankingRetriever",
    "QueryRewriter",
    "MultiQueryRetriever",
    "RefusalHandler",
]
