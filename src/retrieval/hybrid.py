"""BM25HybridRetriever — Reciprocal Rank Fusion of sparse (BM25) + dense (Chroma) retrieval.

Pipeline position:
    query → [BM25 + Dense → RRF] → top-K SearchResult → Reranker / Generator

Phase 2 lever 2c. The retriever keeps two parallel ranked lists (BM25 over
documents, dense over Chroma vectors), then fuses them with RRF:

    score(d) = sum over r in {dense, sparse} of 1 / (rrf_k + rank_r(d))

Why RRF over weighted-sum: RRF is parameter-light (one constant), robust to
score-scale differences across the two retrievers, and the literature shows
it consistently matches or beats tuned weighted-sum on benchmarks like BEIR.
"""

from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from src.vector_store import ChromaVectorStore, SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse multiple ranked ID lists into one via Reciprocal Rank Fusion.

    Args:
        rankings: Iterable of ranked ID sequences. Each sequence is one
            retriever's ranking, most-relevant first.
        rrf_k: RRF constant (60 is the textbook default; smaller emphasizes
            top-rank items more, larger flattens contributions).

    Returns:
        Fused ranking, IDs ordered by descending fused score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


class BM25HybridRetriever:
    """Retriever that fuses BM25 and dense Chroma rankings.

    The BM25 index is built once at construction time over a `documents` mapping.
    Each retrieve() call queries both BM25 and the vector store, then RRF-fuses
    the two rankings before truncating to the requested top-K.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        documents: dict[str, str],
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """Build the BM25 index and store retrieval parameters.

        Args:
            vector_store: Dense retriever (Chroma collection wrapper).
            documents: Mapping of chunk_id → raw document text. BM25 needs
                tokenized text; this dict is the authoritative corpus.
            bm25_top_k: Number of candidates BM25 returns per query.
            dense_top_k: Number of candidates the dense retriever returns.
            rrf_k: RRF fusion constant.
        """
        self._vector_store = vector_store
        self._chunk_ids = list(documents.keys())
        # WHY simple split: rank-bm25 expects pre-tokenized inputs. A whitespace
        # split is good enough for English RAG corpora; nltk stems/stopwords
        # would help marginally but add a runtime dep we don't want here.
        tokenized = [documents[i].lower().split() for i in self._chunk_ids]
        self._bm25 = BM25Okapi(tokenized)
        self._documents = documents
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Run BM25 + dense in parallel, RRF-fuse, return top-K SearchResults.

        Args:
            query: Natural-language query.
            top_k: Number of fused results to return.

        Returns:
            Top-K SearchResult ordered by fused score descending. Score on each
            result is the dense similarity (BM25 ranks aren't directly comparable;
            keeping dense score lets downstream rerankers/refusal-handlers reuse
            it as a confidence proxy).
        """
        # --- Sparse side -------------------------------------------------------
        sparse_scores = self._bm25.get_scores(query.lower().split())
        sparse_ranked = sorted(
            range(len(self._chunk_ids)),
            key=lambda i: sparse_scores[i],
            reverse=True,
        )[: self._bm25_top_k]
        sparse_ids = [self._chunk_ids[i] for i in sparse_ranked]

        # --- Dense side --------------------------------------------------------
        dense_results = self._vector_store.query(
            query_text=query, top_k=self._dense_top_k,
        )
        dense_ids = [r.chunk_id for r in dense_results]
        dense_score_by_id = {r.chunk_id: r.score for r in dense_results}

        # --- Fusion ------------------------------------------------------------
        fused_ids = reciprocal_rank_fusion(
            [sparse_ids, dense_ids], rrf_k=self._rrf_k,
        )[:top_k]

        return [
            SearchResult(
                chunk_id=cid,
                content=self._documents[cid],
                score=dense_score_by_id.get(cid, 0.0),
                metadata={},
                doc_id="",
            )
            for cid in fused_ids
        ]
