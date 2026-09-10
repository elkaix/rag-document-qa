"""BM25HybridRetriever — Reciprocal Rank Fusion of sparse (BM25) + dense (Chroma) retrieval.

RAG Pipeline Position:
    query -> [BM25 + Dense -> RRF] -> top-K SearchResult -> Reranker / Generator

The retriever keeps two parallel ranked lists — BM25 over the whole chunk
corpus, dense over the Chroma vectors — then fuses them with RRF:

    score(d) = sum over r in {dense, sparse} of 1 / (rrf_k + rank_r(d))

Why RRF over weighted-sum: RRF is parameter-light (one constant), robust to
score-scale differences across the two retrievers, and the literature shows it
consistently matches or beats tuned weighted-sum on benchmarks like BEIR. See
``fusion`` for the two ways a fused ordering can mislead you.

Design Decision (why the corpus is not a constructor argument):
    This retriever used to take a ``dict[str, str]`` snapshot, which is what
    made ADR 0004 defer the strategy: the sparse side froze at whatever had
    been ingested when the object was built. It now owns a :class:`ChunkCorpus`,
    which re-reads the collection whenever the store's revision moves. Nothing
    varies across that seam — there is one corpus implementation — so it is an
    *internal* seam, constructed here rather than injected.

Design Decision (why the sparse half lives in its own module):
    ``sparse_index`` builds and scores the BM25 index; this module fuses its
    output with the dense ranking. Two responsibilities, and the sparse one is
    pure — a snapshot in, ids out — so it can be tested without a vector store.
"""

from __future__ import annotations

import logging

from src.domain import Chunk, SearchResult
from src.retrieval.corpus import ChunkCorpus, CorpusSnapshot
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse_index import SparseIndex, build_sparse_index, rank_ids
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class BM25HybridRetriever:
    """Retriever that fuses BM25 and dense Chroma rankings behind one seam."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """Wire the retriever to a store; build nothing yet.

        Args:
            vector_store: The collection both halves read. The dense side
                queries it; the sparse side mirrors it through a ChunkCorpus.
            bm25_top_k: Number of candidates BM25 contributes per query.
            dense_top_k: Number of candidates dense retrieval contributes.
            rrf_k: RRF fusion constant.
        """
        self._vector_store = vector_store
        self._corpus = ChunkCorpus(vector_store)
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_k = rrf_k
        self._index: SparseIndex | None = None
        self._built_from: CorpusSnapshot | None = None

    def _sparse_index(self) -> tuple[SparseIndex, CorpusSnapshot]:
        """Return the BM25 index for the current corpus, rebuilding if stale.

        The snapshot is cached by *identity*: ``ChunkCorpus`` hands back the
        same object while the store's revision is unchanged, so the freshness
        rule lives in one module and this one only asks.
        """
        snapshot = self._corpus.snapshot()
        index = self._index
        if index is None or self._built_from is not snapshot:
            index = build_sparse_index(snapshot)
            logger.debug(
                "Rebuilt BM25 index at store revision %d (%d chunks).",
                snapshot.revision,
                len(snapshot),
            )
        # BUG FIX: this returned ``self._index`` — the attribute, not the local
        #     just built. Retrieval runs on more than one thread (the websocket
        #     path drives it through ``loop.run_in_executor``, and the backend
        #     is a lifespan singleton), so a concurrent rebuild could replace
        #     the attribute between the check and the return, handing this
        #     caller an index built from a *different* snapshot than the one
        #     returned alongside it. The two are then inconsistent:
        #     ``index.chunk_ids`` can name chunks absent from ``snapshot.chunks``
        #     and the caller silently drops them. Returning the local pairs each
        #     caller with the index it actually validated.
        # WHY the two attributes are published last, together: they are the
        #     cache, and a reader that saw the new index paired with the old
        #     ``_built_from`` would rebuild on every call forever. Writing them
        #     after the work means a racing thread sees either the old
        #     consistent pair or the new one, never a torn one. Two threads may
        #     still duplicate a rebuild — that costs time, not correctness, and
        #     ``ChunkCorpus`` already serialises the expensive half.
        self._index = index
        self._built_from = snapshot
        return index, snapshot

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Run BM25 + dense, RRF-fuse the two rankings, return the top `top_k`.

        Args:
            query: Natural-language query.
            top_k: Number of fused results to return.

        Returns:
            Up to `top_k` SearchResults in fused-rank order. Each carries the
            chunk's real metadata and doc_id, so a citation renders the same
            whichever half of the retriever surfaced it.

        Note:
            ``score`` is the dense cosine similarity when the dense side also
            returned the chunk, and 0.0 for a chunk only BM25 found — the two
            score spaces are not comparable, and inventing a number in cosine
            space for a BM25 hit would be worse than reporting none. Ordering
            is by fused rank, *not* by this field; consumers that need "is
            anything here similar enough" must read the best score in the list
            rather than assume position 0 holds it (``RefusalHandler`` does).
        """
        # BUG FIX: both halves were sized from the constructor's constants
        #     alone, so a caller asking for more results than either constant
        #     silently got fewer. ``POST /api/query`` accepts top_k up to 50
        #     (src/api/models.py) while both constants default to 20, so
        #     ``top_k=50`` returned at most 40 chunks — usually far fewer, since
        #     BM25 drops its zero-score non-matches — with nothing telling the
        #     caller the number they asked for was not honoured. Fusing needs at
        #     least ``top_k`` candidates per half to be able to return ``top_k``.
        fetch_n = max(self._dense_top_k, top_k)
        index, snapshot = self._sparse_index()
        sparse_ids = rank_ids(query, index, max(self._bm25_top_k, top_k))

        dense_results = self._vector_store.query(query_text=query, top_k=fetch_n)
        dense_by_id = {r.chunk_id: r for r in dense_results}

        fused_ids = reciprocal_rank_fusion(
            [sparse_ids, list(dense_by_id.keys())],
            rrf_k=self._rrf_k,
        )[:top_k]

        results: list[SearchResult] = []
        for chunk_id in fused_ids:
            dense_hit = dense_by_id.get(chunk_id)
            if dense_hit is not None:
                results.append(dense_hit)
                continue
            chunk = snapshot.chunks.get(chunk_id)
            if chunk is None:
                # Defensive, and unreachable by construction: every sparse id
                # was read out of this very snapshot, which is immutable, so a
                # delete landing mid-query cannot empty it. Kept because the
                # alternative to skipping an unciteable id is a KeyError in the
                # request path, and because it is the one place a future
                # sparse-id source that does *not* come from the snapshot would
                # show up.
                continue
            results.append(_as_result(chunk))
        return results


def _as_result(chunk: Chunk) -> SearchResult:
    """Present a corpus chunk as a scoreless SearchResult.

    Args:
        chunk: A chunk BM25 ranked but dense retrieval did not return.

    Returns:
        The chunk at the Retriever seam, carrying its real metadata and doc_id
        so citations survive, with score 0.0 (see ``retrieve``'s note).
    """
    return SearchResult(
        content=chunk.content,
        metadata=chunk.metadata,
        score=0.0,
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
    )
