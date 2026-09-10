"""BM25HybridRetriever — Reciprocal Rank Fusion of sparse (BM25) + dense (Chroma) retrieval.

RAG Pipeline Position:
    query -> [BM25 + Dense -> RRF] -> top-K SearchResult -> Reranker / Generator

The retriever keeps two parallel ranked lists — BM25 over the whole chunk
corpus, dense over the Chroma vectors — then fuses them with RRF:

    score(d) = sum over r in {dense, sparse} of 1 / (rrf_k + rank_r(d))

Why RRF over weighted-sum: RRF is parameter-light (one constant), robust to
score-scale differences across the two retrievers, and the literature shows it
consistently matches or beats tuned weighted-sum on benchmarks like BEIR.

Design Decision (why the corpus is not a constructor argument):
    This retriever used to take a ``dict[str, str]`` snapshot, which is what
    made ADR 0004 defer the strategy: the sparse side froze at whatever had
    been ingested when the object was built. It now owns a :class:`ChunkCorpus`,
    which re-reads the collection whenever the store's revision moves. Nothing
    varies across that seam — there is one corpus implementation — so it is an
    *internal* seam, constructed here rather than injected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.domain import Chunk, SearchResult
from src.retrieval.corpus import ChunkCorpus, CorpusSnapshot
from src.retrieval.fusion import reciprocal_rank_fusion
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Split text into BM25 terms.

    WHY a plain lowercase split: rank-bm25 expects pre-tokenized input, and a
    whitespace split is good enough for English RAG corpora. Stemming and
    stop-word removal would help marginally and would add a runtime dependency
    (nltk) plus a corpus-language assumption we do not want to make here.
    """
    return text.lower().split()


@dataclass(frozen=True)
class _SparseIndex:
    """A BM25 index over one corpus snapshot, or an explicitly disabled one.

    Attributes:
        chunk_ids: Corpus ids, positionally aligned with the BM25 documents.
        bm25: The index, or None when the corpus cannot support one (empty
            collection, or every chunk tokenizing to nothing). None means
            "skip the sparse side", never "crash on the next query".
    """

    chunk_ids: list[str]
    bm25: BM25Okapi | None


def _build_sparse_index(snapshot: CorpusSnapshot) -> _SparseIndex:
    """Build a BM25 index over a corpus snapshot, degrading instead of raising.

    Args:
        snapshot: The corpus to index.

    Returns:
        An index whose ``bm25`` is None when BM25 is not defined over this
        corpus.

    BUG FIX: ``BM25Okapi`` divides by the corpus size to get the average
        document length and by the term count to get the average IDF, so an
        empty collection and a collection whose chunks all tokenize to nothing
        both raised ZeroDivisionError from inside the library. An empty index
        is a normal state for a fresh deployment — the first query after boot
        and before any upload hit exactly this path — so it must degrade to
        dense-only retrieval rather than fail the request.
    """
    chunk_ids = list(snapshot.chunks.keys())
    tokenized = [_tokenize(snapshot.chunks[cid].content) for cid in chunk_ids]
    if not any(tokenized):
        logger.debug("Sparse index skipped: corpus has no tokenizable text.")
        return _SparseIndex(chunk_ids=chunk_ids, bm25=None)
    return _SparseIndex(chunk_ids=chunk_ids, bm25=BM25Okapi(tokenized))


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
        self._index: _SparseIndex | None = None
        self._built_from: CorpusSnapshot | None = None

    def _sparse_index(self) -> tuple[_SparseIndex, CorpusSnapshot]:
        """Return the BM25 index for the current corpus, rebuilding if stale.

        The snapshot is cached by *identity*: ``ChunkCorpus`` hands back the
        same object while the store's revision is unchanged, so the freshness
        rule lives in one module and this one only asks.
        """
        snapshot = self._corpus.snapshot()
        if self._index is None or self._built_from is not snapshot:
            self._index = _build_sparse_index(snapshot)
            self._built_from = snapshot
            logger.debug(
                "Rebuilt BM25 index at store revision %d (%d chunks).",
                snapshot.revision,
                len(snapshot),
            )
        return self._index, snapshot

    def _sparse_ids(self, query: str, index: _SparseIndex) -> list[str]:
        """Return BM25's ranked candidate ids for `query` (empty when disabled).

        Only chunks with a positive BM25 score are returned. A zero score means
        the query shares no term with the chunk; feeding those into the fusion
        would let arbitrary non-matches inherit a rank, and with a corpus
        smaller than ``bm25_top_k`` that is every chunk in the collection.

        Note:
            BM25Okapi's IDF is ``log(N - df + 0.5) - log(df + 0.5)``, which is
            zero for a term appearing in exactly half the corpus and negative
            above that. On a two-chunk collection a perfectly discriminating
            term therefore scores 0 and is dropped here — the sparse side goes
            quiet on corpora too small for term statistics to mean anything,
            which is the right thing for it to do.

        TRADE-OFF: scoring sorts the whole corpus, so this is O(N log N) per
            query in the number of chunks. Fine at the scale a single-worker
            ChromaDB deployment serves; a corpus large enough to feel it wants
            a real inverted index (Elasticsearch, Vespa) rather than rank-bm25.
        """
        if index.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = index.bm25.get_scores(tokens)
        ranked = sorted(
            (i for i in range(len(index.chunk_ids)) if scores[i] > 0.0),
            key=lambda i: scores[i],
            reverse=True,
        )[: self._bm25_top_k]
        return [index.chunk_ids[i] for i in ranked]

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
        index, snapshot = self._sparse_index()
        sparse_ids = self._sparse_ids(query, index)

        dense_results = self._vector_store.query(query_text=query, top_k=self._dense_top_k)
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
                # The corpus moved between the sparse ranking and this lookup
                # (a delete landing mid-query). Dropping the id is correct:
                # the chunk no longer exists to cite.
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
