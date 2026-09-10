"""The sparse half of hybrid retrieval — a BM25 index over a corpus snapshot.

RAG Pipeline Position:
    corpus snapshot -> [BUILD BM25 INDEX] -> ranked chunk ids -> RRF fusion

What concept it teaches:
    Lexical (sparse) retrieval, and how it fails. BM25 scores a query against
    term statistics of the whole corpus, which means it is only defined when the
    corpus *has* term statistics — and a RAG deployment spends its first moments
    with an empty one. Everything here exists so that "BM25 is not defined for
    this corpus" is an ordinary return value rather than an exception in a
    request path.

Design Decision:
    Split out of ``hybrid`` so that module owns one thing: fusing two rankings.
    Building and scoring the sparse index is the other thing, it is pure (a
    snapshot in, ids out, no store and no I/O), and pure code is testable
    without a retriever, an embedder, or a vector store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.retrieval.corpus import CorpusSnapshot

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
class SparseIndex:
    """A BM25 index over one corpus snapshot, or an explicitly disabled one.

    Attributes:
        chunk_ids: Corpus ids, positionally aligned with the BM25 documents.
        bm25: The index, or None when the corpus cannot support one (empty
            collection, or every chunk tokenizing to nothing). None means
            "skip the sparse side", never "crash on the next query".
    """

    chunk_ids: list[str]
    bm25: BM25Okapi | None


def build_sparse_index(snapshot: CorpusSnapshot) -> SparseIndex:
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
        return SparseIndex(chunk_ids=chunk_ids, bm25=None)
    return SparseIndex(chunk_ids=chunk_ids, bm25=BM25Okapi(tokenized))


def rank_ids(query: str, index: SparseIndex, limit: int) -> list[str]:
    """Return BM25's ranked candidate ids for `query` (empty when disabled).

    Args:
        query: The natural-language query to score the corpus against.
        index: The BM25 index to score with; a ``None`` model means the sparse
            side is disabled for this corpus and no ids are returned.
        limit: Maximum number of candidate ids to return.

    Only chunks with a positive BM25 score are returned. A zero score means
    the query shares no term with the chunk; feeding those into the fusion
    would let arbitrary non-matches inherit a rank, and with a corpus
    smaller than ``limit`` that is every chunk in the collection.

    Note:
        BM25Okapi's IDF is ``log(N - df + 0.5) - log(df + 0.5)``, which is zero
        for a term appearing in exactly half the corpus and negative above that.
        On a two-chunk collection a perfectly discriminating term therefore
        scores 0 and is dropped here — the sparse side goes quiet on corpora too
        small for term statistics to mean anything, which is the right thing for
        it to do.

    TRADE-OFF: scoring sorts the whole corpus, so this is O(N log N) per query
        in the number of chunks. Fine at the scale a single-worker ChromaDB
        deployment serves; a corpus large enough to feel it wants a real
        inverted index (Elasticsearch, Vespa) rather than rank-bm25.
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
    )[:limit]
    return [index.chunk_ids[i] for i in ranked]
