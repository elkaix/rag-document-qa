"""Tests for BM25HybridRetriever — RRF fusion of BM25 (sparse) + dense (Chroma)."""

from __future__ import annotations

from src.domain import SearchResult


def _sr(chunk_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, score=score, metadata={}, doc_id="")


def test_rrf_fusion_asymmetric_inputs():
    """RRF on A=[a,b,c,d], B=[d,a] with rrf_k=60 yields fused order a, d, b, c."""
    from src.retrieval import reciprocal_rank_fusion

    A = ["a", "b", "c", "d"]
    B = ["d", "a"]
    fused = reciprocal_rank_fusion([A, B], rrf_k=60)
    assert fused == ["a", "d", "b", "c"]


def test_hybrid_retrieve_returns_top_k():
    """End-to-end: hybrid retriever combines BM25 and Chroma results into top-K."""
    import chromadb

    from src.retrieval.hybrid import BM25HybridRetriever
    from src.vector_store import ChromaVectorStore

    coll = ChromaVectorStore.open(chromadb.EphemeralClient(), "test_hybrid").collection
    coll.upsert(
        ids=["d1", "d2", "d3", "d4"],
        documents=[
            "Cats are small carnivorous mammals often kept as pets.",
            "Reciprocal rank fusion is a standard sparse-dense combination.",
            "Hybrid search blends BM25 and dense retrieval signals.",
            "Airplanes have fixed wings.",
        ],
    )
    vs = ChromaVectorStore(collection=coll)
    retriever = BM25HybridRetriever(vector_store=vs, bm25_top_k=3, dense_top_k=3, rrf_k=60)
    out = retriever.retrieve("hybrid sparse dense fusion", top_k=2)
    assert len(out) == 2
    assert all(isinstance(r, SearchResult) for r in out)
    # Top result should be one of d2 or d3 (both directly relevant).
    assert out[0].chunk_id in {"d2", "d3"}


# --------------------------------------------------------------------------- #
# Robustness — the behaviour that let this strategy ship (ADR 0009)           #
# --------------------------------------------------------------------------- #


class _FakeStore:
    """A vector store stand-in: the three members BM25HybridRetriever reads.

    WHY a fake and not an ephemeral Chroma collection: these tests pin which
    half of the retriever contributed each result, and that is only observable
    if the dense ranking is fixed rather than whatever an embedding model
    decides. The corpus/revision contract is exercised for real in
    tests/test_retrieval_corpus.py.
    """

    def __init__(self, chunks=None, dense=None) -> None:
        self._chunks = dict(chunks or {})
        self.dense = list(dense or [])
        self.revision = 0

    def all_chunks(self):
        return dict(self._chunks)

    def query(self, query_text=None, top_k=5):
        return self.dense[:top_k]

    def add(self, chunk) -> None:
        self._chunks[chunk.chunk_id] = chunk
        self.revision += 1

    def remove(self, chunk_id: str) -> None:
        del self._chunks[chunk_id]
        self.revision += 1


def _chunk(chunk_id: str, content: str, doc_id: str = "doc"):
    from src.domain import Chunk

    return Chunk(
        content=content,
        metadata={"doc_id": doc_id, "filename": f"{doc_id}.txt"},
        chunk_id=chunk_id,
        doc_id=doc_id,
    )


def _hybrid(store):
    from src.retrieval.hybrid import BM25HybridRetriever

    return BM25HybridRetriever(vector_store=store, bm25_top_k=10, dense_top_k=10)


def test_empty_collection_returns_no_results_instead_of_raising():
    """BM25Okapi divides by the corpus size; an empty index is a normal state.

    A fresh deployment serves queries before its first upload — that request
    must come back empty, not 500.
    """
    assert _hybrid(_FakeStore()).retrieve("anything", top_k=5) == []


def test_corpus_with_no_tokenizable_text_degrades_to_dense_only():
    """Whitespace-only chunks make BM25's average IDF a division by zero."""
    dense = [_sr("c1", "   ", 0.4)]
    store = _FakeStore(chunks={"c1": _chunk("c1", "   ")}, dense=dense)
    assert [r.chunk_id for r in _hybrid(store).retrieve("query", top_k=5)] == ["c1"]


def test_chunks_sharing_no_term_with_the_query_are_not_fused_in():
    """A zero BM25 score is a non-match, not a weak match.

    Without the filter every chunk in a corpus smaller than bm25_top_k inherits
    a sparse rank, so unrelated text is handed to the generator as context.
    """
    store = _FakeStore(
        chunks={
            "c1": _chunk("c1", "cats are carnivorous mammals"),
            "c2": _chunk("c2", "airplanes have fixed wings"),
            "c3": _chunk("c3", "bread is baked from flour"),
        }
    )
    out = _hybrid(store).retrieve("cats", top_k=5)
    assert [r.chunk_id for r in out] == ["c1"]


def test_a_sparse_only_hit_keeps_its_metadata_and_doc_id():
    """The citation fix: a BM25 hit the dense side missed still renders a source.

    ADR 0004 accepted degraded citations while the lever was off by default.
    Wiring it for production (ADR 0009) is exactly when that expires.
    """
    store = _FakeStore(
        chunks={
            "c1": _chunk("c1", "cats are carnivorous mammals", doc_id="animals"),
            "c2": _chunk("c2", "zygomorphic flowers are bilaterally symmetric", doc_id="botany"),
            "c3": _chunk("c3", "bread is baked from flour", doc_id="food"),
        },
        dense=[_sr("c1", "cats are carnivorous mammals", 0.71)],
    )
    out = _hybrid(store).retrieve("zygomorphic", top_k=5)

    sparse_only = next(r for r in out if r.chunk_id == "c2")
    assert sparse_only.doc_id == "botany"
    assert sparse_only.metadata["filename"] == "botany.txt"
    assert sparse_only.score == 0.0


def test_a_dense_hit_keeps_its_real_similarity_score():
    store = _FakeStore(
        chunks={"c1": _chunk("c1", "cats are carnivorous mammals")},
        dense=[_sr("c1", "cats are carnivorous mammals", 0.83)],
    )
    (only,) = _hybrid(store).retrieve("cats", top_k=5)
    assert only.score == 0.83


def test_documents_ingested_after_construction_reach_the_sparse_side():
    """The deferral ADR 0004 named: the corpus used to freeze at construction."""
    store = _FakeStore(
        chunks={
            "c1": _chunk("c1", "cats are carnivorous mammals"),
            "c3": _chunk("c3", "bread is baked from flour"),
        }
    )
    retriever = _hybrid(store)
    assert retriever.retrieve("zygomorphic", top_k=5) == []

    store.add(_chunk("c2", "zygomorphic flowers are bilaterally symmetric"))
    assert [r.chunk_id for r in retriever.retrieve("zygomorphic", top_k=5)] == ["c2"]


def test_deleted_documents_stop_being_retrievable():
    """The other half of the deferral: tombstones the sparse side kept serving."""
    store = _FakeStore(
        chunks={
            "c1": _chunk("c1", "cats are carnivorous mammals"),
            "c2": _chunk("c2", "zygomorphic flowers are bilaterally symmetric"),
            "c3": _chunk("c3", "bread is baked from flour"),
        }
    )
    retriever = _hybrid(store)
    assert [r.chunk_id for r in retriever.retrieve("zygomorphic", top_k=5)] == ["c2"]

    store.remove("c2")
    assert retriever.retrieve("zygomorphic", top_k=5) == []


def test_a_query_with_no_terms_still_returns_dense_results():
    store = _FakeStore(
        chunks={"c1": _chunk("c1", "cats are carnivorous mammals")},
        dense=[_sr("c1", "cats are carnivorous mammals", 0.5)],
    )
    assert [r.chunk_id for r in _hybrid(store).retrieve("   ", top_k=5)] == ["c1"]
