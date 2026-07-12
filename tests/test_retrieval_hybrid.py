"""Tests for BM25HybridRetriever — RRF fusion of BM25 (sparse) + dense (Chroma)."""

from __future__ import annotations

from src.vector_store import SearchResult


def _sr(chunk_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, score=score, metadata={}, doc_id="")


def test_rrf_fusion_asymmetric_inputs():
    """RRF on A=[a,b,c,d], B=[d,a] with rrf_k=60 yields fused order a, d, b, c."""
    from src.retrieval.hybrid import reciprocal_rank_fusion
    A = ["a", "b", "c", "d"]
    B = ["d", "a"]
    fused = reciprocal_rank_fusion([A, B], rrf_k=60)
    assert fused == ["a", "d", "b", "c"]


def test_hybrid_retrieve_returns_top_k():
    """End-to-end: hybrid retriever combines BM25 and Chroma results into top-K."""
    import chromadb
    from src.retrieval.hybrid import BM25HybridRetriever
    from src.vector_store import ChromaVectorStore

    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="test_hybrid", metadata={"hnsw:space": "cosine"},
    )
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
    retriever = BM25HybridRetriever(
        vector_store=vs,
        documents={"d1": coll.get(ids=["d1"])["documents"][0],
                    "d2": coll.get(ids=["d2"])["documents"][0],
                    "d3": coll.get(ids=["d3"])["documents"][0],
                    "d4": coll.get(ids=["d4"])["documents"][0]},
        bm25_top_k=3,
        dense_top_k=3,
        rrf_k=60,
    )
    out = retriever.retrieve("hybrid sparse dense fusion", top_k=2)
    assert len(out) == 2
    assert all(isinstance(r, SearchResult) for r in out)
    # Top result should be one of d2 or d3 (both directly relevant).
    assert out[0].chunk_id in {"d2", "d3"}
