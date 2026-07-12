"""Contract tests for the Retriever seam (issue #16, step 4a).

RAG Pipeline Position:
    Query -> [RETRIEVER] -> list[SearchResult] -> Generator

What concept it teaches:
    A `Retriever` Protocol lets dense, hybrid, reranked, and multi-query
    retrieval be interchangeable behind one interface — `retrieve(query, top_k)
    -> list[SearchResult]`. These tests assert every adapter honours that
    contract, so the QueryEngine (step 4b) can accept any of them by injection.

Why fakes for the composing adapters:
    RerankingRetriever and MultiQueryRetriever compose an *inner* Retriever plus
    an injected re-scorer/rewriter. Their behaviour under test is the
    composition wiring (over-fetch, delegate, dedup) — not the ML model inside
    the reranker or the LLM inside the rewriter. Faking those injected
    collaborators keeps the contract test deterministic and fast; the real
    CrossEncoderReranker / QueryRewriter have their own dedicated tests.
"""

from __future__ import annotations

import chromadb

from src.retrieval import Retriever
from src.vector_store import ChromaVectorStore, SearchResult


def _sr(chunk_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, content=content, score=score, metadata={}, doc_id=""
    )


class _FakeRetriever:
    """A Retriever that returns a scripted list and records the top_k asked for."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self._by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return list(self._by_query.get(query, []))[:top_k]


# --------------------------------------------------------------------------- #
# Slice 1 — Retriever Protocol + DenseRetriever                               #
# --------------------------------------------------------------------------- #

def _chroma_store() -> ChromaVectorStore:
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="test_dense", metadata={"hnsw:space": "cosine"}
    )
    coll.upsert(
        ids=["d1", "d2", "d3"],
        documents=[
            "Paris is the capital of France.",
            "Cats are small carnivorous mammals.",
            "Airplanes have fixed wings and jet engines.",
        ],
        metadatas=[{"filename": "geo.txt"}, {"filename": "animals.txt"}, {"filename": "air.txt"}],
    )
    return ChromaVectorStore(collection=coll)


def test_dense_retriever_conforms_to_protocol():
    """DenseRetriever satisfies the runtime-checkable Retriever Protocol."""
    from src.retrieval import DenseRetriever

    retriever = DenseRetriever(_chroma_store())
    assert isinstance(retriever, Retriever)


def test_dense_retriever_returns_search_results_from_store():
    """retrieve() delegates to the vector store and returns ranked SearchResults."""
    from src.retrieval import DenseRetriever

    retriever = DenseRetriever(_chroma_store())
    out = retriever.retrieve("What is the capital of France?", top_k=2)

    assert len(out) == 2
    assert all(isinstance(r, SearchResult) for r in out)
    # The geography chunk is the obvious top hit.
    assert out[0].chunk_id == "d1"
    assert out[0].metadata["filename"] == "geo.txt"


# --------------------------------------------------------------------------- #
# Slice 2 — BM25HybridRetriever conforms directly                             #
# --------------------------------------------------------------------------- #

def test_hybrid_retriever_conforms_to_protocol():
    """BM25HybridRetriever already exposes retrieve() — it conforms directly."""
    from src.retrieval import BM25HybridRetriever

    store = _chroma_store()
    retriever = BM25HybridRetriever(
        vector_store=store,
        documents={"d1": "Paris is the capital of France."},
    )
    assert isinstance(retriever, Retriever)


# --------------------------------------------------------------------------- #
# Slice 3 — RerankingRetriever composes inner + reranker (over-fetch)         #
# --------------------------------------------------------------------------- #

class _FakeReranker:
    """Records the candidates + final_top_k it received; reverses then truncates."""

    def __init__(self) -> None:
        self.seen_candidates: list[SearchResult] = []
        self.seen_final_top_k: int | None = None

    def rerank(self, query, candidates, final_top_k):
        self.seen_candidates = candidates
        self.seen_final_top_k = final_top_k
        return list(reversed(candidates))[:final_top_k]


def test_reranking_retriever_conforms_to_protocol():
    from src.retrieval import RerankingRetriever

    inner = _FakeRetriever({})
    adapter = RerankingRetriever(inner=inner, reranker=_FakeReranker(), over_fetch_n=20)
    assert isinstance(adapter, Retriever)


def test_reranking_retriever_over_fetches_then_reranks_to_top_k():
    """It fetches `over_fetch_n` from the inner retriever, then reranks to `top_k`."""
    from src.retrieval import RerankingRetriever

    candidates = [_sr(f"c{i}", f"text {i}", 0.5) for i in range(8)]
    inner = _FakeRetriever({"q": candidates})
    reranker = _FakeReranker()
    adapter = RerankingRetriever(inner=inner, reranker=reranker, over_fetch_n=8)

    out = adapter.retrieve("q", top_k=3)

    # Inner was asked for the wide candidate set, not top_k.
    assert inner.calls == [("q", 8)]
    # Reranker received those candidates and the final top_k.
    assert len(reranker.seen_candidates) == 8
    assert reranker.seen_final_top_k == 3
    # Output is the reranker's reordered, truncated result.
    assert [r.chunk_id for r in out] == ["c7", "c6", "c5"]


# --------------------------------------------------------------------------- #
# Slice 4 — MultiQueryRetriever fans out expansions, dedups                    #
# --------------------------------------------------------------------------- #

class _FakeRewriter:
    """Returns a scripted expansion list (the QueryRewriter.expand contract)."""

    def __init__(self, expansions: list[str]) -> None:
        self._expansions = expansions

    def expand(self, query: str) -> tuple[list[str], float, int, int]:
        return self._expansions, 0.0, 0, 0


def _scored(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, content=chunk_id, score=score, metadata={}, doc_id=""
    )


def test_multi_query_retriever_conforms_to_protocol():
    from src.retrieval import MultiQueryRetriever

    adapter = MultiQueryRetriever(inner=_FakeRetriever({}), rewriter=_FakeRewriter(["q"]))
    assert isinstance(adapter, Retriever)


def test_multi_query_fans_out_dedups_and_ranks_best_first():
    """Expansions are retrieved, deduped by chunk_id (keeping the best score), ranked."""
    from src.retrieval import MultiQueryRetriever

    inner = _FakeRetriever({
        "q": [_scored("c1", 0.9), _scored("c2", 0.5)],
        "q2": [_scored("c3", 0.8), _scored("c2", 0.7)],
    })
    adapter = MultiQueryRetriever(inner=inner, rewriter=_FakeRewriter(["q", "q2"]))

    out = adapter.retrieve("q", top_k=2)

    # Both expansions were retrieved.
    assert {c[0] for c in inner.calls} == {"q", "q2"}
    # c2 was deduped to its higher score (0.7), the union ranked best-first,
    # then truncated to top_k: c1(0.9), c3(0.8) win over c2(0.7).
    assert [r.chunk_id for r in out] == ["c1", "c3"]
