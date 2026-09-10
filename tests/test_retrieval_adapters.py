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
import pytest

from src.domain import SearchResult
from src.retrieval import Retriever
from src.vector_store import ChromaVectorStore


def _sr(chunk_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, score=score, metadata={}, doc_id="")


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
    store = ChromaVectorStore.open(chromadb.EphemeralClient(), "test_dense")
    coll = store.collection
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

    retriever = BM25HybridRetriever(vector_store=_chroma_store())
    assert isinstance(retriever, Retriever)


# --------------------------------------------------------------------------- #
# Slice 3 — RerankingRetriever composes inner + reranker (over-fetch)         #
# --------------------------------------------------------------------------- #


class _StubLLM:
    """Minimal generate_with_usage stand-in — never called by these tests."""

    def generate_with_usage(self, prompt, system_prompt=None):
        return "[]", 0, 0


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
    return SearchResult(chunk_id=chunk_id, content=chunk_id, score=score, metadata={}, doc_id="")


def test_multi_query_retriever_conforms_to_protocol():
    from src.retrieval import MultiQueryRetriever

    adapter = MultiQueryRetriever(inner=_FakeRetriever({}), rewriter=_FakeRewriter(["q"]))
    assert isinstance(adapter, Retriever)


def test_multi_query_fans_out_and_fuses_the_rankings_by_rank():
    """Expansions are retrieved, then fused by RRF — consensus beats one high score."""
    from src.retrieval import MultiQueryRetriever

    inner = _FakeRetriever(
        {
            "q": [_scored("c1", 0.9), _scored("c2", 0.5)],
            "q2": [_scored("c3", 0.8), _scored("c2", 0.7)],
        }
    )
    adapter = MultiQueryRetriever(inner=inner, rewriter=_FakeRewriter(["q", "q2"]))

    out = adapter.retrieve("q", top_k=2)

    # Both expansions were retrieved.
    assert {c[0] for c in inner.calls} == {"q", "q2"}
    # c2 placed second under both phrasings, so its two 1/62 contributions beat
    # c1's and c3's single 1/61 — the point of fusing by rank rather than by a
    # score whose scale the seam never promised.
    assert [r.chunk_id for r in out] == ["c2", "c1"]
    # Each survivor still carries the best score it was seen with.
    assert out[0].score == 0.7


def test_multi_query_does_not_discard_results_that_score_zero():
    """The hybrid-under-multi-query case every eval config from phase2e stacks.

    BM25-only hits carry score 0.0 because no comparable cosine similarity
    exists for them. Ranking the union by score sorted exactly those to the
    bottom and truncated them away — deleting what hybrid was added to
    contribute.
    """
    from src.retrieval import MultiQueryRetriever

    inner = _FakeRetriever(
        {
            "q": [_scored("sparse-only", 0.0), _scored("dense-hit", 0.6)],
            "q2": [_scored("sparse-only", 0.0), _scored("other", 0.5)],
        }
    )
    adapter = MultiQueryRetriever(inner=inner, rewriter=_FakeRewriter(["q", "q2"]))

    out = adapter.retrieve("q", top_k=2)
    assert "sparse-only" in [r.chunk_id for r in out]


# --------------------------------------------------------------------------- #
# Factory — config strategy -> Retriever type                                 #
# --------------------------------------------------------------------------- #


def test_build_retriever_dense_is_the_default_strategy():
    from src.retrieval import DenseRetriever, build_retrieval_plan

    retriever = build_retrieval_plan("dense", _chroma_store()).retriever
    assert isinstance(retriever, DenseRetriever)


def test_build_retriever_reranked_composes_dense_and_a_reranker(monkeypatch):
    """reranked wires a RerankingRetriever without loading the real model here."""
    from src.retrieval import RerankingRetriever, build_retrieval_plan

    monkeypatch.setattr("src.retrieval.composition.CrossEncoderReranker", _FakeReranker)
    retriever = build_retrieval_plan("reranked", _chroma_store(), rerank_over_fetch_n=15).retriever
    assert isinstance(retriever, RerankingRetriever)


def test_build_retriever_hybrid_fuses_sparse_and_dense():
    """hybrid is a wired strategy, not a deferred name (ADR 0009)."""
    from src.retrieval import BM25HybridRetriever, build_retrieval_plan

    retriever = build_retrieval_plan("hybrid", _chroma_store()).retriever
    assert isinstance(retriever, BM25HybridRetriever)


def test_build_retriever_multi_query_composes_a_rewriter(monkeypatch):
    """multi_query is wired once a rewriter model is configured (ADR 0009)."""
    from src.retrieval import MultiQueryRetriever, build_retrieval_plan

    monkeypatch.setattr("src.retrieval.composition.QUERY_REWRITER_MODEL", "gpt-4.1-nano")
    retriever = build_retrieval_plan("multi_query", _chroma_store(), llm=_StubLLM()).retriever
    assert isinstance(retriever, MultiQueryRetriever)


@pytest.mark.parametrize("strategy", ["totally-bogus", "", "Dense"])
def test_build_retriever_rejects_unknown_strategies(strategy):
    from src.retrieval import build_retrieval_plan

    with pytest.raises(ValueError, match="Unknown retriever strategy"):
        build_retrieval_plan(strategy, _chroma_store())


def test_build_retriever_multi_query_without_a_model_fails_loudly(monkeypatch):
    """An unconfigured rewriter must not silently degrade to dense retrieval.

    QueryRewriter(model=None) is a legal pass-through, so the composed chain
    would run, expand nothing, and look exactly like dense — a deployment
    believing it had recall it did not have.
    """
    from src.retrieval import build_retrieval_plan

    monkeypatch.setattr("src.retrieval.composition.QUERY_REWRITER_MODEL", None)
    with pytest.raises(ValueError, match="QUERY_REWRITER_MODEL"):
        build_retrieval_plan("multi_query", _chroma_store(), llm=_StubLLM())


def test_build_retriever_multi_query_without_an_llm_fails_loudly(monkeypatch):
    from src.retrieval import build_retrieval_plan

    monkeypatch.setattr("src.retrieval.composition.QUERY_REWRITER_MODEL", "gpt-4.1-nano")
    with pytest.raises(ValueError, match="LLM handler"):
        build_retrieval_plan("multi_query", _chroma_store())


def test_reranker_over_fetches_at_least_the_requested_top_k():
    """The over-fetch was sized from over_fetch_n alone, capping large top_k.

    `POST /api/query` accepts top_k up to 50 while RERANK_OVER_FETCH_N is 20,
    so `top_k=30` under the reranked strategy silently returned 20. Over-fetching
    exists to give the cross-encoder more choice than the caller wants, never
    less.
    """
    from src.retrieval.reranker import RerankingRetriever

    asked: list[int] = []

    class _Inner:
        def retrieve(self, query, top_k=5):
            asked.append(top_k)
            return [
                SearchResult(chunk_id=f"c{i}", content="t", score=1.0, metadata={}, doc_id="d")
                for i in range(top_k)
            ]

    class _PassThrough:
        def rerank(self, query, candidates, final_top_k):
            return candidates[:final_top_k]

    out = RerankingRetriever(inner=_Inner(), reranker=_PassThrough(), over_fetch_n=20).retrieve(
        "q", top_k=30
    )

    assert asked == [30]
    assert len(out) == 30
