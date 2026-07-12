"""Tests for CrossEncoderReranker — re-scores candidates with a cross-encoder model."""

from __future__ import annotations

import pytest

from src.vector_store import SearchResult


def _sr(chunk_id: str, content: str, score: float, metadata: dict | None = None) -> SearchResult:
    return SearchResult(doc_id="", chunk_id=chunk_id, content=content,
                        score=score, metadata=metadata or {})


@pytest.fixture(scope="module")
def reranker():
    from src.retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


def test_obvious_match_ranks_first(reranker):
    """Given five candidates with one obviously-relevant doc, it ranks first after rerank."""
    candidates = [
        _sr("d1", "Pyramids of Giza were built around 2500 BC.", 0.5),
        _sr("d2", "Cats are small carnivorous mammals.", 0.6),
        _sr("d3", "What is the capital of France? Paris is the capital.", 0.4),
        _sr("d4", "Airplanes have fixed wings.", 0.3),
        _sr("d5", "Dogs are domesticated.", 0.2),
    ]
    out = reranker.rerank("What is the capital of France?", candidates, final_top_k=3)
    assert len(out) == 3
    assert out[0].chunk_id == "d3"


def test_rerank_preserves_search_result_shape(reranker):
    candidates = [
        _sr("d1", "hello", 0.5, metadata={"k": "v"}),
        _sr("d2", "world", 0.4),
    ]
    out = reranker.rerank("greeting", candidates, final_top_k=2)
    assert all(isinstance(r, SearchResult) for r in out)
    found = {r.chunk_id: r for r in out}
    assert found["d1"].metadata == {"k": "v"}
