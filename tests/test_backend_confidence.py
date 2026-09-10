"""Tests for the confidence number RAGBackend reports alongside an answer.

RAG Pipeline Position:
  Question -> Retrieve -> Generate -> (answer, sources, [CONFIDENCE])
                                                         ^^^^^^^^^^

`confidence` is the clamped mean of the three best retrieval similarity scores.
It is a user-facing number: the API validates it into `[0, 1]` and the frontend
renders it under every assistant turn.

What these tests pin:
    That the metric reads the three best *scores* and not the first three
    *positions*. The two agree only while the retriever returns results in
    descending-score order — a property the `Retriever` seam never promised and
    that `BM25HybridRetriever` deliberately does not have, because it orders by
    fused RRF rank and gives sparse-only hits a score of 0.0 (ADR 0009).

Why the QueryEngine is stubbed:
    The defect is arithmetic over a result list, and reproducing it through a
    real retriever would mean coaxing an embedding model into returning a
    specific out-of-order ranking. Substituting the engine states the input
    directly, so the test says what it means and cannot pass for the wrong
    reason.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from src.api.schemas.telemetry import StageTelemetry
from src.backend import RAGBackend
from src.database import create_db_and_tables, get_engine
from src.domain import SearchResult
from src.vector_store import ChromaVectorStore


@pytest.fixture
def backend():
    """A wired RAGBackend over disposable stores (mirrors tests/test_backend.py)."""
    engine = get_engine("sqlite://")
    create_db_and_tables(engine)
    collection = ChromaVectorStore.open(
        chromadb.EphemeralClient(), f"test_confidence_{uuid.uuid4().hex}"
    ).collection
    return RAGBackend(engine=engine, collection=collection)


class _StubEngine:
    """Returns a fixed result list, so the confidence arithmetic is the only variable."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def ask(self, question, top_k=None, model=None):
        return (
            self._results,
            "stub answer",
            StageTelemetry(
                retrieve_ms=0.0,
                generate_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            ),
        )


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content="text",
        score=score,
        metadata={"filename": "f.txt"},
        doc_id="d",
    )


class TestConfidenceReadsTheBestScores:
    """The metric must not depend on an ordering the Retriever seam never promised."""

    def test_zero_scored_hits_at_the_front_do_not_sink_confidence(self, backend):
        """The hybrid case: sparse-only hits lead the fused ranking at score 0.0.

        BUG: reading ``results[:3]`` averaged 0.0, 0.0 and 0.88 to 0.293 and
        reported that to the user, when the three best scores average 0.843.
        The default ``TOP_K_RESULTS`` is 5, so the slice discarding the two
        strongest chunks is the ordinary case, not a corner one.
        """
        backend.query_engine = _StubEngine(
            [
                _result("c1", 0.0),
                _result("c2", 0.0),
                _result("c3", 0.88),
                _result("c4", 0.85),
                _result("c5", 0.80),
            ]
        )

        result, _ = backend.query_with_telemetry("anything")

        assert result["confidence"] == pytest.approx(0.8433, abs=1e-4)

    def test_three_results_are_unaffected_whatever_their_order(self, backend):
        """Bounding the defect: at or below three results every score counts anyway."""
        backend.query_engine = _StubEngine(
            [_result("c1", 0.0), _result("c2", 0.85), _result("c3", 0.80)]
        )

        result, _ = backend.query_with_telemetry("anything")

        assert result["confidence"] == pytest.approx(0.55, abs=1e-4)

    def test_descending_input_is_unchanged(self, backend):
        """Dense, reranked and multi-query already sort by score — no behaviour delta."""
        backend.query_engine = _StubEngine(
            [_result("c1", 0.90), _result("c2", 0.60), _result("c3", 0.30)]
        )

        result, _ = backend.query_with_telemetry("anything")

        assert result["confidence"] == pytest.approx(0.60, abs=1e-4)

    def test_only_the_three_best_count_however_many_were_retrieved(self, backend):
        """A fourth, weaker chunk must not dilute the score — nor a fourth stronger one rank in late."""
        backend.query_engine = _StubEngine(
            [_result("c1", 0.10), _result("c2", 0.20), _result("c3", 1.00), _result("c4", 0.90)]
        )

        result, _ = backend.query_with_telemetry("anything")

        # The three best are 1.00, 0.90, 0.20 — position order would have given 0.43.
        assert result["confidence"] == pytest.approx(0.70, abs=1e-4)

    def test_no_results_is_zero_not_a_division_by_zero(self, backend):
        """A refusal or an empty index reports 0.0 confidence."""
        backend.query_engine = _StubEngine([])

        result, _ = backend.query_with_telemetry("anything")

        assert result["confidence"] == 0.0
