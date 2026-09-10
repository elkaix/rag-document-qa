"""Tests for RefusalHandler — pure-logic similarity gate."""

from __future__ import annotations

from src.domain import SearchResult


def _sr(score: float, chunk_id: str = "d1") -> SearchResult:
    return SearchResult(doc_id="", chunk_id=chunk_id, content="x", score=score, metadata={})


def test_refuses_when_top1_below_threshold():
    from src.retrieval import RefusalHandler

    h = RefusalHandler(enabled=True, similarity_threshold=0.35, no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.20), _sr(0.10)]) is True


def test_does_not_refuse_when_top1_above_threshold():
    from src.retrieval import RefusalHandler

    h = RefusalHandler(enabled=True, similarity_threshold=0.35, no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.50), _sr(0.10)]) is False


def test_refuses_on_empty_candidates():
    from src.retrieval import RefusalHandler

    h = RefusalHandler(enabled=True, similarity_threshold=0.35, no_answer_text="I don't know.")
    assert h.should_refuse([]) is True


def test_disabled_handler_never_refuses():
    from src.retrieval import RefusalHandler

    h = RefusalHandler(enabled=False, similarity_threshold=0.35, no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.0)]) is False
    assert h.should_refuse([]) is False


def test_refuse_response_returns_text_and_no_chunks():
    from src.retrieval import RefusalHandler

    h = RefusalHandler(enabled=True, similarity_threshold=0.35, no_answer_text="I cannot answer.")
    chunks, answer = h.refuse_response()
    assert chunks == []
    assert answer == "I cannot answer."


def test_the_gate_reads_the_best_score_not_the_first_position():
    """Hybrid retrieval orders by fused rank, so position 0 need not be the max.

    A BM25-only hit carries score 0.0 (no comparable dense similarity exists).
    Reading candidates[0].score would refuse a question the corpus answers well
    the moment the hybrid strategy is switched on.
    """
    from src.domain import SearchResult
    from src.retrieval import RefusalHandler

    def _r(chunk_id: str, score: float) -> SearchResult:
        return SearchResult(
            content=chunk_id, metadata={}, score=score, doc_id="d", chunk_id=chunk_id
        )

    gate = RefusalHandler(enabled=True, similarity_threshold=0.35, no_answer_text="no")
    fused_order = [_r("sparse-only", 0.0), _r("dense-hit", 0.8)]
    assert gate.should_refuse(fused_order) is False
    assert gate.should_refuse([_r("a", 0.1), _r("b", 0.2)]) is True
