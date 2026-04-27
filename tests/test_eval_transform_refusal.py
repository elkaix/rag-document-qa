"""Tests for RefusalHandler — pure-logic similarity gate."""

from __future__ import annotations

from src.vector_store import SearchResult


def _sr(score: float, chunk_id: str = "d1") -> SearchResult:
    return SearchResult(doc_id="", chunk_id=chunk_id, content="x",
                        score=score, metadata={})


def test_refuses_when_top1_below_threshold():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.20), _sr(0.10)]) is True


def test_does_not_refuse_when_top1_above_threshold():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.50), _sr(0.10)]) is False


def test_refuses_on_empty_candidates():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([]) is True


def test_disabled_handler_never_refuses():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=False, similarity_threshold=0.35,
                        no_answer_text="I don't know.")
    assert h.should_refuse([_sr(0.0)]) is False
    assert h.should_refuse([]) is False


def test_refuse_response_returns_text_and_no_chunks():
    from src.eval.transforms import RefusalHandler
    h = RefusalHandler(enabled=True, similarity_threshold=0.35,
                        no_answer_text="I cannot answer.")
    chunks, answer = h.refuse_response()
    assert chunks == []
    assert answer == "I cannot answer."
