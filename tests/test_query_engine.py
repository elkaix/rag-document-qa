"""Tests for QueryEngine — the shared retrieve->generate module (issue #16, step 4b).

RAG Pipeline Position:
    Query -> [QUERYENGINE: Retriever -> prompt -> LLM -> telemetry] -> Answer

What concept it teaches:
    One deep module owns retrieve->generate for BOTH the synchronous and the
    streaming path, so the answer prompt, context format, and telemetry
    assembly exist in exactly one place. These tests drive the engine with a
    fake Retriever and a fake LLM (its two injected seams) and assert the
    contract the RAGBackend facade and the eval harness both depend on —
    including that sync and streaming issue *identical* answer instructions.
"""

from __future__ import annotations

from src.llm_handler import Usage
from src.query_engine import QueryEngine
from src.query_engine.prompt import ANSWER_SYSTEM_PROMPT, NO_DOCUMENTS_ANSWER
from src.vector_store import SearchResult


# --------------------------------------------------------------------------- #
# Fakes at the two engine seams                                               #
# --------------------------------------------------------------------------- #

class _FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return list(self._results)[:top_k]


class _FakeLLM:
    """Records the (system, user) instructions it is handed; scripts its output."""

    def __init__(self, model: str = "fake-answer", answer: str = "hello world", p: int = 7, c: int = 3):
        self.model = model
        self._answer = answer
        self._p, self._c = p, c
        self.seen_system: list[str | None] = []
        self.seen_user: list[str] = []
        self.seen_messages: list[list[dict]] = []

    def generate_with_usage(self, prompt, system_prompt=None):
        self.seen_system.append(system_prompt)
        self.seen_user.append(prompt)
        return self._answer, self._p, self._c

    def stream_response(self, prompt, system_prompt=None):
        self.seen_system.append(system_prompt)
        self.seen_user.append(prompt)
        for tok in self._answer.split():
            yield tok
        yield Usage(prompt_tokens=self._p, completion_tokens=self._c)

    def stream_messages(self, messages):
        self.seen_messages.append(messages)
        for tok in self._answer.split():
            yield tok
        yield Usage(prompt_tokens=self._p, completion_tokens=self._c)


def _sr(chunk_id: str, content: str, score: float, filename: str = "doc.txt") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, content=content, score=score,
        metadata={"filename": filename, "chunk_index": 0}, doc_id="d1",
    )


def _engine(results, answer_llm=None, reasoning_llm=None, refusal=None, top_k=5):
    return QueryEngine(
        retriever=_FakeRetriever(results),
        llm=answer_llm or _FakeLLM(),
        reasoning_llm=reasoning_llm or _FakeLLM(model="fake-reason", answer="plan step"),
        top_k=top_k,
        refusal=refusal,
    )


# --------------------------------------------------------------------------- #
# Slice 1 — ask() happy path: retrieve -> prompt -> generate -> telemetry      #
# --------------------------------------------------------------------------- #

def test_ask_returns_results_answer_and_telemetry():
    llm = _FakeLLM(answer="Paris is the capital.", p=11, c=4)
    engine = _engine([_sr("c1", "Paris is the capital of France.", 0.9)], answer_llm=llm)

    results, answer, telemetry = engine.ask("What is the capital of France?")

    assert [r.chunk_id for r in results] == ["c1"]
    assert answer == "Paris is the capital."
    assert telemetry.prompt_tokens == 11
    assert telemetry.completion_tokens == 4
    assert telemetry.retrieve_ms >= 0.0
    assert telemetry.generate_ms >= 0.0
    assert telemetry.cost_usd >= 0.0


def test_ask_uses_the_markdown_answer_prompt_and_filename_context():
    llm = _FakeLLM()
    engine = _engine([_sr("c1", "Body text.", 0.8, filename="paper.pdf")], answer_llm=llm)

    engine.ask("Q?")

    # The single markdown answer prompt is the system instruction.
    assert llm.seen_system == [ANSWER_SYSTEM_PROMPT]
    # Context is filename-prefixed, not a bare join.
    assert "[paper.pdf] Body text." in llm.seen_user[0]
    assert llm.seen_user[0].endswith("Question: Q?\n\nAnswer:")


def test_ask_top_k_defaults_and_overrides():
    retriever_results = [_sr(f"c{i}", f"t{i}", 0.5) for i in range(10)]
    engine = _engine(retriever_results, top_k=5)

    engine.ask("q")
    engine.ask("q", top_k=3)

    # First call used the engine default (5); second used the override (3).
    assert engine._retriever.calls == [("q", 5), ("q", 3)]


# --------------------------------------------------------------------------- #
# Slice 2 — ask() no-documents and refusal gate skip generation               #
# --------------------------------------------------------------------------- #

def test_ask_with_no_documents_returns_zero_generation_telemetry():
    llm = _FakeLLM()
    engine = _engine([], answer_llm=llm)

    results, answer, telemetry = engine.ask("anything")

    assert results == []
    assert answer == NO_DOCUMENTS_ANSWER
    assert telemetry.generate_ms == 0.0
    assert telemetry.prompt_tokens == 0
    assert telemetry.cost_usd == 0.0
    # No LLM call was made.
    assert llm.seen_user == []


def test_ask_with_refusal_gate_short_circuits_before_generation():
    from src.retrieval import RefusalHandler

    llm = _FakeLLM()
    gate = RefusalHandler(enabled=True, similarity_threshold=0.5, no_answer_text="I don't know.")
    # Top score 0.3 < threshold 0.5 -> refuse.
    engine = _engine([_sr("c1", "weakly related", 0.3)], answer_llm=llm, refusal=gate)

    results, answer, telemetry = engine.ask("q")

    assert results == []
    assert answer == "I don't know."
    assert telemetry.generate_ms == 0.0
    assert telemetry.prompt_tokens == 0
    assert llm.seen_user == []  # generation skipped


def test_ask_refusal_gate_fires_on_empty_index_before_no_documents_notice():
    """An enabled gate treats an empty retrieval as unanswerable (refuse, not 'no docs')."""
    from src.retrieval import RefusalHandler

    gate = RefusalHandler(enabled=True, similarity_threshold=0.5, no_answer_text="Cannot answer.")
    engine = _engine([], refusal=gate)

    results, answer, telemetry = engine.ask("q")

    assert results == []
    assert answer == "Cannot answer."  # not NO_DOCUMENTS_ANSWER
    assert telemetry.prompt_tokens == 0


# --------------------------------------------------------------------------- #
# Slice 3 — ask_stream events + sync/stream instruction parity                 #
# --------------------------------------------------------------------------- #

def _event_types(events):
    return [t for t, _ in events]


def test_ask_stream_emits_status_reasoning_token_then_result():
    engine = _engine([_sr("c1", "body", 0.9)])
    events = list(engine.ask_stream("q"))

    types = _event_types(events)
    assert "status" in types
    assert "reasoning" in types
    assert "token" in types
    # The terminal event is the internal ("result", StreamResult).
    last_type, last_data = events[-1]
    assert last_type == "result"
    assert [r.chunk_id for r in last_data.results] == ["c1"]
    assert last_data.telemetry.completion_tokens == 3
    # reasoning precedes the first answer token.
    assert types.index("reasoning") < types.index("token")


def test_ask_stream_no_documents_yields_notice_and_empty_result():
    engine = _engine([])
    events = list(engine.ask_stream("q"))

    assert ("token", NO_DOCUMENTS_ANSWER) in events
    last_type, last_data = events[-1]
    assert last_type == "result"
    assert last_data.results == []
    assert last_data.telemetry.generate_ms == 0.0


def test_sync_and_streaming_issue_identical_answer_instructions():
    """The spec's headline guarantee: one prompt, regardless of path."""
    results = [_sr("c1", "shared body", 0.9)]
    sync_llm = _FakeLLM()
    stream_llm = _FakeLLM()

    _engine(results, answer_llm=sync_llm).ask("same question")
    list(_engine(results, answer_llm=stream_llm).ask_stream("same question"))

    # Same system prompt AND same user prompt across both paths.
    assert sync_llm.seen_system[0] == stream_llm.seen_system[0] == ANSWER_SYSTEM_PROMPT
    assert sync_llm.seen_user[0] == stream_llm.seen_user[0]


def test_ask_stream_with_history_uses_multi_turn_messages():
    stream_llm = _FakeLLM()
    engine = _engine([_sr("c1", "body", 0.9)], answer_llm=stream_llm)
    history = [
        {"role": "user", "content": "prior q"},
        {"role": "assistant", "content": "prior a"},
    ]

    list(engine.ask_stream("now q", history=history))

    messages = stream_llm.seen_messages[0]
    assert messages[0] == {"role": "system", "content": ANSWER_SYSTEM_PROMPT}
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Question: now q\n\nAnswer:")
