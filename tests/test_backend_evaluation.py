"""Characterization tests for the backend's evaluation cluster.

RAG Pipeline Position:
    answer -> [EVALUATION] -> per-metric scores persisted against the message

These pin the behaviour of evaluate_faithfulness_realtime / evaluate_message /
get_evaluation before that cluster is extracted from the facade. The cluster is
154 code lines and, per the 2026-09-09 review, had no direct tests — its
skip/dedup branches were only ever exercised incidentally.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import chromadb
import pytest
from sqlmodel import Session, select

from src.backend import RAGBackend
from src.database import create_db_and_tables, get_engine
from src.evaluation import MessageEvaluator
from src.evaluation.message_evaluator import Judges
from src.models.evaluation import MessageEvaluation
from src.models.message import Message, MessageSource
from src.vector_store import ChromaVectorStore


@pytest.fixture
def backend() -> RAGBackend:
    engine = get_engine("sqlite://")
    create_db_and_tables(engine)
    collection = ChromaVectorStore.open(
        chromadb.EphemeralClient(), f"test_eval_{uuid.uuid4().hex}"
    ).collection
    return RAGBackend(engine=engine, collection=collection)


def _seed_turn(backend: RAGBackend, *, with_sources: bool = True) -> str:
    """Persist a user->assistant turn and return the assistant message id."""
    conv_id = backend.create_conversation("Eval fixture")["id"]
    base = datetime.now(UTC)

    with Session(backend.engine) as session:
        user = Message(
            conversation_id=conv_id, role="user", content="What is RAG?",
            created_at=base,
        )
        assistant = Message(
            conversation_id=conv_id, role="assistant",
            content="RAG retrieves then generates.",
            created_at=base + timedelta(seconds=1),
        )
        session.add(user)
        session.add(assistant)
        assistant_id = assistant.id
        if with_sources:
            session.add(
                MessageSource(
                    message_id=assistant_id, doc_id="d1", chunk_id="c1",
                    filename="rag.txt", score=0.9,
                    excerpt="RAG combines retrieval with generation.",
                )
            )
        session.commit()
    return assistant_id


def _fake_judges(calls: list[str], **overrides) -> Judges:
    """Judges that record what was called and never touch a provider.

    Injected through the evaluator's constructor rather than monkeypatched onto
    a module — substituting a judge is part of the interface now.
    """
    def faithfulness(answer, contexts, llm):
        calls.append("faithfulness")
        return 1.0, "supported", '{"claims": []}'

    def relevancy(question, answer, llm):
        calls.append("answer_relevancy")
        return 0.8, "on topic"

    def precision(question, contexts, llm):
        calls.append("context_precision")
        return 0.7, "useful", None

    return Judges(
        faithfulness=overrides.get("faithfulness", faithfulness),
        answer_relevancy=overrides.get("answer_relevancy", relevancy),
        context_precision=overrides.get("context_precision", precision),
    )


def _with_judges(backend: RAGBackend, judges: Judges) -> RAGBackend:
    """Point the facade's evaluator at substitute judges."""
    backend.evaluator = MessageEvaluator(
        session_factory=backend._session,
        judge_llm=backend.eval_llm,
        judges=judges,
    )
    return backend


class TestRealtimeFaithfulness:
    def test_persists_a_score_against_the_message(self, backend):
        _with_judges(backend, _fake_judges([]))
        message_id = _seed_turn(backend)

        result = backend.evaluate_faithfulness_realtime(
            message_id, "RAG retrieves then generates.", ["RAG combines retrieval."]
        )

        assert result["metric"] == "faithfulness"
        assert result["score"] == 1.0
        assert len(backend.get_evaluation(message_id)) == 1

    def test_a_judge_failure_never_reaches_the_caller(self, backend):
        """The streaming endpoint calls this; an exception would kill the stream."""
        def boom(*a, **kw):
            raise RuntimeError("judge timeout")

        _with_judges(backend, _fake_judges([], faithfulness=boom))
        message_id = _seed_turn(backend)

        result = backend.evaluate_faithfulness_realtime(message_id, "answer", ["ctx"])

        assert result == {
            "metric": "faithfulness", "score": 0.0, "reasoning": "judge timeout",
        }
        assert backend.get_evaluation(message_id) == []


class TestEvaluateMessage:
    def test_unknown_message_returns_empty(self, backend):
        assert backend.evaluate_message("does-not-exist") == []

    def test_scores_all_three_metrics(self, backend):
        calls: list[str] = []
        _with_judges(backend, _fake_judges(calls))
        message_id = _seed_turn(backend)

        results = backend.evaluate_message(message_id)

        assert {r["metric"] for r in results} == {
            "faithfulness", "answer_relevancy", "context_precision",
        }
        assert sorted(calls) == ["answer_relevancy", "context_precision", "faithfulness"]

    def test_skips_faithfulness_when_realtime_already_scored_it(
        self, backend
    ):
        """Re-running would duplicate the row and skew aggregations."""
        calls: list[str] = []
        _with_judges(backend, _fake_judges(calls))
        message_id = _seed_turn(backend)

        backend.evaluate_faithfulness_realtime(message_id, "answer", ["ctx"])
        calls.clear()

        backend.evaluate_message(message_id)

        assert "faithfulness" not in calls
        with Session(backend.engine) as session:
            rows = session.exec(
                select(MessageEvaluation).where(
                    MessageEvaluation.message_id == message_id,
                    MessageEvaluation.metric == "faithfulness",
                )
            ).all()
        assert len(rows) == 1, "faithfulness must not be scored twice"

    def test_faithfulness_is_skipped_when_there_are_no_contexts(
        self, backend
    ):
        calls: list[str] = []
        _with_judges(backend, _fake_judges(calls))
        message_id = _seed_turn(backend, with_sources=False)

        backend.evaluate_message(message_id)

        assert "faithfulness" not in calls

    def test_uses_the_preceding_user_message_as_the_question(self, backend):
        seen: dict[str, str] = {}

        def relevancy(question, answer, llm):
            seen["question"] = question
            return 0.8, ""

        _with_judges(backend, _fake_judges([], answer_relevancy=relevancy))
        message_id = _seed_turn(backend)

        backend.evaluate_message(message_id)

        assert seen["question"] == "What is RAG?"


class TestGetEvaluation:
    def test_returns_every_persisted_metric(self, backend):
        _with_judges(backend, _fake_judges([]))
        message_id = _seed_turn(backend)
        backend.evaluate_message(message_id)

        rows = backend.get_evaluation(message_id)

        assert {r["metric"] for r in rows} == {
            "faithfulness", "answer_relevancy", "context_precision",
        }

    def test_unknown_message_returns_empty(self, backend):
        assert backend.get_evaluation("nope") == []


class TestJudgeInjection:
    """Substituting a judge is part of the interface, not a module patch."""

    def test_default_judges_are_the_real_ones(self):
        from src.evaluation import judges as judge_module

        defaults = Judges()
        assert defaults.faithfulness is judge_module.evaluate_faithfulness
        assert defaults.answer_relevancy is judge_module.evaluate_answer_relevancy
        assert defaults.context_precision is judge_module.evaluate_context_precision

    def test_a_single_judge_can_be_replaced(self, backend):
        calls: list[str] = []

        def only_this(question, answer, llm):
            calls.append("replaced")
            return 0.1, "stub"

        _with_judges(backend, _fake_judges([], answer_relevancy=only_this))
        backend.evaluate_message(_seed_turn(backend))

        assert calls == ["replaced"]
