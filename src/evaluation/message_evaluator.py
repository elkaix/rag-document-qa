"""Scoring a persisted message — load, judge what is unscored, persist.

RAG Pipeline Position:
    Answer -> persisted Message -> [MESSAGE EVALUATOR] -> MessageEvaluation rows
                                          ^^^
    The judges in ``judges`` are pure: text in, numbers out. This module is the
    orchestration around them — which message, which question it answered, what
    has already been scored, and where the result goes.

What concept it teaches:
    Separating *scoring* from *deciding what to score*. The judges are testable
    with strings alone; the skip/dedup decisions are testable with a database
    and no LLM.

Why this is its own module:
    These 154 code lines lived inside the 1265-line RAGBackend facade with no
    collaborator behind them, which is why their skip and dedup branches had no
    direct tests. Deleting the facade would not have removed the complexity — it
    would have reappeared in the route handler, where ``evaluate_message`` alone
    would have become the largest function in the API layer.

Design Decision:
    The session factory is injected rather than an engine, so this module and
    the conversation store share the facade's one session-per-operation policy
    instead of each inventing its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, col, select

from src.evaluation.judges import (
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
)
from src.models.evaluation import MessageEvaluation
from src.models.message import Message, MessageSource

logger = logging.getLogger(__name__)

FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"


@dataclass(frozen=True)
class Judges:
    """The three scoring functions, bundled so they can be substituted together.

    Attributes:
        faithfulness: ``(answer, contexts, llm) -> (score, reasoning, details)``
        answer_relevancy: ``(question, answer, llm) -> (score, reasoning)``
        context_precision: ``(question, contexts, llm) -> (score, reasoning, details)``

    WHY injected rather than imported and monkeypatched: substituting a judge
        used to mean reassigning a module global, so a test could only fake them
        by reaching into another module's namespace. Passing them in makes the
        substitution part of the interface.
    """

    faithfulness: Callable[..., tuple[float, str, str | None]] = evaluate_faithfulness
    answer_relevancy: Callable[..., tuple[float, str]] = evaluate_answer_relevancy
    context_precision: Callable[..., tuple[float, str, str | None]] = evaluate_context_precision


@dataclass(frozen=True)
class _MessageUnderTest:
    """What the judges need about one persisted assistant message."""

    answer: str
    question: str
    contexts: list[str]


class MessageEvaluator:
    """Score a persisted assistant message and store the results.

    Attributes are injected so the module can be tested with an in-memory
    database and a fake judge handler.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        judge_llm: Any,
        judges: Judges | None = None,
    ) -> None:
        """Wire the evaluator to its database, judge model, and scoring functions.

        Args:
            session_factory: Returns a fresh short-lived Session. Shared with
                the facade so the session-per-operation policy has one owner.
            judge_llm: The handler the judges call. Deliberately a separate
                model from the one that produced the answer — a model judging
                its own output is less likely to flag its own hallucinations.
            judges: The scoring functions. Defaults to the real ones; pass
                fakes to test the skip and dedup decisions without an LLM.
        """
        self._session = session_factory
        self._judge_llm = judge_llm
        self._judges = judges or Judges()

    # ------------------------------------------------------------------ #
    # Realtime path                                                       #
    # ------------------------------------------------------------------ #

    def score_realtime(self, message_id: str, answer: str, contexts: list[str]) -> dict:
        """Score faithfulness immediately after generation and persist it.

        Called at the end of a streamed answer, while the retrieved contexts are
        still in memory — scoring here avoids reloading MessageSource rows just
        to rebuild the context list.

        Args:
            message_id: The assistant message the score belongs to.
            answer: The full generated answer.
            contexts: The retrieved excerpts the answer should be grounded in.

        Returns:
            A dict with metric, score and reasoning. On failure, a zero-score
            sentinel, so a caller can always read ``["score"]``.

        PATTERN: Fail-safe. Evaluation is a non-critical path; a judge timeout
            or malformed JSON must never break the stream that already delivered
            the answer to the user.
        """
        try:
            score, reasoning, details = self._judges.faithfulness(answer, contexts, self._judge_llm)
            self._persist(message_id, FAITHFULNESS, score, reasoning, details)
            logger.info("Faithfulness score for message %s: %.3f", message_id, score)
            return {"metric": FAITHFULNESS, "score": score, "reasoning": reasoning}
        except Exception as exc:
            logger.error("score_realtime failed for message %s: %s", message_id, exc)
            return {"metric": FAITHFULNESS, "score": 0.0, "reasoning": str(exc)}

    # ------------------------------------------------------------------ #
    # On-demand path                                                      #
    # ------------------------------------------------------------------ #

    def score_message(self, message_id: str) -> list[dict]:
        """Score every metric not already recorded for a message.

        Args:
            message_id: The assistant message to evaluate.

        Returns:
            One dict per metric, whether freshly scored or read back from an
            earlier scoring. Empty when the message does not exist.

        WHY each metric is skipped when present: the realtime path may already
            have scored faithfulness. Re-running it would write a second row and
            skew any aggregation over the table.
        """
        loaded = self._load(message_id)
        if loaded is None:
            logger.warning("score_message: message %s not found", message_id)
            return []

        return [
            entry
            for entry in (
                self._faithfulness(message_id, loaded),
                self._answer_relevancy(message_id, loaded),
                self._context_precision(message_id, loaded),
            )
            if entry is not None
        ]

    def scores_for(self, message_id: str) -> list[dict]:
        """Return every stored score for a message without calling a judge.

        Args:
            message_id: The assistant message to read.

        Returns:
            One dict per stored metric; empty when nothing has been scored.
        """
        with self._session() as session:
            rows = session.exec(
                select(MessageEvaluation).where(MessageEvaluation.message_id == message_id)
            ).all()
            return [
                {
                    "metric": row.metric,
                    "score": row.score,
                    "reasoning": row.reasoning,
                    "details": row.details,
                    "judge_model": row.judge_model,
                    "evaluated_at": row.evaluated_at.isoformat(),
                }
                for row in rows
            ]

    # ------------------------------------------------------------------ #
    # Per-metric steps                                                    #
    # ------------------------------------------------------------------ #

    def _faithfulness(self, message_id: str, msg: _MessageUnderTest) -> dict | None:
        existing = self._existing(message_id, FAITHFULNESS)
        if existing is not None:
            # WHY details here and not on the other two: the frontend renders a
            #     claim-level breakdown for faithfulness, and it must look the
            #     same whether the score came from this call or the realtime one.
            return {
                "metric": FAITHFULNESS,
                "score": existing.score,
                "reasoning": existing.reasoning,
                "details": existing.details,
            }
        if not msg.contexts:
            return None
        score, reasoning, details = self._judges.faithfulness(
            msg.answer, msg.contexts, self._judge_llm
        )
        self._persist(message_id, FAITHFULNESS, score, reasoning, details)
        return {
            "metric": FAITHFULNESS,
            "score": score,
            "reasoning": reasoning,
            "details": details,
        }

    def _answer_relevancy(self, message_id: str, msg: _MessageUnderTest) -> dict | None:
        existing = self._existing(message_id, ANSWER_RELEVANCY)
        if existing is not None:
            return {
                "metric": ANSWER_RELEVANCY,
                "score": existing.score,
                "reasoning": existing.reasoning,
            }
        if not msg.question:
            return None
        score, reasoning = self._judges.answer_relevancy(msg.question, msg.answer, self._judge_llm)
        self._persist(message_id, ANSWER_RELEVANCY, score, reasoning, None)
        return {"metric": ANSWER_RELEVANCY, "score": score, "reasoning": reasoning}

    def _context_precision(self, message_id: str, msg: _MessageUnderTest) -> dict | None:
        existing = self._existing(message_id, CONTEXT_PRECISION)
        if existing is not None:
            return {
                "metric": CONTEXT_PRECISION,
                "score": existing.score,
                "reasoning": existing.reasoning,
            }
        if not (msg.question and msg.contexts):
            return None
        score, reasoning, details = self._judges.context_precision(
            msg.question, msg.contexts, self._judge_llm
        )
        self._persist(message_id, CONTEXT_PRECISION, score, reasoning, details)
        return {"metric": CONTEXT_PRECISION, "score": score, "reasoning": reasoning}

    # ------------------------------------------------------------------ #
    # Database helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load(self, message_id: str) -> _MessageUnderTest | None:
        """Gather the answer, its retrieved contexts, and the question it answered."""
        with self._session() as session:
            msg = session.get(Message, message_id)
            if msg is None:
                return None

            sources = session.exec(
                select(MessageSource).where(MessageSource.message_id == message_id)
            ).all()

            # WHY the closest earlier user message: in a linear thread it is the
            #     question this answer responded to. Ordering desc + first picks
            #     it without needing an explicit parent link.
            user_msg = session.exec(
                select(Message)
                .where(
                    Message.conversation_id == msg.conversation_id,
                    Message.role == "user",
                    Message.created_at < msg.created_at,
                )
                .order_by(col(Message.created_at).desc())
            ).first()

            return _MessageUnderTest(
                answer=msg.content,
                question=user_msg.content if user_msg else "",
                contexts=[s.excerpt for s in sources if s.excerpt],
            )

    def _existing(self, message_id: str, metric: str) -> MessageEvaluation | None:
        with self._session() as session:
            return session.exec(
                select(MessageEvaluation).where(
                    MessageEvaluation.message_id == message_id,
                    MessageEvaluation.metric == metric,
                )
            ).first()

    def _persist(
        self,
        message_id: str,
        metric: str,
        score: float,
        reasoning: str,
        details: str | None,
    ) -> None:
        with self._session() as session:
            session.add(
                MessageEvaluation(
                    message_id=message_id,
                    metric=metric,
                    score=score,
                    reasoning=reasoning,
                    details=details,
                    judge_model=self._judge_llm.model,
                )
            )
            session.commit()
