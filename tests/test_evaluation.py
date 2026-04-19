"""Tests for the RAG evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from sqlmodel import Session, SQLModel, select

from src.database import get_engine
from src.models.evaluation import MessageEvaluation
from src.models.message import Message
from src.models.conversation import Conversation


def _setup_db():
    """Create an in-memory DB with all tables."""
    engine = get_engine("sqlite://")
    import src.models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    return engine


def test_message_evaluation_crud():
    """MessageEvaluation can be created, read, and cascades on message delete."""
    engine = _setup_db()
    with Session(engine) as session:
        conv = Conversation(title="test")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        msg = Message(conversation_id=conv.id, role="assistant", content="test answer")
        session.add(msg)
        session.commit()
        session.refresh(msg)

        evaluation = MessageEvaluation(
            message_id=msg.id,
            metric="faithfulness",
            score=0.85,
            reasoning="All claims supported.",
            details='{"claims": []}',
            judge_model="gpt-4.1-mini",
        )
        session.add(evaluation)
        session.commit()

        result = session.exec(
            select(MessageEvaluation).where(MessageEvaluation.message_id == msg.id)
        ).first()
        assert result is not None
        assert result.metric == "faithfulness"
        assert result.score == 0.85
        assert result.judge_model == "gpt-4.1-mini"

    with Session(engine) as session:
        msg = session.exec(select(Message)).first()
        session.delete(msg)
        session.commit()

        evals = session.exec(select(MessageEvaluation)).all()
        assert len(evals) == 0
