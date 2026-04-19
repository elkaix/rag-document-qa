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


import json
from unittest.mock import MagicMock

from src.evaluation import (
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
)


def _mock_llm(response_json: dict) -> MagicMock:
    """Create a mock LLMHandler that returns a fixed JSON string."""
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response_json)
    return llm


def test_evaluate_faithfulness_all_supported():
    llm = _mock_llm({
        "claims": [
            {"claim": "LoRA freezes weights", "supported": True, "evidence": "context says so"},
            {"claim": "LoRA uses low-rank matrices", "supported": True, "evidence": "mentioned"},
        ],
        "score": 1.0,
        "reasoning": "All claims supported.",
    })
    score, reasoning, details = evaluate_faithfulness(
        answer="LoRA freezes weights and uses low-rank matrices.",
        contexts=["LoRA freezes the original weights and adds low-rank matrices."],
        llm=llm,
    )
    assert score == 1.0
    assert details is not None
    parsed = json.loads(details)
    assert len(parsed["claims"]) == 2


def test_evaluate_faithfulness_partial():
    llm = _mock_llm({
        "claims": [
            {"claim": "LoRA freezes weights", "supported": True, "evidence": "yes"},
            {"claim": "LoRA was invented in 2025", "supported": False, "evidence": None},
        ],
        "score": 0.5,
        "reasoning": "One claim unsupported.",
    })
    score, reasoning, details = evaluate_faithfulness(
        answer="LoRA freezes weights. LoRA was invented in 2025.",
        contexts=["LoRA freezes the original weights."],
        llm=llm,
    )
    assert score == 0.5


def test_evaluate_faithfulness_malformed_json():
    llm = MagicMock()
    llm.generate.return_value = "This is not JSON at all"
    score, reasoning, details = evaluate_faithfulness(
        answer="test", contexts=["test"], llm=llm,
    )
    assert score == 0.0
    assert "failed" in reasoning.lower() or "error" in reasoning.lower()
    assert details is None


def test_evaluate_answer_relevancy():
    llm = _mock_llm({
        "score": 0.9,
        "reasoning": "Answer directly addresses the question.",
    })
    score, reasoning = evaluate_answer_relevancy(
        question="What is LoRA?",
        answer="LoRA is a fine-tuning technique.",
        llm=llm,
    )
    assert score == 0.9
    assert len(reasoning) > 0


def test_evaluate_context_precision():
    llm = _mock_llm({
        "chunks": [
            {"chunk_index": 0, "relevant": True},
            {"chunk_index": 1, "relevant": False},
        ],
        "score": 0.5,
        "reasoning": "Only one chunk was relevant.",
    })
    score, reasoning, details = evaluate_context_precision(
        question="What is LoRA?",
        contexts=["LoRA is about low-rank adaptation.", "The weather is nice today."],
        llm=llm,
    )
    assert score == 0.5
    parsed = json.loads(details)
    assert len(parsed["chunks"]) == 2
