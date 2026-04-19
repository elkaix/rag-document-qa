"""
MessageEvaluation SQLModel table -- LLM-as-judge scores for RAG answers.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                                                      |
                                                              [MESSAGE] -> [EVALUATION]

What concept it teaches:
  LLM-as-judge evaluation -- using a separate LLM to score the quality of
  generated answers across multiple dimensions (faithfulness, relevancy,
  context precision).

Why this approach over alternatives:
  A dedicated table (vs. a JSON column on Message) keeps evaluation data
  normalized and queryable -- e.g. "show all low-faithfulness answers" or
  "average scores over time" are simple SQL queries.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class MessageEvaluation(SQLModel, table=True):
    """One evaluation metric score for an assistant message.

    Each row stores a single metric (faithfulness, answer_relevancy, or
    context_precision) with the judge's score, reasoning, and optional
    structured details (claim-level or chunk-level breakdown as JSON).
    """

    __tablename__ = "message_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)

    message_id: str = Field(
        foreign_key="messages.id",
        ondelete="CASCADE",
        index=True,
    )

    metric: str

    score: float

    reasoning: str = Field(default="")

    details: Optional[str] = Field(default=None)

    judge_model: str

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
