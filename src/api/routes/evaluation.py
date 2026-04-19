"""
Evaluation routes: POST and GET for on-demand RAG answer evaluation.

RAG Pipeline Position:
  Document → Chunks → Embeddings → Vector Store → Retrieval → Generator → Answer
                                                                              |
                                                              [EVALUATION ROUTES]

What concept it teaches:
  On-demand evaluation endpoints — the user clicks "Evaluate" and the
  backend runs all three metrics (faithfulness, answer_relevancy,
  context_precision) against the stored message and its sources.

Why this approach over alternatives:
  Real-time evaluation (during streaming) only runs faithfulness because
  it's fast and critical. The full 3-metric evaluation is triggered on-demand
  via these REST endpoints so that it doesn't slow down the answer stream.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["evaluation"])


@router.post(
    "/messages/{message_id}/evaluate",
    summary="Run full evaluation on a message",
)
async def evaluate_message(message_id: str, request: Request):
    """Trigger all three evaluation metrics for a stored assistant message.

    Runs faithfulness (if not already scored), answer_relevancy, and
    context_precision. Results are persisted to the MessageEvaluation table.

    Returns:
        List of score dicts, or 404 if message not found / no results.
    """
    backend = request.app.state.backend
    results = backend.evaluate_message(message_id)
    if not results:
        raise HTTPException(status_code=404, detail="Message not found or evaluation failed.")
    return results


@router.get(
    "/messages/{message_id}/evaluation",
    summary="Get existing evaluation scores for a message",
)
async def get_evaluation(message_id: str, request: Request):
    """Retrieve previously computed evaluation scores for a message.

    Returns:
        List of score dicts (may be empty if not yet evaluated).
    """
    backend = request.app.state.backend
    return backend.get_evaluation(message_id)
