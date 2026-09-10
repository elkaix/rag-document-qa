"""Answer evaluation — LLM-as-judge scoring for generated answers.

RAG Pipeline Position:
    Query -> Retrieve -> Generate -> Answer -> [EVALUATION] -> scores

What this package holds:
    - ``judges``: the three metric functions (faithfulness, answer relevancy,
      context precision) plus the shared fenced-JSON parser. Pure scoring: they
      take text and an LLM handler and return numbers.
    - ``message_evaluator``: orchestration for a *persisted* message — load it
      and its sources, find the question it answered, score what has not been
      scored yet, and persist the results.

Why this is a package rather than a module:
    ``src/evaluation.py`` used to sit beside ``src/eval/`` with a near-identical
    name, shared by production (``RAGBackend``) and the harness
    (``src/eval/runner.py``). Every reader's first guess — "this is the old code
    the eval package replaced" — was wrong, and the harness importing "upward"
    out of its own package read like a layering violation even though it was
    not. Names are re-exported here so existing imports keep working.
"""

from src.evaluation.judges import (
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
    parse_json_response,
)
from src.evaluation.message_evaluator import MessageEvaluator

__all__ = [
    "MessageEvaluator",
    "evaluate_answer_relevancy",
    "evaluate_context_precision",
    "evaluate_faithfulness",
    "parse_json_response",
]
