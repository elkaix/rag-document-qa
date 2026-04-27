"""
Generation metrics — answer correctness, context recall, and thin
wrappers around src.evaluation.* LLM-as-judge functions for batch use.

Eval Harness Position:
  Generator → answer  ─┐
  Retriever → context ─┼─→ [GENERATION METRICS] → scores + details
  GoldSet  → gold_*  ─┘                            (per-question metrics)

Design decisions:
  - REUSE the existing src/evaluation.py judges rather than duplicate
    their prompts; this module exposes them in a metric-shaped signature
    that returns (score, details_dict) for the runner to log.
  - Answer correctness is the MEAN of two sub-scores: embedding cosine
    and LLM-judge factual match. Either alone is unreliable — cosine
    rewards lexical overlap, judge rewards meaning. Combining is more
    robust without being statistically fancy.
  - Embedder is the same all-MiniLM-L6-v2 ChromaDB ships with — zero
    new dependency.
  - context_recall is computed without an LLM call: it's a set-overlap
    ratio over chunk IDs. Cheap, deterministic, defendable.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.evaluation import (
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
)

# PATTERN: Lazy global — the SentenceTransformer model is ~80 MB on disk and
# takes ~0.5 s to load. We defer loading until the first call so importing
# this module doesn't pay the cost when only other metrics are needed.
_EMBEDDER: SentenceTransformer | None = None


def _embed(text: str) -> np.ndarray:
    """Encode a string to a 384-dim vector using all-MiniLM-L6-v2.

    Args:
        text: Input string to encode.

    Returns:
        Float64 numpy array of shape (384,).
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        # WHY: ChromaDB ships this model, so it's already a transitive dep.
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(_EMBEDDER.encode(text), dtype=float)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, returning 0.0 for zero-norm inputs.

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector has zero norm.
    """
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0 or nv == 0:  # DEFENSIVE: zero-norm → treat as orthogonal
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def context_recall(
    gold_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> float:
    """Compute recall of gold chunk IDs against retrieved chunk IDs.

    Recall = |gold ∩ retrieved| / |gold|. Returns NaN when gold is empty
    because recall is undefined with no positive examples.

    WHY set-overlap instead of LLM: recall over known IDs is deterministic
    and cheap. It answers "did we find what we needed?" without LLM cost.

    Args:
        gold_chunk_ids: Ground-truth chunk IDs that should be retrieved.
        retrieved_chunk_ids: Chunk IDs returned by the retriever (full set,
            not truncated by top-k before calling this function).

    Returns:
        Recall in [0.0, 1.0], or float('nan') if gold_chunk_ids is empty.
    """
    if not gold_chunk_ids:
        return float("nan")
    gold_set = set(gold_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)
    return len(gold_set & retrieved_set) / len(gold_set)


def _judge_factual_match(
    generated: str,
    gold: str,
    llm: Any,
) -> tuple[float, str]:
    """Ask the LLM to score factual agreement between generated and gold answers.

    Instructs the LLM to return strict JSON: {"factual_match": float, "reasoning": str}.
    Score is clamped to [0, 1]. On JSON parse failure, returns (0.0, default message).

    Args:
        generated: The answer produced by the RAG pipeline.
        gold: The reference (ground-truth) answer.
        llm: LLM handler with generate(prompt, system_prompt=) method.

    Returns:
        Tuple of (factual_match_score, reasoning_string).
    """
    system_prompt = (
        "You are a factual accuracy evaluator. "
        "Compare the generated answer to the gold answer and score factual agreement. "
        "Use 1.0 for full factual match, 0.5 for partial match, 0.0 for no match. "
        "Respond ONLY with valid JSON — no prose, no code fences."
    )
    user_prompt = (
        f"Generated answer: {generated}\n\n"
        f"Gold answer: {gold}\n\n"
        'Return JSON with exactly: {"factual_match": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
    )

    raw = llm.generate(user_prompt, system_prompt=system_prompt)

    # DEFENSIVE: strip markdown fences before parsing, identical to evaluation.py
    stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    stripped = re.sub(r"\s*```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
        score = max(0.0, min(1.0, float(parsed["factual_match"])))
        reasoning = str(parsed.get("reasoning", ""))
        return score, reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        return (0.0, "Judge returned malformed JSON; defaulting to 0.0")


def answer_correctness(
    generated: str,
    gold: str,
    llm: Any,
) -> tuple[float, dict]:
    """Score answer correctness as the mean of embedding cosine and LLM factual match.

    TRADE-OFF: cosine similarity captures lexical/semantic overlap but can be
    fooled by paraphrasing or topic drift. The LLM judge captures factual
    agreement but is noisy. Averaging both sub-scores is more robust than
    either alone without adding statistical complexity.

    Args:
        generated: The answer produced by the RAG pipeline.
        gold: The reference (ground-truth) answer.
        llm: LLM handler with generate(prompt, system_prompt=) method.

    Returns:
        Tuple of (combined_score, details_dict). combined_score is in [0, 1].
        details_dict contains "cosine", "judge_factual_match", "judge_reasoning".
    """
    cos = _cosine(_embed(generated), _embed(gold))
    judge_score, judge_reasoning = _judge_factual_match(generated, gold, llm)
    combined = (cos + judge_score) / 2.0
    details: dict = {
        "cosine": cos,
        "judge_factual_match": judge_score,
        "judge_reasoning": judge_reasoning,
    }
    return combined, details


def judge_faithfulness(
    answer: str,
    contexts: list[str],
    llm: Any,
) -> tuple[float, dict]:
    """Wrap evaluate_faithfulness for batch-mode use in the eval harness.

    Returns (score, details_dict) instead of the raw (score, reasoning, details_json)
    tuple so the runner can log a uniform structure across all metrics.

    Args:
        answer: The answer generated by the RAG pipeline.
        contexts: Retrieved text chunks used to generate the answer.
        llm: LLM handler with generate(prompt, system_prompt=) method.

    Returns:
        Tuple of (faithfulness_score, details_dict).
        details_dict is parsed from the judge's JSON output, plus "reasoning".
    """
    score, reasoning, details_json = evaluate_faithfulness(answer, contexts, llm)
    details: dict = json.loads(details_json) if details_json else {}
    details["reasoning"] = reasoning
    return score, details


def judge_answer_relevancy(
    question: str,
    answer: str,
    llm: Any,
) -> tuple[float, dict]:
    """Wrap evaluate_answer_relevancy for batch-mode use in the eval harness.

    Args:
        question: The original user question.
        answer: The answer generated by the RAG pipeline.
        llm: LLM handler with generate(prompt, system_prompt=) method.

    Returns:
        Tuple of (relevancy_score, details_dict).
        details_dict contains "reasoning".
    """
    # WHY only two-tuple: evaluate_answer_relevancy has no details_json — it
    # returns just (score, reasoning). Wrap into a dict for uniform structure.
    score, reasoning = evaluate_answer_relevancy(question, answer, llm)
    details: dict = {"reasoning": reasoning}
    return score, details


def judge_context_precision(
    question: str,
    contexts: list[str],
    llm: Any,
) -> tuple[float, dict]:
    """Wrap evaluate_context_precision for batch-mode use in the eval harness.

    Args:
        question: The original user question.
        contexts: Retrieved text chunks (ordered by retrieval rank).
        llm: LLM handler with generate(prompt, system_prompt=) method.

    Returns:
        Tuple of (precision_score, details_dict).
        details_dict is parsed from the judge's JSON output, plus "reasoning".
    """
    score, reasoning, details_json = evaluate_context_precision(question, contexts, llm)
    details: dict = json.loads(details_json) if details_json else {}
    details["reasoning"] = reasoning
    return score, details
