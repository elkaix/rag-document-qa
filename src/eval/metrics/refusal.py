"""
Refusal correctness — does the answer correctly refuse on unanswerable
questions and correctly attempt on answerable ones?

Eval Harness Position:
  Pipeline → answer + is_unanswerable → [REFUSAL] → 1.0 / 0.0
                                         ^^^^^^^
  Two-stage detection:
    1. Regex fast path on canonical refusal phrases.
    2. LLM-as-judge fallback when the regex doesn't trigger but
       refusal is ambiguous.

Design decisions:
  - Binary score (0.0 / 1.0). A partial refusal isn't really a thing
    in this dataset; either the model bailed out or it didn't.
  - Regex first because LLM-judge calls cost real money on a 200-row
    dev set, and the canonical phrasings catch ~80% of cases.
  - Fallback uses the same JSON parsing convention as
    src.evaluation.* for consistency.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


# Canonical refusal patterns. Case-insensitive; match anywhere in answer.
_REFUSAL_PATTERNS = [
    re.compile(r"\bcannot answer\b", re.IGNORECASE),
    re.compile(r"\bcan't answer\b", re.IGNORECASE),
    re.compile(r"\bunable to answer\b", re.IGNORECASE),
    re.compile(r"\bdo(es)? not contain\b", re.IGNORECASE),
    re.compile(r"\bdoesn't contain\b", re.IGNORECASE),
    re.compile(r"\bnot (mentioned|stated|provided|specified)\b", re.IGNORECASE),
    re.compile(r"\bno information\b", re.IGNORECASE),
    re.compile(r"\bcontext does not (address|cover|contain|mention)\b", re.IGNORECASE),
    re.compile(r"\bI don'?t know\b", re.IGNORECASE),
    re.compile(r"\bcannot be answered\b", re.IGNORECASE),
]


def is_refusal(answer: str) -> bool:
    """Heuristic check: does the answer use canonical refusal phrasing?

    Args:
        answer: The generated answer text to classify.

    Returns:
        True if any canonical refusal pattern matches; False otherwise.
    """
    return any(p.search(answer) for p in _REFUSAL_PATTERNS)


def _judge_is_refusal(answer: str, llm) -> bool:
    """LLM-as-judge fallback for ambiguous refusal detection.

    WHY: When the regex fast path doesn't match, we delegate to an LLM
    that can understand hedged, indirect, or colloquial refusals that
    don't use canonical phrasing (e.g. "I'd rather not speculate").

    PATTERN: Strict JSON output — {"is_refusal": bool} — avoids
    parsing freeform text and keeps the judge deterministic.

    Args:
        answer: The generated answer text to classify.
        llm: Any object with a generate(prompt, system_prompt=) method.

    Returns:
        True if the LLM judges the answer as a refusal; False otherwise
        (also False on JSON parse failure, logged as warning).
    """
    system_prompt = (
        "You are a classification assistant. "
        "Respond ONLY with a JSON object in this exact format: "
        '{"is_refusal": true} or {"is_refusal": false}. '
        "No other text."
    )
    user_prompt = (
        "Does the following answer refuse to answer the question "
        "(e.g. claims it cannot answer, lacks context, or doesn't know)?\n\n"
        f"Answer: {answer}"
    )

    raw = llm.generate(user_prompt, system_prompt=system_prompt)

    # PATTERN: Strip markdown code fences before parsing JSON — LLMs often
    # wrap JSON in ```json ... ``` even when instructed not to.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    try:
        parsed = json.loads(cleaned)
        return bool(parsed.get("is_refusal", False))
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("LLM judge returned non-JSON response: %r (%s)", raw, exc)
        return False


def refusal_correctness(answer: str, is_unanswerable: bool, llm) -> float:
    """Score whether the answer correctly handles an answerable/unanswerable question.

    Scoring logic:
      - refused AND is_unanswerable  → 1.0 (correct refusal)
      - not refused AND answerable   → 1.0 (correct attempt)
      - refused AND answerable       → 0.0 (false refusal)
      - not refused AND unanswerable → 0.0 (missed refusal)

    TRADE-OFF: Two-stage detection keeps LLM costs low. The regex fast
    path handles canonical phrasings without any API call. Only truly
    ambiguous answers pay the LLM-judge cost.

    Args:
        answer: The generated answer text.
        is_unanswerable: Ground-truth flag — True if the question has
            no answer in the retrieved context.
        llm: Any object with a generate(prompt, system_prompt=) method.
            Only called when the regex fast path doesn't match.

    Returns:
        1.0 if refusal classification matches is_unanswerable, else 0.0.
    """
    # PATTERN: Regex fast path — no LLM call if the answer uses
    # canonical refusal phrasing. Covers ~80% of cases.
    if is_refusal(answer):
        refused = True
    else:
        # Fallback: delegate to LLM judge for ambiguous cases.
        refused = _judge_is_refusal(answer, llm)

    return 1.0 if (refused == is_unanswerable) else 0.0
