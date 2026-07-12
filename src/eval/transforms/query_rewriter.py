"""QueryRewriter — LLM-based query expansion with token/cost capture.

Pipeline position:
    user query → [QueryRewriter] → {q, q', q''} → Retriever → ...

Phase 2 lever 2e. Expansion gives the retriever multiple lexical/semantic
formulations of the same intent, which raises recall on questions where the
original phrasing diverges from the corpus phrasing. We use a tiny model
(gpt-4.1-nano) because the task is cheap and we don't want this lever to
dominate the cost ledger.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from src.telemetry import pricing

logger = logging.getLogger(__name__)


class _LLMHandler(Protocol):
    """Structural type for any object exposing generate_with_usage."""

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None,
    ) -> tuple[str, int, int]: ...


class QueryRewriter:
    """Expands one user query into up to N alternative phrasings via an LLM."""

    SYSTEM_PROMPT = (
        "You rewrite user search queries into alternative phrasings that preserve "
        "the original intent but vary surface form. Respond ONLY with a JSON "
        "array of strings — no prose, no code fences."
    )

    def __init__(
        self,
        model: str | None,
        max_expansions: int,
        llm: _LLMHandler | None,
    ) -> None:
        """Configure the rewriter.

        Args:
            model: LLM model name. None disables rewriting (pass-through).
            max_expansions: Cap on the number of alternative phrasings to return.
            llm: Object exposing generate_with_usage(prompt, system_prompt). Required
                if model is not None.
        """
        self._model = model
        self._max_expansions = max_expansions
        self._llm = llm

    def expand(self, query: str) -> tuple[list[str], float, int, int]:
        """Expand `query` into up to N+1 unique phrasings.

        Returns:
            (queries, cost_usd, prompt_tokens, completion_tokens). The original
            query is always the first element. When `model is None`, returns
            ([query], 0.0, 0, 0) and skips the LLM call.
        """
        if self._model is None:
            return [query], 0.0, 0, 0
        if self._llm is None:
            raise ValueError("QueryRewriter has model set but no llm handler provided.")

        user_prompt = (
            f'Original query: "{query}"\n\n'
            f"Return a JSON array of up to {self._max_expansions} alternative "
            f"phrasings of this query. Do NOT include the original."
        )
        raw, p_t, c_t = self._llm.generate_with_usage(
            user_prompt, system_prompt=self.SYSTEM_PROMPT,
        )
        cost = pricing.cost_usd(self._model, p_t, c_t)

        expansions = self._parse_expansions(raw)
        # Always lead with original; dedupe; cap at original + max_expansions.
        ordered: list[str] = [query]
        for alt in expansions:
            if alt and alt not in ordered:
                ordered.append(alt)
            if len(ordered) >= self._max_expansions + 1:
                break
        return ordered, cost, p_t, c_t

    @staticmethod
    def _parse_expansions(raw: str) -> list[str]:
        """Strip code fences and parse the JSON array; return [] on failure."""
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(
                "QueryRewriter got non-JSON response — falling back to [query] only."
            )
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if isinstance(item, str)]
