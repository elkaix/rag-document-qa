"""Multi-query expansion — the LLM rewriter and the Retriever adapter that composes it.

Pipeline position:
    user query → [MultiQueryRetriever → QueryRewriter] → {q, q', q''} → inner Retriever → union

Two collaborators live here:

- `QueryRewriter` expands one user query into alternative phrasings via a tiny LLM
  (gpt-4.1-nano — cheap, so this lever doesn't dominate the cost ledger). Expansion
  raises recall when the user's phrasing diverges from the corpus phrasing.
- `MultiQueryRetriever` presents the `Retriever` interface by *composing* an inner
  Retriever: it fans the expansions out, unions the results, and dedups by
  chunk_id keeping each chunk's best score. The "compose rather than conform"
  adapter from ADR 0004.

The rewriter reports its own token cost; that number is dropped at this seam
(the pure `Retriever` interface has no cost channel). Surfacing multi-query cost
into telemetry is deferred to the QueryEngine convergence (step 4c).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from src.retrieval.base import Retriever
from src.telemetry import pricing
from src.vector_store import SearchResult

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


class _Rewriter(Protocol):
    """Structural type for a query expander (the one collaborator we inject)."""

    def expand(self, query: str) -> tuple[list[str], float, int, int]: ...


class MultiQueryRetriever:
    """Retriever adapter: fan an inner Retriever out over rewritten queries.

    Presents `retrieve(query, top_k)` while delegating expansion to a rewriter and
    candidate generation to an inner Retriever — so multi-query retrieval is
    interchangeable with any other strategy behind the same seam.
    """

    def __init__(self, inner: Retriever, rewriter: _Rewriter) -> None:
        """Compose an inner Retriever with a query expander.

        Args:
            inner: The Retriever run once per expanded query.
            rewriter: Produces the alternative phrasings (original query first).
        """
        self._inner = inner
        self._rewriter = rewriter

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve for every expansion, union, dedup by chunk_id, rank best-first.

        Args:
            query: The original user query.
            top_k: Number of results to return after the union is ranked. Each
                expansion is itself retrieved at `top_k` before the union.

        Returns:
            Up to `top_k` SearchResults ordered by descending score. When a chunk
            surfaces under several expansions, its highest score wins (dense
            similarities share the embedding space, so they compare directly).
        """
        best: dict[str, SearchResult] = {}
        for expansion in self._rewriter.expand(query)[0]:
            for result in self._inner.retrieve(expansion, top_k=top_k):
                current = best.get(result.chunk_id)
                if current is None or result.score > current.score:
                    best[result.chunk_id] = result
        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]
