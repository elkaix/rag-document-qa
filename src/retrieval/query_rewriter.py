"""Multi-query expansion — the LLM rewriter and the Retriever adapter that composes it.

Pipeline position:
    user query → [MultiQueryRetriever → QueryRewriter] → {q, q', q''} → inner Retriever → union

Two collaborators live here:

- `QueryRewriter` expands one user query into alternative phrasings via a tiny LLM
  (gpt-4.1-nano — cheap, so this lever doesn't dominate the cost ledger). Expansion
  raises recall when the user's phrasing diverges from the corpus phrasing.
- `MultiQueryRetriever` presents the `Retriever` interface by *composing* an inner
  Retriever: it fans the expansions out and fuses the per-expansion rankings
  with RRF, keeping each chunk's best observed score for downstream consumers.
  The "compose rather than conform" adapter from ADR 0004.

The rewriter reports its own token cost, but the pure `Retriever` interface has
no cost channel, so it is *logged* at this seam rather than returned. This was
deliberate in step 4c: the eval harness's old `rewriter_cost_usd` field had zero
readers (verified), so convergence dropped it rather than plumb a cost path
nothing consumed. Wiring the strategy for production (ADR 0009) did not change
that arithmetic — it only meant an operator now needs to *see* the spend, which
a log line does without every adapter growing a field. A structured channel can
replace the log the day a consumer exists to read one.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from src.domain import SearchResult
from src.retrieval.base import Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.telemetry import pricing

logger = logging.getLogger(__name__)


class _LLMHandler(Protocol):
    """Structural type for any object exposing generate_with_usage."""

    def generate_with_usage(
        self,
        prompt: str,
        system_prompt: str | None = None,
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
        # BUG FIX: only the JSON parse was guarded, so a provider outage, an
        #          expired key, a rate limit, or a timeout raised straight out
        #          of retrieval and failed the user's whole question. Query
        #          expansion is a recall *optimisation*: without it retrieval
        #          still works, it is just narrower. Degrading to the original
        #          query is strictly better than answering nothing, so the one
        #          place this lever can reach the network is where it is caught.
        # WHY a broad except: the raising types are whichever SDK the configured
        #          provider happens to use — openai, anthropic, requests, each
        #          with its own exception tree. Enumerating them would couple
        #          this module to every provider and still miss the next one.
        #          The failure is logged with its traceback, not swallowed.
        try:
            raw, p_t, c_t = self._llm.generate_with_usage(
                user_prompt,
                system_prompt=self.SYSTEM_PROMPT,
            )
        except Exception:
            logger.exception("Query expansion failed — retrieving with the original query only.")
            return [query], 0.0, 0, 0
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
            logger.warning("QueryRewriter got non-JSON response — falling back to [query] only.")
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
        """Retrieve for every expansion, then fuse the rankings by RRF.

        Args:
            query: The original user query.
            top_k: Number of results to return after fusion. Each expansion is
                itself retrieved at `top_k` before the rankings are fused.

        Returns:
            Up to `top_k` SearchResults in fused-rank order — a chunk that ranks
            well under several phrasings beats one that ranks highest under a
            single phrasing. Each result keeps the best score it was seen with,
            for consumers that read the field.

        BUG FIX: this ranked the union by ``score`` and truncated. That is only
            correct when every inner result's score lives in one comparable
            space, which stops being true the moment the inner retriever is
            ``BM25HybridRetriever`` — its sparse-only hits carry 0.0 because no
            cosine similarity exists for them. Every eval config from
            phase2e onward stacks multi-query over hybrid, so the adapter was
            sorting exactly the results hybrid exists to contribute to the
            bottom of the list and then cutting them off. Fusing by rank
            removes the comparability assumption instead of documenting it —
            the same fix ``RefusalHandler`` needed for the same reason.
        """
        # WHY the cost is logged rather than returned: `retrieve(query, top_k)`
        #     is the whole Retriever seam, and widening it with a spend channel
        #     would make every adapter and every caller carry a field that one
        #     lever produces and nothing consumes — the same verified-zero-
        #     readers reasoning that retired the eval harness's
        #     `rewriter_cost_usd`. Logging surfaces the spend for the operator
        #     who turns this strategy on; a structured channel can replace the
        #     log the day a consumer exists to read one.
        expansions, cost_usd, prompt_tokens, completion_tokens = self._rewriter.expand(query)
        if cost_usd or prompt_tokens or completion_tokens:
            logger.info(
                "Query expansion: %d queries, %d+%d tokens, $%.6f",
                len(expansions),
                prompt_tokens,
                completion_tokens,
                cost_usd,
            )
        best: dict[str, SearchResult] = {}
        rankings: list[list[str]] = []
        for expansion in expansions:
            results = self._inner.retrieve(expansion, top_k=top_k)
            rankings.append([r.chunk_id for r in results])
            for result in results:
                current = best.get(result.chunk_id)
                if current is None or result.score > current.score:
                    best[result.chunk_id] = result
        fused = reciprocal_rank_fusion(rankings)[:top_k]
        return [best[chunk_id] for chunk_id in fused]
