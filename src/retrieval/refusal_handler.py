"""RefusalHandler — answerability gate based on the best retrieval similarity.

Pipeline position:
    Retriever (post-rerank) candidates → [RefusalHandler] → answer or refusal text

The gate reads the *best* score among the candidates, not the first one —
hybrid retrieval orders by fused rank, so position 0 is not the top score.
See should_refuse for the full reasoning.

Phase 2 lever 2g. SQuAD v2 includes 'unanswerable' questions whose gold
answer is the empty string. Phase 1's pipeline always tries to answer,
which means it scores poorly on `refusal_correctness`. RefusalHandler is
a deterministic short-circuit: when no candidate clears the similarity
threshold, return a fixed no-answer text instead of calling the LLM.
"""

from __future__ import annotations

from src.domain import SearchResult


class RefusalHandler:
    """Deterministic answerability gate driven by the best similarity score in the set."""

    def __init__(
        self,
        enabled: bool,
        similarity_threshold: float,
        no_answer_text: str,
    ) -> None:
        """Configure the gate.

        Args:
            enabled: When False, should_refuse always returns False.
            similarity_threshold: The best score in the set must be >= this to
                NOT refuse. Deliberately not "top-1": see should_refuse.
            no_answer_text: Text returned in place of an LLM answer on refusal.
        """
        self._enabled = enabled
        self._threshold = similarity_threshold
        self._no_answer_text = no_answer_text

    def should_refuse(self, candidates: list[SearchResult]) -> bool:
        """Return True if the pipeline should short-circuit to no-answer text.

        Args:
            candidates: Retrieved chunks. May be empty. Need not be sorted.

        Returns:
            True when the handler is enabled and no candidate reaches the
            threshold (or candidates is empty); False otherwise.

        BUG FIX: this read ``candidates[0].score``, which assumed the retriever
            returns results in descending-score order. Dense, reranked and
            multi-query retrieval all do, so the two forms agree there — but
            hybrid retrieval orders by *fused rank*, and a BM25-only hit at
            position 0 carries score 0.0 (no comparable dense similarity
            exists). The gate would then refuse a question the corpus answers
            well. The seam does promise descending *relevance* (see
            ``Retriever.retrieve``) — what it never promised, and what this code
            assumed, is descending *score*. Asking for the best score in the set
            is the question the gate actually means, and it is the only form
            that survives a strategy whose scores are not monotone in rank.
        """
        if not self._enabled:
            return False
        if not candidates:
            return True
        return max(c.score for c in candidates) < self._threshold

    def refuse_response(self) -> tuple[list[SearchResult], str]:
        """Return ([], no_answer_text) — used when should_refuse is True.

        Returns:
            A 2-tuple of (empty chunk list, configured no-answer text).
        """
        return [], self._no_answer_text
