"""RefusalHandler — answerability gate based on top-1 retrieval similarity.

Pipeline position:
    Retriever (post-rerank) candidates → [RefusalHandler] → answer or refusal text

Phase 2 lever 2g. SQuAD v2 includes 'unanswerable' questions whose gold
answer is the empty string. Phase 1's pipeline always tries to answer,
which means it scores poorly on `refusal_correctness`. RefusalHandler is
a deterministic short-circuit: when no candidate clears the similarity
threshold, return a fixed no-answer text instead of calling the LLM.
"""

from __future__ import annotations

from src.vector_store import SearchResult


class RefusalHandler:
    """Deterministic answerability gate driven by top-1 similarity score."""

    def __init__(
        self,
        enabled: bool,
        similarity_threshold: float,
        no_answer_text: str,
    ) -> None:
        """Configure the gate.

        Args:
            enabled: When False, should_refuse always returns False.
            similarity_threshold: Top-1 score must be >= this to NOT refuse.
            no_answer_text: Text returned in place of an LLM answer on refusal.
        """
        self._enabled = enabled
        self._threshold = similarity_threshold
        self._no_answer_text = no_answer_text

    def should_refuse(self, candidates: list[SearchResult]) -> bool:
        """Return True if the pipeline should short-circuit to no-answer text.

        Args:
            candidates: Retrieved chunks ordered by descending similarity.
                        May be empty.

        Returns:
            True when the handler is enabled and the top-1 score is below
            the threshold (or candidates is empty); False otherwise.
        """
        if not self._enabled:
            return False
        if not candidates:
            return True
        return candidates[0].score < self._threshold

    def refuse_response(self) -> tuple[list[SearchResult], str]:
        """Return ([], no_answer_text) — used when should_refuse is True.

        Returns:
            A 2-tuple of (empty chunk list, configured no-answer text).
        """
        return [], self._no_answer_text
