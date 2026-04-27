"""Tests for QueryRewriter — LLM query expansion with cost capture."""

from __future__ import annotations


class _StubLLM:
    """Records calls and returns canned responses + token counts."""

    def __init__(self, response: str, prompt_tokens: int = 50, completion_tokens: int = 30):
        self._response = response
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[tuple[str, str | None]] = []

    def generate_with_usage(self, prompt: str, system_prompt: str | None = None
                             ) -> tuple[str, int, int]:
        self.calls.append((prompt, system_prompt))
        return self._response, self._prompt_tokens, self._completion_tokens


def test_no_model_passthrough():
    """When model is None, expand returns [query] unchanged with zero cost."""
    from src.eval.transforms import QueryRewriter
    rw = QueryRewriter(model=None, max_expansions=3, llm=None)
    queries, cost, p_t, c_t = rw.expand("What is RAG?")
    assert queries == ["What is RAG?"]
    assert cost == 0.0
    assert p_t == 0
    assert c_t == 0


def test_expansion_returns_dedup_list_and_cost():
    """With a real model name and stub LLM, expand returns deduped expansions + cost."""
    from src.eval.transforms import QueryRewriter
    stub = _StubLLM(
        response='["What does RAG stand for?", "Define retrieval augmented generation", '
                 '"What is RAG?"]',
        prompt_tokens=80, completion_tokens=40,
    )
    rw = QueryRewriter(model="gpt-4.1-nano", max_expansions=3, llm=stub)
    queries, cost, p_t, c_t = rw.expand("What is RAG?")
    # Original query is always first; duplicate dropped; max_expansions=3 cap respected.
    assert queries[0] == "What is RAG?"
    assert "What does RAG stand for?" in queries
    assert "Define retrieval augmented generation" in queries
    assert len(queries) == len(set(queries))  # no duplicates
    assert len(queries) <= 4  # original + at most max_expansions
    # Cost was computed from the stub's token counts at gpt-4.1-nano price.
    assert cost > 0.0
    assert p_t == 80
    assert c_t == 40


def test_malformed_llm_response_falls_back_to_passthrough():
    """If the LLM returns non-JSON, expand returns [query] and logs a warning."""
    from src.eval.transforms import QueryRewriter
    stub = _StubLLM(response="not json at all", prompt_tokens=50, completion_tokens=10)
    rw = QueryRewriter(model="gpt-4.1-nano", max_expansions=3, llm=stub)
    queries, cost, _, _ = rw.expand("What is RAG?")
    assert queries == ["What is RAG?"]
    # Cost is still charged because the call did happen.
    assert cost > 0.0
