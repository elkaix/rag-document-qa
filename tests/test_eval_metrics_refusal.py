"""Tests for src.eval.metrics.refusal."""

from __future__ import annotations

import json

import pytest

from src.eval.metrics.refusal import is_refusal, refusal_correctness


class FakeLLM:
    """Returns a fixed JSON string from generate(); records calls."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return json.dumps(self.payload)


class TestIsRefusal:
    @pytest.mark.parametrize(
        "answer",
        [
            "I cannot answer this based on the provided context.",
            "The context does not contain information about that.",
            "I don't know based on the documents I was given.",
            "This question cannot be answered from the retrieved sources.",
            "There is no information in the context to answer this question.",
            "The provided context does not address this question.",
        ],
    )
    def test_clear_refusals_match(self, answer: str):
        assert is_refusal(answer) is True

    @pytest.mark.parametrize(
        "answer",
        [
            "The Eiffel Tower is in Paris.",
            "According to the context, X equals Y.",
            "Yes — the documents state that ...",
        ],
    )
    def test_clear_answers_dont_match(self, answer: str):
        assert is_refusal(answer) is False


class TestRefusalCorrectness:
    def test_correct_refusal_on_unanswerable(self):
        llm = FakeLLM({"is_refusal": True})
        score = refusal_correctness(
            answer="I cannot answer this based on the provided context.",
            is_unanswerable=True,
            llm=llm,
        )
        assert score == 1.0
        assert llm.calls == []

    def test_incorrect_attempt_on_unanswerable(self):
        llm = FakeLLM({"is_refusal": False})
        score = refusal_correctness(
            answer="The capital of France is Paris.",
            is_unanswerable=True,
            llm=llm,
        )
        assert score == 0.0

    def test_correct_attempt_on_answerable(self):
        llm = FakeLLM({"is_refusal": False})
        score = refusal_correctness(
            answer="The capital of France is Paris.",
            is_unanswerable=False,
            llm=llm,
        )
        assert score == 1.0

    def test_incorrect_refusal_on_answerable(self):
        llm = FakeLLM({"is_refusal": True})
        score = refusal_correctness(
            answer="I cannot answer this based on the provided context.",
            is_unanswerable=False,
            llm=llm,
        )
        assert score == 0.0

    def test_llm_judge_fallback_on_ambiguous(self):
        llm = FakeLLM({"is_refusal": True})
        score = refusal_correctness(
            answer="That's an interesting question, but I'd rather not speculate.",
            is_unanswerable=True,
            llm=llm,
        )
        assert score == 1.0
        assert len(llm.calls) == 1
