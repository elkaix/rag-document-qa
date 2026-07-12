"""Tests for src.eval.metrics.generation."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from src.eval.metrics.generation import answer_correctness, context_recall


class FakeLLM:
    """Returns queued JSON payloads in order from generate()."""

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[str] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(prompt)
        if not self.payloads:
            raise RuntimeError("FakeLLM out of payloads")
        return json.dumps(self.payloads.pop(0))


class TestContextRecall:
    def test_perfect_recall(self):
        gold = ["c1", "c2"]
        retrieved = ["c1", "c2", "c3"]
        assert context_recall(gold, retrieved) == 1.0

    def test_partial_recall(self):
        gold = ["c1", "c2", "c3", "c4"]
        retrieved = ["c1", "c2", "x"]
        assert context_recall(gold, retrieved) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert context_recall(["c1"], ["x", "y"]) == 0.0

    def test_empty_gold_returns_nan(self):
        assert math.isnan(context_recall([], ["x"]))


class TestAnswerCorrectness:
    def test_high_similarity_and_judge_match(self):
        with patch(
            "src.eval.metrics.generation._embed",
            side_effect=lambda text: np.array([1.0, 0.0, 0.0]),
        ):
            llm = FakeLLM([{"factual_match": 1.0, "reasoning": "Same answer."}])
            score, details = answer_correctness(
                generated="Paris is the capital of France.",
                gold="The capital of France is Paris.",
                llm=llm,
            )
        assert score == pytest.approx(1.0)
        assert details["cosine"] == pytest.approx(1.0)
        assert details["judge_factual_match"] == pytest.approx(1.0)

    def test_low_similarity_and_judge_mismatch(self):
        with patch(
            "src.eval.metrics.generation._embed",
            side_effect=[
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
            ],
        ):
            llm = FakeLLM([{"factual_match": 0.0, "reasoning": "Different."}])
            score, details = answer_correctness(
                generated="The Eiffel Tower is in Paris.",
                gold="The Statue of Liberty is in New York.",
                llm=llm,
            )
        assert score == pytest.approx(0.0)

    def test_partial_match(self):
        with patch(
            "src.eval.metrics.generation._embed",
            side_effect=[np.array([1.0, 0.0]), np.array([0.5, 0.5])],
        ):
            llm = FakeLLM([{"factual_match": 0.5, "reasoning": "Partly."}])
            score, _ = answer_correctness(
                generated="Mostly right.", gold="The answer.", llm=llm
            )
        # cosine = 0.5/sqrt(0.5) ≈ 0.7071; judge = 0.5; mean ≈ 0.6036
        assert score == pytest.approx((1 / np.sqrt(2) + 0.5) / 2, abs=1e-3)
