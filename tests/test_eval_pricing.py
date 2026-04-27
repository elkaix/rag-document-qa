"""Tests for src.eval.pricing."""

from __future__ import annotations

import logging

import pytest

from src.eval.pricing import MODEL_PRICES, ModelPrice, cost_usd


class TestModelPriceTable:
    def test_known_models_present(self):
        for model_id in ("gpt-5-mini", "gpt-4.1-mini", "gpt-4.1-nano"):
            assert model_id in MODEL_PRICES
            assert isinstance(MODEL_PRICES[model_id], ModelPrice)

    def test_prices_positive(self):
        for price in MODEL_PRICES.values():
            assert price.prompt_per_1m > 0
            assert price.completion_per_1m > 0


class TestCostUsd:
    def test_known_model_basic(self):
        price = MODEL_PRICES["gpt-4.1-mini"]
        result = cost_usd("gpt-4.1-mini", prompt_tokens=1_000_000, completion_tokens=0)
        assert result == pytest.approx(price.prompt_per_1m)

    def test_combined_cost(self):
        price = MODEL_PRICES["gpt-4.1-mini"]
        result = cost_usd(
            "gpt-4.1-mini", prompt_tokens=500_000, completion_tokens=500_000
        )
        expected = 0.5 * price.prompt_per_1m + 0.5 * price.completion_per_1m
        assert result == pytest.approx(expected)

    def test_zero_tokens(self):
        assert cost_usd("gpt-4.1-mini", 0, 0) == 0.0

    def test_unknown_model_returns_zero_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = cost_usd("nonexistent-model", 100, 100)
        assert result == 0.0
        assert any("unknown model" in rec.message.lower() for rec in caplog.records)

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError):
            cost_usd("gpt-4.1-mini", -1, 0)
        with pytest.raises(ValueError):
            cost_usd("gpt-4.1-mini", 0, -1)
