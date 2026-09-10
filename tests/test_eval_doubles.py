"""Tests for src.eval.doubles — the shared eval LLM double and its switch."""

from __future__ import annotations

import json

from src.eval.doubles import (
    DUMMY_OVERRIDE_ENV,
    DummyEvalLLM,
    LLMOverrides,
    resolve_llm_overrides,
)


class TestDummyEvalLLM:
    def test_judge_prompt_returns_parseable_json(self):
        raw = DummyEvalLLM().generate("rate this", system_prompt="Respond in JSON")
        verdict = json.loads(raw)
        assert verdict["score"] == 1.0
        assert verdict["is_refusal"] is False

    def test_score_bearing_prompt_also_returns_json(self):
        raw = DummyEvalLLM().generate('return a "score" please')
        assert json.loads(raw)["factual_match"] == 1.0

    def test_answer_prompt_returns_placeholder(self):
        assert DummyEvalLLM().generate("What is RAG?") == "<dummy>"


class TestResolveLLMOverrides:
    def test_returns_empty_overrides_when_unset(self, monkeypatch):
        monkeypatch.delenv(DUMMY_OVERRIDE_ENV, raising=False)
        assert resolve_llm_overrides() == LLMOverrides(llm=None, judge_llm=None)

    def test_fills_both_slots_when_enabled(self, monkeypatch):
        monkeypatch.setenv(DUMMY_OVERRIDE_ENV, "1")
        overrides = resolve_llm_overrides()
        assert isinstance(overrides.llm, DummyEvalLLM)
        assert isinstance(overrides.judge_llm, DummyEvalLLM)

    def test_generator_and_judge_share_one_double(self, monkeypatch):
        monkeypatch.setenv(DUMMY_OVERRIDE_ENV, "1")
        overrides = resolve_llm_overrides()
        assert overrides.llm is overrides.judge_llm

    def test_any_other_value_is_not_enabled(self, monkeypatch):
        monkeypatch.setenv(DUMMY_OVERRIDE_ENV, "true")
        assert resolve_llm_overrides().llm is None
