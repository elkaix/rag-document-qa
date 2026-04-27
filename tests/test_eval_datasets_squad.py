"""Tests for src.eval.datasets.squad_v2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.datasets.squad_v2 import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    load_frozen,
    sample_and_freeze,
)
from src.eval.schemas import EvalQuestion


@pytest.fixture
def temp_freeze_path(tmp_path: Path) -> Path:
    return tmp_path / "questions.jsonl"


class TestSampleAndFreeze:
    def test_sample_size_matches(self, temp_freeze_path: Path):
        result = sample_and_freeze(
            output_path=temp_freeze_path, sample_size=10, seed=42
        )
        assert len(result) == 10
        assert temp_freeze_path.exists()

    def test_seed_reproducibility(self, tmp_path: Path):
        a = sample_and_freeze(
            output_path=tmp_path / "a.jsonl", sample_size=5, seed=99
        )
        b = sample_and_freeze(
            output_path=tmp_path / "b.jsonl", sample_size=5, seed=99
        )
        assert [q.id for q in a] == [q.id for q in b]

    def test_includes_unanswerable_rows(self, temp_freeze_path: Path):
        result = sample_and_freeze(
            output_path=temp_freeze_path, sample_size=50, seed=7
        )
        n_unanswerable = sum(1 for q in result if q.is_unanswerable)
        n_answerable = sum(1 for q in result if not q.is_unanswerable)
        assert n_unanswerable > 0
        assert n_answerable > 0

    def test_each_row_has_required_fields(self, temp_freeze_path: Path):
        result = sample_and_freeze(
            output_path=temp_freeze_path, sample_size=5, seed=1
        )
        for q in result:
            assert isinstance(q, EvalQuestion)
            assert q.id
            assert q.question
            if not q.is_unanswerable:
                assert q.gold_answer
                assert len(q.gold_chunk_ids) >= 1
            else:
                assert q.gold_answer is None
                assert q.gold_chunk_ids == []


class TestLoadFrozen:
    def test_round_trip(self, temp_freeze_path: Path):
        original = sample_and_freeze(
            output_path=temp_freeze_path, sample_size=5, seed=5
        )
        loaded = load_frozen(temp_freeze_path)
        assert loaded == original

    def test_loads_from_jsonl_file(self, tmp_path: Path):
        path = tmp_path / "manual.jsonl"
        manual = [
            EvalQuestion(id="q1", question="Why?", gold_answer="A", gold_chunk_ids=["c"]),
            EvalQuestion(id="q2", question="How?", is_unanswerable=True),
        ]
        with path.open("w") as f:
            for q in manual:
                f.write(q.model_dump_json() + "\n")
        loaded = load_frozen(path)
        assert loaded == manual

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_frozen(tmp_path / "missing.jsonl")


class TestDefaults:
    def test_default_paths_and_constants_exposed(self):
        assert DEFAULT_SAMPLE_SIZE == 200
        assert DEFAULT_SEED == 12345
        assert "squad_v2_dev_200" in str(DEFAULT_OUTPUT_PATH)
