"""Tests for src.eval.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.eval.config import (
    EvalCfg,
    EvalConfig,
    GeneratorCfg,
    PipelineCfg,
    RetrieverCfg,
    ChunkerCfg,
    load_config,
)


def _baseline_dict() -> dict:
    return {
        "name": "baseline",
        "description": "test",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64},
            "retriever": {"top_k": 5},
            "generator": {"model": "gpt-5-mini", "reasoning_model": "gpt-4.1-nano"},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"],
            "judge_model": "gpt-4.1-mini",
            "bootstrap_n": 1000,
            "permutation_n": 10000,
            "seed": 42,
        },
    }


class TestEvalConfigConstruction:
    def test_full_construction(self):
        cfg = EvalConfig.model_validate(_baseline_dict())
        assert cfg.name == "baseline"
        assert cfg.pipeline.chunker.strategy == "recursive"
        assert cfg.eval.datasets == ["squad_v2_dev_200"]

    def test_defaults_applied_when_omitted(self):
        d = _baseline_dict()
        del d["pipeline"]["chunker"]["chunk_size"]
        del d["eval"]["bootstrap_n"]
        cfg = EvalConfig.model_validate(d)
        assert cfg.pipeline.chunker.chunk_size == 512  # default
        assert cfg.eval.bootstrap_n == 1000  # default

    def test_missing_required_field_raises(self):
        d = _baseline_dict()
        del d["name"]
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(d)

    def test_unknown_dataset_raises(self):
        d = _baseline_dict()
        d["eval"]["datasets"] = ["unknown_dataset"]
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(d)

    def test_invalid_chunker_strategy_raises(self):
        d = _baseline_dict()
        d["pipeline"]["chunker"]["strategy"] = "wrong"
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(d)


class TestLoadConfig:
    def test_loads_yaml_file(self, tmp_path: Path):
        path = tmp_path / "test.yaml"
        path.write_text(yaml.safe_dump(_baseline_dict()))
        cfg = load_config(path)
        assert cfg.name == "baseline"
        assert cfg.pipeline.retriever.top_k == 5

    def test_round_trip(self, tmp_path: Path):
        original = EvalConfig.model_validate(_baseline_dict())
        path = tmp_path / "rt.yaml"
        path.write_text(yaml.safe_dump(original.model_dump()))
        loaded = load_config(path)
        assert loaded == original

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")

    def test_baseline_yaml_loads(self):
        """The shipped baseline config must always validate."""
        cfg = load_config(Path("configs/eval/baseline.yaml"))
        assert cfg.name == "baseline"
        assert "squad_v2_dev_200" in cfg.eval.datasets
