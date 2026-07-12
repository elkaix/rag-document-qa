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
        from src.config import CHUNK_SIZE

        d = _baseline_dict()
        del d["pipeline"]["chunker"]["chunk_size"]
        del d["eval"]["bootstrap_n"]
        cfg = EvalConfig.model_validate(d)
        # Step 4c: the chunk-size default now derives from production config
        # (single source of truth), not a hard-coded eval literal.
        assert cfg.pipeline.chunker.chunk_size == CHUNK_SIZE
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


# --- Phase 2 schema additions ----------------------------------------------


def test_phase2_subconfigs_default_to_off(tmp_path):
    """Loading an existing baseline-shape YAML must produce all-default Phase 2 blocks."""
    yaml_text = """
name: legacy_baseline
description: existing config without phase 2 blocks
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    cfg = load_config(p)
    assert cfg.pipeline.embedder.name == "chroma_default"
    assert cfg.pipeline.hybrid.enabled is False
    assert cfg.pipeline.reranker.model is None
    assert cfg.pipeline.query_rewriter.model is None
    assert cfg.pipeline.refusal_handler.enabled is False
    assert cfg.eval.spend_ceiling_usd is None


def test_phase2_subconfig_typed_values(tmp_path):
    """Phase 2 fields validate to the right types."""
    yaml_text = """
name: phase2g
description: refusal handler enabled
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  embedder: {name: bge_small_en_v1_5}
  hybrid: {enabled: true, bm25_top_k: 20, dense_top_k: 20, rrf_k: 60}
  reranker: {model: ms_marco_minilm_l6_v2, rerank_top_n: 20, final_top_k: 5}
  query_rewriter: {model: gpt-4.1-nano, max_expansions: 3}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
  refusal_handler: {enabled: true, similarity_threshold: 0.35}
eval:
  datasets: [squad_v2_dev_200]
  spend_ceiling_usd: 1.5
"""
    p = tmp_path / "phase2g.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    cfg = load_config(p)
    assert cfg.pipeline.embedder.name == "bge_small_en_v1_5"
    assert cfg.pipeline.hybrid.enabled is True
    assert cfg.pipeline.reranker.model == "ms_marco_minilm_l6_v2"
    assert cfg.pipeline.query_rewriter.model == "gpt-4.1-nano"
    assert cfg.pipeline.refusal_handler.similarity_threshold == 0.35
    assert cfg.eval.spend_ceiling_usd == 1.5


def test_phase2_unknown_field_rejected(tmp_path):
    """extra='forbid' must reject unknown keys at load time."""
    yaml_text = """
name: bad
description: typo in field name
pipeline:
  chunker: {strategy: recursive, chunk_size: 512, chunk_overlap: 64}
  retriever: {top_k: 5}
  hybrid: {enabld: true}
  generator: {model: gpt-5-mini, reasoning_model: gpt-4.1-nano}
eval:
  datasets: [squad_v2_dev_200]
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    from src.eval.config import load_config
    with pytest.raises(ValidationError):
        load_config(p)
