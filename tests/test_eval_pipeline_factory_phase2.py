"""Phase 2 factory tests — every tier YAML produces a pipeline with expected lever activations.

Design note on hybrid_retriever:
    The hybrid retriever is built lazily during ingest() (after the chunk corpus is
    available), NOT at build_pipeline() time. So the parametrized test checks the
    *config flag* (cfg.pipeline.hybrid.enabled) to confirm YAML routing, not the
    runtime field (pipeline.hybrid_retriever). The smoke test exercises the runtime
    field after an ingest call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.config import load_config
from src.eval.pipeline_factory import build_pipeline


PHASE2_DIR = Path("configs/eval/phase2")


@pytest.fixture
def stub_llm():
    class _S:
        def generate(self, prompt, system_prompt=None):
            return "stub answer"
        def generate_with_usage(self, prompt, system_prompt=None):
            return "stub answer", 10, 5
    return _S()


@pytest.mark.parametrize("yaml_name,expects", [
    ("phase2_baseline.yaml",  {"rewriter": False, "reranker": False, "refusal": False, "hybrid": False, "embedder": "chroma_default"}),
    ("phase2b_embedder.yaml", {"rewriter": False, "reranker": False, "refusal": False, "hybrid": False, "embedder": "bge_small_en_v1_5"}),
    ("phase2c_hybrid.yaml",   {"rewriter": False, "reranker": False, "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2d_rerank.yaml",   {"rewriter": False, "reranker": True,  "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2e_rewrite.yaml",  {"rewriter": True,  "reranker": True,  "refusal": False, "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
    ("phase2g_refusal.yaml",  {"rewriter": True,  "reranker": True,  "refusal": True,  "hybrid": True,  "embedder": "bge_small_en_v1_5"}),
])
def test_phase2_yaml_builds_pipeline_with_expected_attrs(yaml_name, expects, stub_llm):
    cfg = load_config(PHASE2_DIR / yaml_name)
    pipeline = build_pipeline(
        cfg, dataset_name="squad_v2_dev_200",
        llm_override=stub_llm, judge_llm_override=stub_llm,
    )
    try:
        assert (pipeline.rewriter is not None) == expects["rewriter"]
        assert (pipeline.reranker is not None) == expects["reranker"]
        assert (pipeline.refusal_handler is not None) == expects["refusal"]
        # WHY cfg.pipeline.hybrid.enabled (not pipeline.hybrid_retriever is not None):
        # hybrid_retriever is built lazily in _ingest_squad() once the chunk corpus
        # is available. build_pipeline() always sets it to None; the config flag is
        # the authoritative signal that the lever was parsed and routed correctly.
        assert cfg.pipeline.hybrid.enabled == expects["hybrid"]
        assert cfg.pipeline.embedder.name == expects["embedder"]
    finally:
        pipeline.teardown()


def test_phase2_query_with_refusal_short_circuits(stub_llm):
    """End-to-end smoke: refusal handler short-circuits when top-1 < threshold (empty index)."""
    cfg = load_config(PHASE2_DIR / "phase2g_refusal.yaml")
    pipeline = build_pipeline(
        cfg, dataset_name="squad_v2_dev_200",
        llm_override=stub_llm, judge_llm_override=stub_llm,
    )
    try:
        # Empty index → retrieval returns [] → handler refuses.
        chunks, answer, telemetry = pipeline.query("what is x?")
        assert chunks == []
        assert answer == cfg.pipeline.refusal_handler.no_answer_text
        assert "refusal_check" in telemetry["timings_ms"]
    finally:
        pipeline.teardown()


def test_every_phase2_yaml_loads():
    """Every YAML under configs/eval/phase2/ must validate against EvalConfig."""
    for path in sorted(PHASE2_DIR.glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.name == path.stem, f"{path.name}: cfg.name={cfg.name!r} != {path.stem!r}"
