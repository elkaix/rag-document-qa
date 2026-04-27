"""
EvalConfig — typed pipeline-and-eval configuration loaded from YAML.

Eval Harness Position:
  configs/eval/*.yaml → load_config() → EvalConfig → EvalRunner.run()

Design decisions:
  - Pydantic v2 with nested models (ChunkerCfg, RetrieverCfg, etc.) so
    each subsystem owns its own validation surface.
  - Literal["..."] on dataset names and chunker strategy gives schema-
    level rejection of typos before the runner spins up.
  - YAML over JSON for human authorability — eval configs are written
    by hand, not generated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ChunkerCfg(BaseModel):
    """Chunking strategy configuration.

    Teaches: how chunking parameters propagate from config to the pipeline.
    Why Literal: rejects typos ("recusive") at validation time, not at runtime.
    Pipeline position: INDEXING step — Document → [Chunker] → Embeddings.
    """

    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrieverCfg(BaseModel):
    """Retriever configuration.

    Teaches: separating retrieval hyperparameters from the retriever implementation.
    Pipeline position: QUERYING step — Embeddings → [Retriever] → Top-K chunks.
    """

    top_k: int = 5


class GeneratorCfg(BaseModel):
    """LLM generator configuration.

    Teaches: optional reasoning model pattern — some providers support a
    lightweight "thinking" model before the final answer model.
    Pipeline position: QUERYING step — Chunks → [Generator] → Answer.
    """

    model: str = "gpt-5-mini"
    # WHY: reasoning_model is optional — None disables the CoT pre-pass.
    reasoning_model: str | None = "gpt-4.1-nano"


class PipelineCfg(BaseModel):
    """Aggregates all pipeline-level sub-configs into one validated structure.

    Pattern: nested Pydantic models — each sub-config validates independently,
    and the parent catches any cross-field issues at one boundary.
    """

    chunker: ChunkerCfg
    retriever: RetrieverCfg
    generator: GeneratorCfg


class EvalCfg(BaseModel):
    """Evaluation harness parameters.

    Teaches: statistical evaluation design — bootstrap CI and permutation
    tests are the two workhorses for comparing RAG pipeline variants.
    Why seed: reproducibility across runs and machines.
    """

    datasets: list[Literal["squad_v2_dev_200", "ml_papers_v1"]]
    judge_model: str = "gpt-4.1-mini"
    bootstrap_n: int = 1000
    permutation_n: int = 10000
    seed: int = 42


class EvalConfig(BaseModel):
    """Root configuration model for one named evaluation run.

    Each YAML file in configs/eval/ maps 1:1 to one EvalConfig instance.
    The name field doubles as a human-readable label in eval reports.
    """

    name: str
    description: str = ""
    pipeline: PipelineCfg
    # TRADE-OFF: `eval` shadows the Python builtin, but as a field name on a
    # Pydantic model it is unambiguous — accessed as cfg.eval.datasets, never
    # called as a function. Kept for schema clarity over renaming.
    eval: EvalCfg


def load_config(path: Path) -> EvalConfig:
    """Load a YAML config file, validate it, and return an EvalConfig.

    Args:
        path: Filesystem path to a YAML config file.

    Returns:
        Validated EvalConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If the YAML content fails schema validation.
    """
    # WHY: Raise FileNotFoundError explicitly rather than letting yaml.safe_load
    # raise a less informative OSError. Callers can distinguish "wrong path"
    # from "bad content" without inspecting exception types.
    if not path.exists():
        raise FileNotFoundError(f"Eval config not found: {path}")

    raw = yaml.safe_load(path.read_text())
    return EvalConfig.model_validate(raw)
