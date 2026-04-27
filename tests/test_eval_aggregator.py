"""Tests for src.eval.aggregator."""

from __future__ import annotations

import pytest

from src.eval.aggregator import aggregate
from src.eval.config import EvalConfig
from src.eval.schemas import AggregatedMetric, EvalResult


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "test", "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200", "ml_papers_v1"],
            "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 200, "permutation_n": 100, "seed": 42,
        },
    })


def _r(qid: str, dataset: str, metrics: dict[str, float], error: str | None = None) -> EvalResult:
    return EvalResult(
        question_id=qid, dataset=dataset,
        retrieved_chunk_ids=[], retrieved_chunks=[],
        generated_answer="", metrics=metrics, metric_details={},
        timings_ms={}, tokens={"prompt": 0, "completion": 0}, cost_usd=0.0,
        error=error,
    )


class TestAggregate:
    def test_per_dataset_and_combined(self):
        cfg = _baseline_config()
        results = []
        # 5 squad results with recall_at_5 = 1.0
        results += [_r(f"s{i}", "squad_v2_dev_200", {"recall_at_5": 1.0}) for i in range(5)]
        # 5 ml_papers results with recall_at_5 = 0.0
        results += [_r(f"m{i}", "ml_papers_v1", {"recall_at_5": 0.0}) for i in range(5)]

        aggregated, warnings = aggregate(results, cfg)
        assert warnings == []

        by_key = {(a.metric_name, a.dataset): a for a in aggregated}
        # Three rows: squad-only, ml_papers-only, combined
        assert by_key[("recall_at_5", "squad_v2_dev_200")].mean == pytest.approx(1.0)
        assert by_key[("recall_at_5", "ml_papers_v1")].mean == pytest.approx(0.0)
        assert by_key[("recall_at_5", None)].mean == pytest.approx(0.5)

    def test_low_n_skipped_and_warned(self):
        cfg = _baseline_config()
        # Only 2 squad results — below the 3-sample minimum.
        results = [
            _r("s1", "squad_v2_dev_200", {"recall_at_5": 1.0}),
            _r("s2", "squad_v2_dev_200", {"recall_at_5": 0.5}),
        ]
        aggregated, warnings = aggregate(results, cfg)
        # Per-dataset row skipped; combined also <3 samples → also skipped.
        assert all(a.metric_name != "recall_at_5" or a.dataset is None for a in aggregated) or aggregated == []
        # At least one warning mentions the skipped metric.
        assert any("recall_at_5" in w for w in warnings)

    def test_excludes_errored_results(self):
        cfg = _baseline_config()
        results = [_r(f"s{i}", "squad_v2_dev_200", {"recall_at_5": 1.0}) for i in range(5)]
        results.append(_r("err", "squad_v2_dev_200", {"recall_at_5": 0.0}, error="boom"))
        aggregated, _ = aggregate(results, cfg)
        # Errored row excluded → mean still 1.0, n=5.
        squad = next(a for a in aggregated if a.metric_name == "recall_at_5" and a.dataset == "squad_v2_dev_200")
        assert squad.mean == pytest.approx(1.0)
        assert squad.n == 5

    def test_multiple_metrics(self):
        cfg = _baseline_config()
        results = [
            _r(f"s{i}", "squad_v2_dev_200",
               {"recall_at_5": 1.0, "faithfulness": 0.9})
            for i in range(5)
        ]
        aggregated, _ = aggregate(results, cfg)
        names = {a.metric_name for a in aggregated}
        assert names == {"recall_at_5", "faithfulness"}

    def test_seed_propagated(self):
        """Two runs with the same config should yield identical CIs."""
        cfg = _baseline_config()
        results = [_r(f"s{i}", "squad_v2_dev_200", {"r": float(i % 2)}) for i in range(20)]
        a1, _ = aggregate(results, cfg)
        a2, _ = aggregate(results, cfg)
        assert {(a.metric_name, a.dataset, a.mean, a.ci_low, a.ci_high) for a in a1} == \
               {(a.metric_name, a.dataset, a.mean, a.ci_low, a.ci_high) for a in a2}
