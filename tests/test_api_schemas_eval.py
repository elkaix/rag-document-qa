"""Tests for src.api.schemas.eval DTOs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.api.schemas.eval import (
    AggregatedMetricDTO,
    EvalResultDTO,
    RunDetailDTO,
    RunStatusDTO,
    RunSubmitRequest,
    RunSubmitResponse,
    RunSummaryDTO,
)
from src.eval.schemas import RunMetadata


def _meta() -> RunMetadata:
    now = datetime.now(timezone.utc)
    return RunMetadata(
        run_id="r1", config_name="baseline", config_path="x.yaml",
        git_sha="abc1234", started_at=now, finished_at=now,
        env_hash="h", eval_set_versions={"squad_v2_dev_200": "v1"},
        n_questions=10, n_errors=0,
    )


class TestRunSummaryDTO:
    def test_construction(self):
        now = datetime.now(timezone.utc)
        d = RunSummaryDTO(
            run_id="r1", config_name="baseline",
            started_at=now, finished_at=now,
            n_questions=10, n_errors=0, headline_metric=0.84,
        )
        assert d.headline_metric == 0.84

    def test_headline_metric_optional(self):
        now = datetime.now(timezone.utc)
        d = RunSummaryDTO(
            run_id="r1", config_name="baseline",
            started_at=now, finished_at=now,
            n_questions=10, n_errors=0, headline_metric=None,
        )
        assert d.headline_metric is None


class TestAggregatedMetricDTO:
    def test_construction(self):
        d = AggregatedMetricDTO(
            metric_name="recall_at_5", dataset="squad_v2_dev_200",
            mean=0.84, ci_low=0.81, ci_high=0.87, n=200,
        )
        assert d.dataset == "squad_v2_dev_200"

    def test_dataset_can_be_none(self):
        d = AggregatedMetricDTO(
            metric_name="recall_at_5", dataset=None,
            mean=0.84, ci_low=0.81, ci_high=0.87, n=200,
        )
        assert d.dataset is None


class TestRunDetailDTO:
    def test_construction(self):
        d = RunDetailDTO(
            metadata=_meta(),
            aggregated=[AggregatedMetricDTO(
                metric_name="x", dataset=None, mean=0.5,
                ci_low=0.4, ci_high=0.6, n=10,
            )],
            cost={"total_usd": 0.01, "mean_usd_per_query": 0.001},
            n_results=10,
        )
        assert d.metadata.run_id == "r1"
        assert d.n_results == 10


class TestEvalResultDTO:
    def test_construction(self):
        d = EvalResultDTO(
            question_id="q1", dataset="squad_v2_dev_200",
            generated_answer="ans", metrics={"recall_at_5": 1.0},
            error=None,
        )
        assert d.error is None


class TestRunSubmitRequest:
    def test_construction(self):
        r = RunSubmitRequest(config_name="baseline")
        assert r.config_name == "baseline"

    def test_missing_config_name_raises(self):
        with pytest.raises(ValidationError):
            RunSubmitRequest()


class TestRunSubmitResponse:
    def test_valid_status(self):
        r = RunSubmitResponse(run_id="r1", status="queued")
        assert r.status == "queued"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            RunSubmitResponse(run_id="r1", status="invalid")


class TestRunStatusDTO:
    def test_construction(self):
        s = RunStatusDTO(
            run_id="r1", status="running",
            progress=0.5, n_completed=5, n_total=10,
            error_message=None,
        )
        assert s.progress == 0.5
