"""
API DTOs for the eval routes.

API Layer Position:
  src/api/routes/eval.py → [DTOs] → JSON response

Design decisions:
  - DTOs separate from internal eval.schemas: API layer can evolve
    independently of the storage/runner layer.
  - Reuse RunMetadata (internal schema) inside RunDetailDTO instead of
    cloning fields — RunMetadata is JSON-serialisable already.
  - EvalResultDTO is a SLIM projection: omits retrieved_chunks (long
    text) and metric_details (LLM judge JSON blobs) so list views are
    fast. The dedicated /runs/{id}/results/{qid} endpoint returns the
    full EvalResult.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from src.eval.schemas import RunMetadata


class RunSummaryDTO(BaseModel):
    """Lightweight run summary for list views.

    Teaches: projection pattern — expose only the fields the UI needs
    for a list row, not the full run record. Keeps list endpoints fast.

    Pipeline role: Response body for GET /api/eval/runs (list endpoint).
    """

    run_id: str
    config_name: str
    started_at: datetime
    finished_at: datetime
    n_questions: int
    n_errors: int
    # WHY: headline_metric is optional because a run may have failed
    #      before any metrics were computed.
    headline_metric: float | None  # recall_at_5 mean if present, else None


class AggregatedMetricDTO(BaseModel):
    """One aggregated metric for API responses.

    Teaches: mirror-DTO pattern — mirrors AggregatedMetric (internal)
    but lives in the API layer so the two can diverge independently.

    Pipeline role: Element of RunDetailDTO.aggregated; also usable as
    a standalone response for per-metric endpoints.
    """

    metric_name: str
    # WHY: dataset=None means the metric is combined across all datasets.
    dataset: str | None
    mean: float
    ci_low: float
    ci_high: float
    n: int


class RunDetailDTO(BaseModel):
    """Full run detail — metadata + aggregated metrics + cost summary.

    Teaches: composition over duplication — embeds RunMetadata directly
    rather than copying its 10+ fields. The internal schema is already
    JSON-serialisable via Pydantic, so nesting is zero-cost.

    Pipeline role: Response body for GET /api/eval/runs/{run_id}.
    Results are paginated separately to keep this response bounded.
    """

    # PATTERN: Reuse internal RunMetadata directly — avoids field drift
    #          between the storage layer and the API layer.
    metadata: RunMetadata
    aggregated: list[AggregatedMetricDTO]
    cost: dict[str, float]
    # WHY: n_results tells the UI how many pages to expect without
    #      requiring it to load all results upfront.
    n_results: int  # results paginated separately


class EvalResultDTO(BaseModel):
    """Slim per-question result for UI table rows.

    Teaches: projection pattern — retrieved_chunks and metric_details
    are excluded here (they can be MBs per run) and are only returned
    by the dedicated /runs/{id}/results/{qid} endpoint.

    Pipeline role: Element of the paginated results list response for
    GET /api/eval/runs/{run_id}/results.
    """

    question_id: str
    dataset: str
    generated_answer: str
    metrics: dict[str, float]
    # WHY: error is None for successful evaluations; non-None means the
    #      pipeline raised an exception for this question.
    error: str | None


class RunSubmitRequest(BaseModel):
    """Request body for POST /api/eval/runs.

    Teaches: thin request model — only the config name is needed; all
    other run parameters come from the config file itself.

    Pipeline role: Validated by FastAPI before reaching the route handler.
    """

    # WHY: config_name must match a file in configs/eval/ — validation
    #      of that constraint happens in the route handler, not here.
    config_name: str  # must match a file in configs/eval/


class RunSubmitResponse(BaseModel):
    """Response body for POST /api/eval/runs.

    Teaches: async job pattern — the run is queued immediately and the
    caller polls /runs/{run_id}/status for progress.

    Pipeline role: Returned synchronously by the submit endpoint; the
    run_id is the handle for all subsequent status and result queries.
    """

    run_id: str
    status: Literal["queued", "running", "completed", "failed"]


class RunStatusDTO(BaseModel):
    """Polling response for GET /api/eval/runs/{run_id}/status.

    Teaches: progress reporting pattern — progress (0.0–1.0) and
    n_completed/n_total give the UI enough information to render a
    progress bar without polling the full result list.

    Pipeline role: Returned by the status endpoint; polled by the UI
    until status is "completed" or "failed".
    """

    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    # WHY: float 0.0–1.0 maps directly to a CSS/recharts progress bar.
    progress: float  # 0.0 - 1.0
    n_completed: int
    n_total: int
    # WHY: error_message is None unless status == "failed"; surfacing it
    #      here avoids a separate error endpoint for the common case.
    error_message: str | None
