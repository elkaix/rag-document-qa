"""
Eval API routes.

API Layer Position:
  Frontend (/eval pages) → [routes] → eval package + storage + registry

Design decisions:
  - Long-running runs dispatched via FastAPI's BackgroundTasks; the
    POST returns 202 immediately with a run_id the client can poll.
  - RunRegistry tracks in-flight progress for the status endpoint;
    persisted runs live on disk via storage.save_run.
  - Pagination on results so a 200-question run doesn't ship 200KB+
    of generated text per page load.
  - 409 on eval-set version mismatch in compare so the UI can surface
    a clear "these runs aren't comparable" banner.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status

from src.api.schemas.eval import (
    AggregatedMetricDTO,
    EvalResultDTO,
    RunDetailDTO,
    RunStatusDTO,
    RunSubmitRequest,
    RunSubmitResponse,
    RunSummaryDTO,
)
from src.api.services.eval_runs import RunRegistry
from src.eval.compare import compare_runs as _compare_runs_impl
from src.eval.config import load_config
from src.eval.runner import EvalRunner
from src.eval.schemas import CompareResult, EvalResult
from src.eval.storage import compute_run_id, list_runs, load_run

router = APIRouter(prefix="/api/eval", tags=["eval"])

# WHY: module-level attribute so monkeypatch.setattr("src.api.routes.eval.CONFIGS_DIR", ...)
# works in tests. Route handlers read this name from module globals at call time.
CONFIGS_DIR = Path("configs/eval")


# --------------------------------------------------------------------------- #
# Dependency helper                                                            #
# --------------------------------------------------------------------------- #

def _get_registry(request: Request) -> RunRegistry:
    """Extract the shared RunRegistry from app.state.

    PATTERN: Thin helper matching the get_backend pattern in dependencies.py.
    Routes call _get_registry(request) to stay testable and explicit.

    WHY lazy init: Starlette's TestClient does not run the lifespan when used
    outside a context manager (as in the test fixture). Lazy init guarantees
    a registry exists even when the lifespan startup hook hasn't fired —
    the registry is still correct because it's a plain in-memory dict.
    """
    if not hasattr(request.app.state, "run_registry"):
        # PATTERN: thread-safe because attribute assignment on a single object is
        # atomic in CPython; worst case two threads both create a registry and one
        # overwrites the other — acceptable for test scenarios.
        request.app.state.run_registry = RunRegistry()
    return request.app.state.run_registry


# --------------------------------------------------------------------------- #
# Background worker                                                            #
# --------------------------------------------------------------------------- #

def _run_eval_in_background(
    config_name: str,
    run_id: str,
    registry: RunRegistry,
) -> None:
    """Synchronous worker invoked via BackgroundTasks.

    WHY sync (not async): EvalRunner is CPU/IO-mixed and calls blocking LLM
    APIs. Sync BackgroundTasks workers are run in a threadpool by Starlette,
    keeping the event loop free. An async worker would block the loop.

    Pipeline position: DISPATCH — called once per POST /api/eval/run,
    runs the full EvalRunner lifecycle, then marks the run done/failed
    in the registry.
    """
    # WHY live import of storage: the tmp_eval_runs fixture reloads
    # src.eval.storage after setting EVAL_RUNS_DIR. Importing at call time
    # ensures we see the reloaded module attribute value.
    import src.eval.storage as _storage

    cfg_path = CONFIGS_DIR / f"{config_name}.yaml"
    cfg = load_config(cfg_path)

    # PATTERN: respect EVAL_LLM_OVERRIDE_DUMMY=1 — same logic as cli._cmd_run.
    # This makes the test harness fast (no real LLM calls).
    llm_override = None
    judge_llm_override = None
    if os.getenv("EVAL_LLM_OVERRIDE_DUMMY") == "1":
        from src.eval.cli import _DummyLLM
        dummy = _DummyLLM()
        llm_override = dummy
        judge_llm_override = dummy

    runner = EvalRunner(
        cfg,
        config_path=cfg_path,
        llm_override=llm_override,
        judge_llm_override=judge_llm_override,
        on_progress=lambda done, total: registry.update_progress(run_id, done),
        # WHY run_id_override: we pre-computed the run_id at submit time so the
        # registry could be populated before the run starts. Passing it here
        # ensures EvalRunner saves to the same directory the status endpoint expects.
        run_id_override=run_id,
    )

    try:
        runner.run()
        registry.mark_completed(run_id)
    except Exception as exc:
        registry.mark_failed(run_id, str(exc))


# --------------------------------------------------------------------------- #
# GET /api/eval/configs                                                        #
# --------------------------------------------------------------------------- #

@router.get(
    "/configs",
    response_model=list[str],
    summary="List available eval config names",
)
def list_configs() -> list[str]:
    """Return the names (without .yaml extension) of all configs in configs/eval/.

    The client uses this to populate the "New eval run" dropdown — no need
    to expose the full YAML content at this level.
    """
    # WHY: CONFIGS_DIR is read as a module global so monkeypatching works.
    if not CONFIGS_DIR.exists():
        return []
    return [p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml"))]


# --------------------------------------------------------------------------- #
# POST /api/eval/run                                                           #
# --------------------------------------------------------------------------- #

@router.post(
    "/run",
    response_model=RunSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an eval run (non-blocking)",
)
def submit_run(
    body: RunSubmitRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> RunSubmitResponse:
    """Queue an eval run and return immediately with a run_id to poll.

    PATTERN: async job — the route validates the config exists, pre-computes
    the run_id (same algorithm as EvalRunner so they agree on the directory
    name), registers the run in the registry as "queued", dispatches via
    BackgroundTasks, then returns 202. The client polls /runs/{run_id}/status.

    WHY pre-compute run_id: the registry must track the run BEFORE it
    starts, so the status endpoint can return "queued" immediately after
    submission. EvalRunner accepts run_id_override to use the same id.
    """
    config_name = body.config_name
    cfg_path = CONFIGS_DIR / f"{config_name}.yaml"

    # 404 if the config file doesn't exist.
    if not cfg_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config '{config_name}' not found in {CONFIGS_DIR}.",
        )

    # Pre-compute run_id using the same algorithm as EvalRunner.run().
    started_at = datetime.now(timezone.utc)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    run_id = compute_run_id(config_name, started_at, git_sha)

    # Register before dispatch so status can return "queued" immediately.
    registry = _get_registry(request)
    # WHY n_total=0: we don't know question count until the runner loads datasets.
    # update_progress transitions the entry to "running" on first call.
    registry.register(run_id, n_total=0)

    # Dispatch the synchronous worker via BackgroundTasks (runs in threadpool).
    background_tasks.add_task(_run_eval_in_background, config_name, run_id, registry)

    return RunSubmitResponse(run_id=run_id, status="queued")


# --------------------------------------------------------------------------- #
# GET /api/eval/runs                                                           #
# --------------------------------------------------------------------------- #

@router.get(
    "/runs",
    response_model=list[RunSummaryDTO],
    summary="List all completed eval runs",
)
def list_eval_runs() -> list[RunSummaryDTO]:
    """Return all persisted runs sorted by started_at descending.

    WHY disk-only: completed runs are on disk; in-flight runs lack aggregated
    metrics so they aren't useful in the list view. The status endpoint covers
    in-flight monitoring.
    """
    # WHY live import of list_runs: called via the function (which reads
    # EVAL_RUNS_DIR from the module global at call time), so the reloaded
    # module attribute is always used correctly.
    runs = list_runs()
    result: list[RunSummaryDTO] = []
    for meta in runs:
        # Compute headline metric: recall_at_5 mean if available, else None.
        headline: float | None = None
        try:
            run_data = load_run(meta.run_id)
            aggregated = run_data["aggregated"]
            for agg in aggregated:
                if agg.metric_name == "recall_at_5" and agg.dataset is not None:
                    headline = agg.mean
                    break
        except Exception:
            # TRADE-OFF: if a run is partially written, skip its headline metric.
            pass
        result.append(
            RunSummaryDTO(
                run_id=meta.run_id,
                config_name=meta.config_name,
                started_at=meta.started_at,
                finished_at=meta.finished_at,
                n_questions=meta.n_questions,
                n_errors=meta.n_errors,
                headline_metric=headline,
            )
        )
    return result


# --------------------------------------------------------------------------- #
# GET /api/eval/runs/{run_id}                                                  #
# --------------------------------------------------------------------------- #

@router.get(
    "/runs/{run_id}",
    response_model=RunDetailDTO,
    summary="Get full detail for one eval run",
)
def get_run(run_id: str) -> RunDetailDTO:
    """Return metadata + aggregated metrics + cost for a single run.

    Results are paginated separately — this response is bounded regardless
    of how many questions the run evaluated.
    """
    try:
        run_data = load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    meta = run_data["metadata"]
    aggregated = run_data["aggregated"]
    cost = run_data["cost"]
    results = run_data["results"]

    agg_dtos = [
        AggregatedMetricDTO(
            metric_name=a.metric_name,
            dataset=a.dataset,
            mean=a.mean,
            ci_low=a.ci_low,
            ci_high=a.ci_high,
            n=a.n,
        )
        for a in aggregated
    ]

    return RunDetailDTO(
        metadata=meta,
        aggregated=agg_dtos,
        cost=cost,
        n_results=len(results),
    )


# --------------------------------------------------------------------------- #
# GET /api/eval/runs/{run_id}/results                                          #
# --------------------------------------------------------------------------- #

@router.get(
    "/runs/{run_id}/results",
    summary="Paginated per-question results for a run",
)
def get_run_results(
    run_id: str,
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page (max 200)"),
) -> dict:
    """Return a paginated slice of per-question results.

    WHY paginated: a 200-question run has ~200KB of generated text + metrics.
    Streaming the whole payload on every page load is wasteful; pagination
    lets the UI load one screenful at a time.

    Returns:
        Dict with keys: items, page, page_size, total.
    """
    try:
        run_data = load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    results: list[EvalResult] = run_data["results"]
    total = len(results)

    # WHY 0-indexed slice: page=1 → items[0:page_size], page=2 → items[page_size:2*page_size].
    start = (page - 1) * page_size
    end = start + page_size
    page_items = results[start:end]

    dtos = [
        EvalResultDTO(
            question_id=r.question_id,
            dataset=r.dataset,
            generated_answer=r.generated_answer,
            metrics=r.metrics,
            error=r.error,
        )
        for r in page_items
    ]

    return {"items": [d.model_dump() for d in dtos], "page": page, "page_size": page_size, "total": total}


# --------------------------------------------------------------------------- #
# GET /api/eval/runs/{run_id}/results/{question_id}                            #
# --------------------------------------------------------------------------- #

@router.get(
    "/runs/{run_id}/results/{question_id}",
    response_model=EvalResult,
    summary="Get full EvalResult for one question",
)
def get_question_result(run_id: str, question_id: str) -> EvalResult:
    """Return the full EvalResult (including retrieved_chunks, metric_details).

    WHY separate endpoint: the list view uses EvalResultDTO (slim), which omits
    the large fields. This endpoint returns the full schema for the detail drawer.
    """
    try:
        run_data = load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    results: list[EvalResult] = run_data["results"]
    for r in results:
        if r.question_id == question_id:
            return r

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Question '{question_id}' not found in run '{run_id}'.",
    )


# --------------------------------------------------------------------------- #
# GET /api/eval/runs/{run_id}/status                                           #
# --------------------------------------------------------------------------- #

@router.get(
    "/runs/{run_id}/status",
    response_model=RunStatusDTO,
    summary="Poll the status of an in-progress or completed eval run",
)
def get_run_status(run_id: str, request: Request) -> RunStatusDTO:
    """Return the current status and progress for a run.

    WHY two-phase lookup:
      1. Check in-memory registry (covers queued/running/recently-completed).
      2. If not in registry, check disk (covers runs from previous server
         restarts that the registry has evicted).
      3. 404 if neither source has the run.
    """
    registry = _get_registry(request)
    entry = registry.get(run_id)

    if entry is not None:
        progress = (
            (entry.n_completed / entry.n_total)
            if entry.n_total > 0 and entry.status == "completed"
            else (1.0 if entry.status == "completed" else 0.0)
        )
        return RunStatusDTO(
            run_id=run_id,
            status=entry.status,
            progress=progress,
            n_completed=entry.n_completed,
            n_total=entry.n_total,
            error_message=entry.error_message,
        )

    # Fallback: check disk for runs not in the registry (e.g. post-restart).
    try:
        run_data = load_run(run_id)
        meta = run_data["metadata"]
        return RunStatusDTO(
            run_id=run_id,
            status="completed",
            progress=1.0,
            n_completed=meta.n_questions,
            n_total=meta.n_questions,
            error_message=None,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )


# --------------------------------------------------------------------------- #
# GET /api/eval/compare                                                        #
# --------------------------------------------------------------------------- #

@router.get(
    "/compare",
    response_model=CompareResult,
    summary="Compare two eval runs (delta metrics + per-question diff)",
)
def compare_runs(
    a: str = Query(..., description="run_id of run A"),
    b: str = Query(..., description="run_id of run B"),
) -> CompareResult:
    """Compute metric deltas and per-question diffs between two runs.

    TRADE-OFF: 409 on eval-set version mismatch rather than silently computing
    a misleading comparison. Different eval-set versions mean different questions,
    so per-question deltas are meaningless. Surfaces this clearly to the UI.
    """
    try:
        return _compare_runs_impl(a, b)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        # WHY 409 (Conflict): the comparison is a logical conflict — the two
        # runs are not comparable because they evaluated different question sets.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
