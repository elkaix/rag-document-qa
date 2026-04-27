# RAG Eval Harness — Sub-plan 1C: API + Frontend `/eval` Page

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Sub-plans 1A + 1B complete (CLI works locally).

**Goal:** Surface the eval system in the existing FastAPI + React app. After this sub-plan, a developer can browse runs, drill into per-question results, compare two runs side-by-side, and trigger a new run — all from the UI.

**Architecture:** New `src/api/routes/eval.py` exposes runs/compare/configs over REST; long-running runs are dispatched via FastAPI `BackgroundTasks` with status polling. New React route `/eval/*` mounts three views (RunsList, RunDetail, CompareView) plus a NewEvalRunDialog, all using TanStack Query against the new endpoints.

**Tech Stack:** FastAPI (existing), Pydantic v2, TanStack Query (existing), Tailwind + shadcn/ui (existing), `recharts` (NEW — small chart lib), React Router v7 (existing).

**Spec:** [`docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md`](../specs/2026-04-26-rag-eval-harness-phase-1-design.md) §10–11.

---

## File Structure

**New backend files:**

| Path | Responsibility |
|------|----------------|
| `src/api/routes/eval.py` | All `/api/eval/*` endpoints. |
| `src/api/services/eval_runs.py` | In-process registry of in-flight runs (run_id → status / progress); thread-safe. |
| `src/api/schemas/eval.py` | Request/response DTOs for the API (separate from internal `eval.schemas`). |

**New frontend files:**

| Path | Responsibility |
|------|----------------|
| `frontend/src/api/eval.ts` | Typed client + TanStack Query hooks. |
| `frontend/src/pages/EvalPage.tsx` | Route `/eval/*` shell + sub-route dispatch. |
| `frontend/src/components/eval/RunsList.tsx` | Sortable/filterable table of runs. |
| `frontend/src/components/eval/RunDetail.tsx` | Per-run metrics chart + per-question table. |
| `frontend/src/components/eval/CompareView.tsx` | Side-by-side bars + significance markers + diff. |
| `frontend/src/components/eval/NewEvalRunDialog.tsx` | Config picker + submit + status poll. |
| `frontend/src/components/eval/MetricBars.tsx` | Reusable bar chart with CI whiskers. |
| `frontend/src/components/eval/SignificanceBadge.tsx` | "★ p<0.05" badge component. |

**Modified files:**

| Path | Change |
|------|--------|
| `src/api/main.py` | `app.include_router(eval_router)`. |
| `frontend/src/App.tsx` | Add `<Route path="/eval/*" element={<EvalPage />} />`. |
| `frontend/src/components/layout/sidebar.tsx` | Add "Evaluation" nav link. |
| `frontend/package.json` | Add `recharts` dep. |

**Tests:**
`tests/test_api_eval_routes.py` (FastAPI TestClient), plus React Testing Library tests for each frontend component.

---

## Task 1 — Add `recharts` to frontend deps

**Files:** `frontend/package.json`

- [ ] `cd frontend && bun add recharts` (or `npm install recharts` if bun not installed). Verify lockfile updates.
- [ ] `cd frontend && bun run build` succeeds.
- [ ] Commit: `chore(frontend): add recharts dep for eval metric charts`.

---

## Task 2 — Backend DTOs (`src/api/schemas/eval.py`)

**Files:** `src/api/schemas/eval.py`

**DTOs:**
```python
class RunSummaryDTO(BaseModel):
    run_id: str
    config_name: str
    started_at: datetime
    finished_at: datetime
    n_questions: int
    n_errors: int
    headline_metric: float | None  # recall_at_5 mean, or None if absent

class AggregatedMetricDTO(BaseModel):
    metric_name: str
    dataset: str | None
    mean: float
    ci_low: float
    ci_high: float
    n: int

class RunDetailDTO(BaseModel):
    metadata: RunMetadata        # reuse from src/eval/schemas.py
    aggregated: list[AggregatedMetricDTO]
    cost: dict[str, float]
    n_results: int               # results paginated separately

class EvalResultDTO(BaseModel):
    # Slim version for the UI table — full retrieved_chunks excluded from list view
    question_id: str
    dataset: str
    generated_answer: str
    metrics: dict[str, float]
    error: str | None

class RunSubmitRequest(BaseModel):
    config_name: str             # must match a file in configs/eval/

class RunSubmitResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]

class RunStatusDTO(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float              # 0.0 - 1.0
    n_completed: int
    n_total: int
    error_message: str | None
```

**Tests** (`tests/test_api_eval_routes.py` will exercise these as part of route tests).

Commit: `feat(api): add eval API DTOs`.

---

## Task 3 — In-process run registry (`src/api/services/eval_runs.py`)

**Files:** `tests/test_api_eval_runs_service.py`, `src/api/services/eval_runs.py`

**API:**
```python
@dataclass
class RunStatus:
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    n_completed: int
    n_total: int
    error_message: str | None = None

class RunRegistry:
    def register(self, run_id: str, n_total: int) -> None
    def update_progress(self, run_id: str, n_completed: int) -> None
    def mark_completed(self, run_id: str) -> None
    def mark_failed(self, run_id: str, error: str) -> None
    def get(self, run_id: str) -> RunStatus | None
    def list_active(self) -> list[RunStatus]
```

**Implementation notes:**
- Backed by an internal `dict[str, RunStatus]` guarded by `threading.Lock` (FastAPI runs route handlers in a threadpool).
- Singleton instance attached to `app.state.run_registry` at startup.
- Completed runs auto-evicted after 1 hour (so the registry doesn't grow unbounded); inspectable runs live in `eval_runs/` on disk.

**Test cases:**
- Register → update_progress → mark_completed transitions status correctly.
- Concurrent updates from threads remain consistent (use `concurrent.futures.ThreadPoolExecutor` with 10 workers updating the same run).
- Auto-eviction after artificial old timestamp.

Commit: `feat(api): add thread-safe in-process eval run registry`.

---

## Task 4 — `src/api/routes/eval.py`

**Files:** `tests/test_api_eval_routes.py`, `src/api/routes/eval.py`

**Endpoints** (all use `Annotated[..., Depends(get_backend)]` per project FastAPI conventions):

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| GET    | `/api/eval/configs` | `list[str]` | Names of `configs/eval/*.yaml`. |
| POST   | `/api/eval/run` | `RunSubmitResponse` | Body `RunSubmitRequest`. Validates config exists, computes run_id, registers in registry, dispatches via `BackgroundTasks`. Returns 202 with `status="queued"`. |
| GET    | `/api/eval/runs` | `list[RunSummaryDTO]` | Sorted descending by `started_at`. |
| GET    | `/api/eval/runs/{run_id}` | `RunDetailDTO` | 404 if not found. |
| GET    | `/api/eval/runs/{run_id}/results` | `Page[EvalResultDTO]` | Paginated; `?page=1&page_size=50` query params. |
| GET    | `/api/eval/runs/{run_id}/results/{question_id}` | `EvalResult` (full) | Full record incl. retrieved_chunks. 404 if not found. |
| GET    | `/api/eval/runs/{run_id}/status` | `RunStatusDTO` | For polling. |
| GET    | `/api/eval/compare?a={id_a}&b={id_b}` | `CompareResult` | 409 on eval-set version mismatch. |

**Background task implementation:**
```python
async def _run_eval_in_background(config_name, run_id, registry):
    config = load_config(Path("configs/eval") / f"{config_name}.yaml")
    runner = EvalRunner(config, on_progress=lambda done, total: registry.update_progress(run_id, done))
    try:
        await asyncio.get_event_loop().run_in_executor(None, runner.run)
        registry.mark_completed(run_id)
    except Exception as e:
        registry.mark_failed(run_id, str(e))
```

(`EvalRunner` from 1B gets an optional `on_progress` callback.)

**Test cases (FastAPI TestClient):**
- `GET /api/eval/configs` returns `["baseline"]` after fixture creates baseline.yaml.
- `POST /api/eval/run` with valid config returns 202, run_id; subsequent `GET .../status` reaches `completed`.
- `POST` with unknown config returns 404.
- `GET .../runs` lists the completed run.
- `GET .../runs/{id}` returns metadata + aggregated.
- `GET .../runs/{id}/results?page=1&page_size=2` returns 2 of N.
- `GET .../runs/{missing_id}` returns 404.
- `GET .../compare?a=&b=` returns CompareResult; mismatched eval set versions returns 409.

Register the router in `src/api/main.py`: `app.include_router(eval_router, prefix="/api/eval", tags=["eval"])`.

Commit: `feat(api): add /api/eval/* routes for runs, results, compare, configs`.

---

## Task 5 — Frontend: API client + types

**Files:** `frontend/src/api/eval.ts`

**Contents:**
- TypeScript interfaces mirroring the backend DTOs (RunSummary, AggregatedMetric, RunDetail, EvalResult, CompareResult, MetricDelta).
- Fetch helpers: `listRuns()`, `getRun(id)`, `getRunResults(id, page, pageSize)`, `getRunResult(id, qid)`, `submitRun(configName)`, `getRunStatus(id)`, `compareRuns(idA, idB)`, `listConfigs()`.
- TanStack Query hooks: `useRunsList()`, `useRun(id)`, `useRunResults(id)`, `useRunStatus(id, { refetchInterval: 1000 })`, `useCompareRuns(a, b)`, `useConfigs()`, `useSubmitRun()` (mutation).

Commit: `feat(frontend): add eval API client and TanStack Query hooks`.

---

## Task 6 — `MetricBars.tsx` (reusable chart with CI whiskers)

**Files:** `frontend/src/components/eval/MetricBars.tsx`

**Props:** `metrics: AggregatedMetricDTO[]`, optional `comparison?: { b: AggregatedMetricDTO[]; deltas: MetricDelta[] }`.

**Implementation:** Use `recharts` `<BarChart>` with `<ErrorBar>` for the CI; if `comparison` is provided, render two bars per metric and overlay a "★" annotation on bars where `delta.significant`. Tailwind for layout.

**Tests** (React Testing Library): renders correct number of bars; significance badge appears when `comparison.deltas[].significant === true`.

Commit: `feat(frontend): add MetricBars chart component for aggregated metrics`.

---

## Task 7 — `RunsList.tsx`

**Files:** `frontend/src/components/eval/RunsList.tsx`

**Behavior:**
- Fetches via `useRunsList()`.
- Sortable columns: `started_at`, `config_name`, `n_questions`, `n_errors`, `headline_metric`. Default sort: `started_at` desc.
- Search box filters by `config_name` substring (debounced 300ms).
- "New Run" button opens `NewEvalRunDialog`.
- Row click navigates to `/eval/runs/{run_id}`.
- Multi-select with checkboxes; "Compare Selected" button enabled when exactly 2 selected → navigates to `/eval/compare?a=&b=`.

Commit: `feat(frontend): add RunsList with sort, filter, multi-select compare`.

---

## Task 8 — `RunDetail.tsx`

**Files:** `frontend/src/components/eval/RunDetail.tsx`

**Behavior:**
- Top: header (run_id, config_name, started_at, n_questions, n_errors, total cost).
- `MetricBars` for aggregated (combined view; tabs to switch dataset filter).
- Per-stage latency chart (separate `recharts` `<BarChart>`).
- Cost summary card.
- Bottom: paginated per-question table (50 rows/page) using `useRunResults`. Columns: `question_id` (truncated), `dataset`, `error`, score per metric. Click row → expands to show full `EvalResult` (question, gold, generated, retrieved chunks, judge reasoning).
- Loading + empty + error states.

Commit: `feat(frontend): add RunDetail with metrics chart and per-question table`.

---

## Task 9 — `CompareView.tsx`

**Files:** `frontend/src/components/eval/CompareView.tsx`

**Behavior:**
- Reads `?a=&b=` from URL via `useSearchParams`.
- `useCompareRuns(a, b)`.
- Header: two run summaries side-by-side.
- `MetricBars` in comparison mode (two bars per metric, ★ on significant deltas).
- Two cards: "Top Wins" (largest positive deltas) and "Top Regressions" (largest negative). Each shows up to 5 questions with `(gold, a_answer, a_score) → (b_answer, b_score)`.
- Handle 409 (version mismatch) with a clear error banner.

Commit: `feat(frontend): add CompareView with side-by-side bars and per-question diff`.

---

## Task 10 — `NewEvalRunDialog.tsx`

**Files:** `frontend/src/components/eval/NewEvalRunDialog.tsx`

**Behavior:**
- Modal with shadcn/ui `<Dialog>`.
- Config picker: `<Select>` populated by `useConfigs()`.
- "Start Run" button → `useSubmitRun().mutate({config_name})`.
- On success: closes dialog, opens a smaller "Run in progress" toast that polls `useRunStatus(run_id, { refetchInterval: 1000 })` and shows progress bar + percent. On `completed`, the toast becomes a button "View Run" that navigates to `/eval/runs/{run_id}`.

Commit: `feat(frontend): add NewEvalRunDialog with config picker and progress toast`.

---

## Task 11 — `EvalPage.tsx` + routing + sidebar nav

**Files:** `frontend/src/pages/EvalPage.tsx`, `frontend/src/App.tsx`, `frontend/src/components/layout/sidebar.tsx`

**`EvalPage`:** sub-route dispatcher.

```tsx
<Routes>
  <Route index element={<RunsList />} />
  <Route path="runs/:runId" element={<RunDetail />} />
  <Route path="compare" element={<CompareView />} />
</Routes>
```

**`App.tsx`:** add `<Route path="/eval/*" element={<EvalPage />} />`.

**`sidebar.tsx`:** add a `<NavLink to="/eval">Evaluation</NavLink>` near the existing "Documents" link, with the same active-styling pattern.

**Manual smoke test (documented for the executing engineer):**
1. `python -m src.api.main` and `cd frontend && bun run dev`.
2. Open `http://localhost:3000/eval` — RunsList should render (empty initially).
3. Click "New Run" → pick `baseline` → submit → wait for completion → click "View Run" → RunDetail renders.
4. Trigger a second run with a different config (create `configs/eval/topk_10.yaml`) → from RunsList check both → "Compare Selected" → CompareView renders bars and (for many metrics) a `★` badge.

Commit: `feat(frontend): wire EvalPage routes and add Evaluation sidebar link`.

---

## Sub-plan 1C Completion Checklist

- [ ] Backend tests pass: `python -m pytest tests/test_api_eval_*.py -v`.
- [ ] Frontend tests pass: `cd frontend && bun run test`.
- [ ] Frontend builds: `cd frontend && bun run build`.
- [ ] Manual smoke test (above) succeeds end-to-end.
- [ ] Sidebar shows "Evaluation" link.
- [ ] No new file in `src/api/routes/eval.py` exceeds 250 lines (split if needed).
- [ ] `git status` clean.

Once green: proceed to **Sub-plan 1D — Observability**.
