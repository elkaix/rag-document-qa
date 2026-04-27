/**
 * EvalPage — sub-route dispatcher for /eval/*.
 *
 * RAG Pipeline Position:
 *   This is an EVALUATION layer component — it sits outside the ingestion/
 *   retrieval/generation pipeline and provides tooling to measure how well
 *   that pipeline performs.
 *
 *   /eval                  → RunsList   (browse & manage evaluation runs)
 *   /eval/runs/:runId      → RunDetail  (per-question breakdown + metrics)
 *   /eval/compare?a=&b=    → CompareView (side-by-side metric comparison)
 *
 * WHY nested <Routes>:
 *   The parent router registers /eval/* (wildcard). React Router strips the
 *   matched prefix (/eval) and passes the remainder to this component's own
 *   <Routes>, which resolves the sub-path. This avoids co-locating eval
 *   routing logic in the root router and keeps eval concerns self-contained.
 *
 * PATTERN: Page components are thin route dispatchers. Business logic lives
 *          inside the individual feature components (RunsList, RunDetail,
 *          CompareView), not here.
 */

import { Routes, Route } from "react-router";

import { RunsList } from "@/components/eval/runs-list";
import { RunDetail } from "@/components/eval/run-detail";
import { CompareView } from "@/components/eval/compare-view";

export function EvalPage() {
  return (
    <Routes>
      {/* /eval → full list of evaluation runs */}
      <Route index element={<RunsList />} />
      {/* /eval/runs/:runId → per-question detail for one run */}
      <Route path="runs/:runId" element={<RunDetail />} />
      {/* /eval/compare?a=<id>&b=<id> → side-by-side metric comparison */}
      <Route path="compare" element={<CompareView />} />
    </Routes>
  );
}
