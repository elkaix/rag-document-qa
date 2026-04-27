/**
 * Eval API client + TanStack Query hooks.
 *
 * API Layer Position:
 *   React components → [hooks] → fetch → /api/eval/*
 *
 * Design notes:
 *   - Field names match the backend's snake_case JSON wire format, consistent
 *     with the convention in types.ts (e.g. doc_id, chunk_id, created_at).
 *   - Uses the same `request<T>` helper pattern as client.ts — a thin wrapper
 *     around fetch that throws on non-2xx with the backend's `detail` message.
 *   - useRunStatus polls every 1s by default; pass `refetchInterval: false`
 *     once a run hits "completed" / "failed" to stop polling.
 *   - Submitting a run invalidates the runs list so the new run shows
 *     up immediately.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

// ---------------------------------------------------------------------------
// Types — mirror the backend DTOs from src/api/schemas/eval.py and
// src/eval/schemas.py. Field names stay snake_case to match JSON wire format,
// consistent with how types.ts handles doc_id, chunk_id, created_at, etc.
// ---------------------------------------------------------------------------

export interface RunMetadata {
  run_id: string;
  config_name: string;
  config_path: string;
  git_sha: string;
  started_at: string; // ISO timestamp
  finished_at: string;
  env_hash: string;
  eval_set_versions: Record<string, string>;
  n_questions: number;
  n_errors: number;
  warnings: string[];
}

export interface RunSummary {
  run_id: string;
  config_name: string;
  started_at: string;
  finished_at: string;
  n_questions: number;
  n_errors: number;
  headline_metric: number | null;
}

export interface AggregatedMetric {
  metric_name: string;
  dataset: string | null;
  mean: number;
  ci_low: number;
  ci_high: number;
  n: number;
}

export interface RunDetail {
  metadata: RunMetadata;
  aggregated: AggregatedMetric[];
  cost: Record<string, number>;
  n_results: number;
}

export interface EvalResultRow {
  question_id: string;
  dataset: string;
  generated_answer: string;
  metrics: Record<string, number>;
  error: string | null;
}

export interface PageResults {
  items: EvalResultRow[];
  page: number;
  page_size: number;
  total: number;
}

export interface RunStatus {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  n_completed: number;
  n_total: number;
  error_message: string | null;
}

export interface MetricDelta {
  metric_name: string;
  dataset: string | null;
  a_mean: number;
  a_ci: [number, number];
  b_mean: number;
  b_ci: [number, number];
  delta: number;
  p_value: number;
  significant: boolean;
}

export interface CompareResult {
  run_a: RunMetadata;
  run_b: RunMetadata;
  deltas: MetricDelta[];
  per_question_diff: Array<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Fetch helpers — mirror the request<T> helper from client.ts: same base URL
// env var, same error contract (throws Error with backend's detail message).
// WHY: a separate request helper here avoids coupling this module to client.ts
//      while staying consistent — easier to read in isolation.
// ---------------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// API functions — one per backend endpoint.
// ---------------------------------------------------------------------------

export async function listConfigs(): Promise<string[]> {
  return request<string[]>("/api/eval/configs");
}

export async function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/api/eval/runs");
}

export async function getRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/api/eval/runs/${encodeURIComponent(runId)}`);
}

export async function getRunResults(
  runId: string,
  page = 1,
  pageSize = 50,
): Promise<PageResults> {
  const qs = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  return request<PageResults>(
    `/api/eval/runs/${encodeURIComponent(runId)}/results?${qs}`,
  );
}

export async function getRunResult(
  runId: string,
  questionId: string,
): Promise<unknown> {
  return request(
    `/api/eval/runs/${encodeURIComponent(runId)}/results/${encodeURIComponent(questionId)}`,
  );
}

export async function getRunStatus(runId: string): Promise<RunStatus> {
  return request<RunStatus>(
    `/api/eval/runs/${encodeURIComponent(runId)}/status`,
  );
}

export async function submitRun(
  configName: string,
): Promise<{ run_id: string; status: RunStatus["status"] }> {
  return request("/api/eval/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_name: configName }),
  });
}

export async function compareRuns(
  idA: string,
  idB: string,
): Promise<CompareResult> {
  const qs = new URLSearchParams({ a: idA, b: idB });
  return request<CompareResult>(`/api/eval/compare?${qs}`);
}

// ---------------------------------------------------------------------------
// TanStack Query hooks — one per query/mutation, matching the style in
// use-documents.ts and use-conversations.ts (named exports, no default).
// Query keys follow the existing pattern: ["noun"] or ["noun", param, ...].
// ---------------------------------------------------------------------------

export function useConfigs() {
  return useQuery({ queryKey: ["eval-configs"], queryFn: listConfigs });
}

export function useRunsList() {
  return useQuery({ queryKey: ["eval-runs"], queryFn: listRuns });
}

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: ["eval-run", runId],
    queryFn: () => getRun(runId!),
    enabled: !!runId,
  });
}

export function useRunResults(
  runId: string | undefined,
  page = 1,
  pageSize = 50,
) {
  return useQuery({
    queryKey: ["eval-run-results", runId, page, pageSize],
    queryFn: () => getRunResults(runId!, page, pageSize),
    enabled: !!runId,
  });
}

// PATTERN: refetchInterval drives live progress updates for in-flight runs.
//          The caller should pass `refetchInterval: false` once status reaches
//          "completed" or "failed" to stop polling and save network calls.
export function useRunStatus(
  runId: string | undefined,
  opts?: { refetchInterval?: number | false },
) {
  return useQuery({
    queryKey: ["eval-run-status", runId],
    queryFn: () => getRunStatus(runId!),
    enabled: !!runId,
    refetchInterval: opts?.refetchInterval ?? 1000,
  });
}

export function useCompareRuns(
  idA: string | undefined,
  idB: string | undefined,
) {
  return useQuery({
    queryKey: ["eval-compare", idA, idB],
    queryFn: () => compareRuns(idA!, idB!),
    enabled: !!idA && !!idB,
  });
}

export function useSubmitRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (configName: string) => submitRun(configName),
    // WHY: invalidate the runs list so the newly queued run appears
    //      immediately in the UI without a manual page refresh.
    onSuccess: () => qc.invalidateQueries({ queryKey: ["eval-runs"] }),
  });
}
