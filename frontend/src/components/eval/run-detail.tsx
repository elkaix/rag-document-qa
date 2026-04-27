/**
 * RunDetail — single eval run page.
 *
 * Frontend Position:
 *   /eval/runs/:runId → [RunDetail] → expand row → load full result → show judge details
 *
 * WHY one file: five sections (header, dataset filter, metrics chart, cost
 * summary, paginated question table) all share a single runId and a single
 * expand-row state. Splitting them into files adds indirection with no gain.
 *
 * PATTERN: Loading / error / not-found handled before reaching render so
 *   the happy-path JSX at the bottom reads cleanly without nested guards.
 */

import { useState } from "react";
import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronDown, ChevronRight } from "lucide-react";
import { useNavigate } from "react-router";

import { useRun, useRunResults, getRunResult } from "@/api/eval";
import type { EvalResultRow } from "@/api/eval";
import { MetricBars } from "@/components/eval/metric-bars";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PAGE_SIZE = 50;

// Preferred metric columns: show in this order when present, cap at 4.
// WHY: an uncapped list causes horizontal overflow on typical viewports.
const PREFERRED_METRIC_COLS = [
  "recall_at_5",
  "faithfulness",
  "answer_correctness",
  "refusal_correctness",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** ISO timestamp → short locale string. */
const dateFmt = new Intl.DateTimeFormat(undefined, {
  dateStyle: "short",
  timeStyle: "short",
});

function formatDate(iso: string): string {
  try {
    return dateFmt.format(new Date(iso));
  } catch {
    return iso;
  }
}

/**
 * Format a USD amount to 4 decimal places (e.g. $0.0001).
 *
 * WHY 4 decimal places: LLM inference costs per query are often sub-cent.
 * $0.001 precision loses meaningful differences between cheap and mid-tier
 * models. 4 decimals catches all common pricing granularity.
 */
function formatUSD(value: number): string {
  return `$${value.toFixed(4)}`;
}

/**
 * Truncate a question_id for table display — UUIDs are 36 chars, too wide.
 */
function shortId(id: string): string {
  return id.length > 12 ? id.slice(0, 12) + "…" : id;
}

/**
 * Derive the metric columns to show in the per-question table.
 *
 * WHY derive from results instead of hardcode: the metrics in a run
 * depend on the eval config. Using the first few rows ensures we display
 * what's actually present rather than empty columns.
 *
 * Cap at 4 columns to prevent horizontal overflow.
 */
function deriveMetricCols(items: EvalResultRow[]): string[] {
  if (items.length === 0) return [];
  const present = new Set(Object.keys(items[0].metrics));
  const preferred = PREFERRED_METRIC_COLS.filter((m) => present.has(m));
  if (preferred.length > 0) return preferred.slice(0, 4);
  // Fallback: first 4 keys from the first row.
  return Object.keys(items[0].metrics).slice(0, 4);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Loading skeleton — mirrors the visual density of the loaded page. */
function RunDetailSkeleton() {
  return (
    <div className="flex flex-col gap-6 p-6">
      <Skeleton className="h-8 w-40" />
      <div className="grid grid-cols-2 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-6 w-full" />
        ))}
      </div>
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-48 w-full" />
    </div>
  );
}

/**
 * HeaderCard — run provenance at a glance.
 *
 * WHY 2-column grid: pairs label+value compactly without a full table.
 * git_sha in monospace because it's a hash meant to be compared visually,
 * not read as prose.
 */
function HeaderCard({
  meta,
}: {
  meta: {
    run_id: string;
    config_name: string;
    started_at: string;
    finished_at: string;
    n_questions: number;
    n_errors: number;
    git_sha: string;
  };
}) {
  const fields: Array<[string, React.ReactNode]> = [
    ["Run ID", <span className="font-mono text-xs">{meta.run_id}</span>],
    ["Config", meta.config_name],
    ["Started", formatDate(meta.started_at)],
    ["Finished", formatDate(meta.finished_at)],
    ["Questions", meta.n_questions],
    [
      "Errors",
      <span className={meta.n_errors > 0 ? "text-destructive font-medium" : undefined}>
        {meta.n_errors}
      </span>,
    ],
    ["Git SHA", <span className="font-mono text-xs">{meta.git_sha.slice(0, 12)}</span>],
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Run Detail</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
          {fields.map(([label, value]) => (
            <div key={label} className="flex gap-2">
              <dt className="text-muted-foreground shrink-0 w-24">{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

/**
 * CostCard — cost and token totals for the run.
 *
 * WHY separate card from the header: cost summary is a different concern
 * from run provenance. Engineers debugging quality look at the header;
 * engineers managing budget look at cost. Keeping them separate avoids
 * visual clutter and lets each card be skipped independently.
 *
 * Key mapping:
 *   total_usd, mean_usd_per_query → from aggregate_costs()
 *   total_prompt, total_completion → from aggregate_tokens()
 */
function CostCard({ cost }: { cost: Record<string, number> }) {
  const totalUsd = cost["total_usd"] ?? 0;
  const meanUsd = cost["mean_usd_per_query"] ?? 0;
  const totalPrompt = cost["total_prompt"] ?? 0;
  const totalCompletion = cost["total_completion"] ?? 0;

  const fields: Array<[string, string]> = [
    ["Total cost", formatUSD(totalUsd)],
    ["Mean / query", formatUSD(meanUsd)],
    ["Prompt tokens", totalPrompt.toLocaleString()],
    ["Completion tokens", totalCompletion.toLocaleString()],
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cost Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
          {fields.map(([label, value]) => (
            <div key={label} className="flex gap-2">
              <dt className="text-muted-foreground shrink-0 w-36">{label}</dt>
              <dd className="tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

/**
 * ExpandedDetail — lazy-fetched full result for an expanded table row.
 *
 * PATTERN: The query is only enabled when this component mounts (i.e.
 * when expandedId === qid). Collapsing the row unmounts the component
 * and the query result is cached — re-expanding the same row is instant.
 *
 * WHY not store the result in local state: TanStack Query's cache already
 * does this for us. No useState needed; no prop drilling.
 */
function ExpandedDetail({
  runId,
  questionId,
}: {
  runId: string;
  questionId: string;
}) {
  const { data, isLoading, isError } = useQuery({
    // WHY both runId and questionId in key: ensures different runs' results
    // for the same question don't collide in the cache.
    queryKey: ["eval-run-result", runId, questionId],
    queryFn: () => getRunResult(runId, questionId),
  });

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-4">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    );
  }

  if (isError) {
    return (
      <p className="p-4 text-sm text-destructive">
        Failed to load question detail.
      </p>
    );
  }

  // TRADE-OFF: getRunResult returns `unknown` because the backend's EvalResult
  // schema isn't mirrored in the frontend. We assert to a local interface here
  // rather than adding a full DTO — this component is the only consumer.
  const r = data as {
    question_id?: string;
    retrieved_chunks?: string[];
    generated_answer?: string;
    metric_details?: Record<string, unknown>;
    error?: string | null;
  } | null;

  if (!r) return null;

  return (
    <div className="flex flex-col gap-4 p-4 text-sm bg-muted/30 rounded-b-md">
      {r.error && (
        <div className="text-destructive">
          <span className="font-medium">Error:</span> {r.error}
        </div>
      )}

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Generated Answer</p>
        <p className="whitespace-pre-wrap">{r.generated_answer ?? "—"}</p>
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">
          Retrieved Chunks ({r.retrieved_chunks?.length ?? 0})
        </p>
        {(r.retrieved_chunks ?? []).map((chunk, i) => (
          <p key={i} className="text-xs text-muted-foreground border-l-2 pl-2 mb-1 line-clamp-3">
            {chunk}
          </p>
        ))}
      </div>

      {r.metric_details && Object.keys(r.metric_details).length > 0 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-1">Judge Reasoning</p>
          {Object.entries(r.metric_details).map(([metric, detail]) => (
            <div key={metric} className="mb-2">
              <p className="text-xs font-medium">{metric}</p>
              <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                {typeof detail === "string"
                  ? detail
                  : JSON.stringify(detail, null, 2)}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * RunDetail renders the full detail page for a single eval run.
 *
 * Five sections:
 *   1. Header card — run provenance
 *   2. Dataset filter + MetricBars chart
 *   3. Cost summary card
 *   4. Paginated per-question table with expand-row
 *   5. Pagination controls
 *
 * State:
 *   page          — current results page (1-indexed)
 *   selectedDs    — "all" or a dataset name for the metric chart filter
 *   expandedId    — question_id of the expanded table row (null = none)
 */
export function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();

  const [page, setPage] = useState(1);
  const [selectedDs, setSelectedDs] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // WHY useRun for metadata/aggregated/cost, useRunResults for paginated rows:
  // the run detail response is bounded regardless of question count;
  // rows are paginated separately so large runs don't ship all data at once.
  const {
    data: runDetail,
    isLoading: runLoading,
    isError: runError,
    error: runErrorObj,
  } = useRun(runId);

  const {
    data: pageResults,
    isLoading: resultsLoading,
  } = useRunResults(runId, page, PAGE_SIZE);

  // -------------------------------------------------------------------------
  // Loading state
  // -------------------------------------------------------------------------

  if (runLoading) return <RunDetailSkeleton />;

  // -------------------------------------------------------------------------
  // Error state
  // -------------------------------------------------------------------------

  if (runError) {
    const msg = runErrorObj instanceof Error ? runErrorObj.message : "Unknown error";
    const isNotFound = msg.includes("404") || msg.toLowerCase().includes("not found");
    return (
      <div className="p-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate("/eval")}
          className="mb-4 -ml-2"
        >
          <ChevronLeft className="mr-1 size-4" />
          Back to runs
        </Button>
        <div
          className={`rounded-md border px-4 py-3 text-sm ${
            isNotFound
              ? "border-muted text-muted-foreground"
              : "border-destructive/30 bg-destructive/10 text-destructive"
          }`}
        >
          {isNotFound
            ? `Run "${runId}" not found. It may have been deleted or the ID is invalid.`
            : `Failed to load run: ${msg}`}
        </div>
      </div>
    );
  }

  if (!runDetail) return <RunDetailSkeleton />;

  // -------------------------------------------------------------------------
  // Derived state — dataset filter and metric chart props
  // -------------------------------------------------------------------------

  const { metadata, aggregated, cost, n_results } = runDetail;

  // Discover distinct non-null dataset names from aggregated metrics.
  // WHY: dataset=null rows are the "combined" aggregates; only non-null
  //      values are real dataset names we want as filter buttons.
  const availableDatasets = [
    ...new Set(
      aggregated
        .map((m) => m.dataset)
        .filter((d): d is string => d !== null),
    ),
  ];

  // Filter metrics for the chart:
  //   "all"       → show combined rows (dataset === null)
  //   dataset name → show per-dataset rows for that dataset only
  const filteredMetrics =
    selectedDs === "all"
      ? aggregated.filter((m) => m.dataset === null)
      : aggregated.filter((m) => m.dataset === selectedDs);

  // -------------------------------------------------------------------------
  // Derived state — per-question table
  // -------------------------------------------------------------------------

  const items = pageResults?.items ?? [];
  const metricCols = deriveMetricCols(items);
  const totalPages = Math.ceil(n_results / PAGE_SIZE);

  // -------------------------------------------------------------------------
  // Handlers
  // -------------------------------------------------------------------------

  function toggleRow(qid: string) {
    setExpandedId((prev) => (prev === qid ? null : qid));
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Back navigation */}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => navigate("/eval")}
        className="-ml-2 self-start"
      >
        <ChevronLeft className="mr-1 size-4" />
        All runs
      </Button>

      {/* Section 1: Header card */}
      <HeaderCard meta={metadata} />

      {/* Section 2: Dataset filter + metric chart */}
      <Card>
        <CardHeader>
          <CardTitle>Aggregated Metrics</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {/*
           * Dataset filter pills — "All" shows combined (dataset=null) rows;
           * clicking a dataset name shows only that dataset's rows.
           *
           * WHY pill buttons over a <select>: pills make all options visible
           * at once and are faster to toggle for 2-4 datasets — the typical
           * range for this eval harness.
           */}
          {availableDatasets.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {["all", ...availableDatasets].map((ds) => (
                <button
                  key={ds}
                  type="button"
                  onClick={() => setSelectedDs(ds)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    selectedDs === ds
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  }`}
                >
                  {ds === "all" ? "All" : ds}
                </button>
              ))}
            </div>
          )}

          <MetricBars metrics={filteredMetrics} height={280} />
        </CardContent>
      </Card>

      {/* Section 3: Cost summary */}
      <CostCard cost={cost} />

      {/* Section 4: Per-question table */}
      <Card>
        <CardHeader>
          <CardTitle>
            Per-question Results
            <span className="ml-2 text-sm font-normal text-muted-foreground">
              ({n_results} total)
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {resultsLoading ? (
            <div className="flex flex-col gap-2 p-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  {/* Expand toggle — no label, just space */}
                  <TableHead className="w-8" />
                  <TableHead className="w-32 font-medium">Question ID</TableHead>
                  <TableHead className="w-32">Dataset</TableHead>
                  <TableHead>Error</TableHead>
                  {metricCols.map((col) => (
                    <TableHead key={col} className="text-right tabular-nums">
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((row) => (
                  <>
                    {/*
                     * PATTERN: key on question_id so React doesn't reuse DOM
                     * nodes across rows when the page changes. Using array index
                     * as key would cause stale expanded state after pagination.
                     */}
                    <TableRow
                      key={row.question_id}
                      className="cursor-pointer"
                      onClick={() => toggleRow(row.question_id)}
                      data-state={
                        expandedId === row.question_id ? "selected" : undefined
                      }
                    >
                      <TableCell className="pr-0">
                        {expandedId === row.question_id ? (
                          <ChevronDown className="size-4 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="size-4 text-muted-foreground" />
                        )}
                      </TableCell>
                      <TableCell
                        className="font-mono text-xs"
                        title={row.question_id}
                      >
                        {shortId(row.question_id)}
                      </TableCell>
                      <TableCell className="text-xs">{row.dataset}</TableCell>
                      <TableCell className="max-w-xs truncate text-xs text-destructive">
                        {row.error ?? ""}
                      </TableCell>
                      {metricCols.map((col) => (
                        <TableCell key={col} className="text-right tabular-nums text-xs">
                          {row.metrics[col] !== undefined
                            ? row.metrics[col].toFixed(3)
                            : "—"}
                        </TableCell>
                      ))}
                    </TableRow>

                    {/*
                     * PATTERN: Render ExpandedDetail only when this row is
                     * expanded. Mounting triggers the lazy getRunResult query;
                     * unmounting on collapse keeps the result cached via
                     * TanStack Query — re-expanding is instant.
                     *
                     * WHY a separate <tr>: we need the expanded content to
                     * span all table columns cleanly. A colSpan cell inside
                     * its own row is the only way to achieve this in HTML tables.
                     */}
                    {expandedId === row.question_id && runId && (
                      <TableRow key={`${row.question_id}-detail`}>
                        <TableCell
                          colSpan={4 + metricCols.length}
                          className="p-0"
                        >
                          <ExpandedDetail
                            runId={runId}
                            questionId={row.question_id}
                          />
                        </TableCell>
                      </TableRow>
                    )}
                  </>
                ))}

                {items.length === 0 && (
                  <TableRow>
                    <TableCell
                      colSpan={4 + metricCols.length}
                      className="py-8 text-center text-muted-foreground"
                    >
                      No results on this page.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Section 5: Pagination controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            Prev
          </Button>
          <span className="text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
