/**
 * CompareView — side-by-side comparison of two eval runs.
 *
 * Frontend Position:
 *   /eval/compare?a=<id>&b=<id> → [CompareView] → MetricBars (compare mode)
 *                                              → top wins / regressions cards
 *
 * WHY one file: the five sections (missing-params guard, 409 banner, run header,
 * metric chart, and top wins/regressions cards) all depend on a single
 * useCompareRuns call and share derived state. Splitting adds indirection with
 * no benefit.
 *
 * PATTERN: Loading / error / missing-params handled before reaching the happy
 * path so the final render block reads cleanly without nested ternaries.
 */

import React from "react";
import { Link, useSearchParams } from "react-router";

import { useCompareRuns } from "@/api/eval";
import type { AggregatedMetric, MetricDelta } from "@/api/eval";
import { MetricBars } from "@/components/eval/metric-bars";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// ---------------------------------------------------------------------------
// Local types
// ---------------------------------------------------------------------------

/**
 * Typed view of the per_question_diff rows from the compare API.
 *
 * WHY local interface over Record<string, unknown>:
 *   The backend emits this shape consistently (see src/eval/compare.py),
 *   but the TypeScript wire type is Array<Record<string, unknown>> to avoid
 *   importing a heavyweight DTO. We cast once here and access fields safely
 *   throughout this component.
 */
interface PerQuestionDiff {
  question_id: string;
  dataset: string;
  a_score: number;
  b_score: number;
  delta: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** ISO timestamp → short locale string — matches run-detail.tsx. */
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
 * Truncate a run_id / question_id for display — UUIDs are 36 chars, too wide.
 * Shows the first 12 characters followed by an ellipsis.
 */
function shortId(id: string): string {
  return id.length > 12 ? id.slice(0, 12) + "…" : id;
}

/**
 * Format a signed delta as "+0.50" or "-0.50".
 *
 * WHY explicit "+" prefix: without it, positive gains are visually
 * indistinguishable from neutral values at a glance. The sign makes
 * win / regression intent obvious.
 */
function fmtDelta(delta: number): string {
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`;
}

/**
 * Rebuild AggregatedMetric arrays from MetricDelta rows for one side of the
 * comparison.
 *
 * WHY this helper exists:
 *   The compare API returns MetricDelta rows (one per metric×dataset) that
 *   contain both sides' means and confidence intervals as flat fields.
 *   MetricBars expects two independent AggregatedMetric[] arrays. This helper
 *   reconstructs each side's array without a second API call.
 *
 * n=0 because the compare endpoint does not surface per-side question counts.
 * MetricBars does not use the `n` field for rendering.
 *
 * @param deltas - Array of MetricDelta objects from CompareResult.deltas.
 * @param side   - Which side's values to extract: "a" or "b".
 * @returns      AggregatedMetric array suitable for MetricBars.
 */
function mapAggregatedFromDeltas(
  deltas: MetricDelta[],
  side: "a" | "b",
): AggregatedMetric[] {
  return deltas.map((d) => ({
    metric_name: d.metric_name,
    dataset: d.dataset,
    mean: side === "a" ? d.a_mean : d.b_mean,
    ci_low: side === "a" ? d.a_ci[0] : d.b_ci[0],
    ci_high: side === "a" ? d.a_ci[1] : d.b_ci[1],
    // TRADE-OFF: n is not surfaced by the compare API. MetricBars does not
    // render n, so 0 is a safe placeholder — it won't mislead in the chart.
    n: 0,
  }));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Loading skeleton matching the visual density of the compare page. */
function CompareSkeleton() {
  return (
    <div className="flex flex-col gap-6 p-6">
      <Skeleton className="h-7 w-48" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="h-5 w-full" />
        ))}
      </div>
      <Skeleton className="h-72 w-full" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    </div>
  );
}

/**
 * RunSummaryPanel — compact summary of one run's provenance.
 *
 * WHY dl + grid layout: mirrors HeaderCard in run-detail.tsx so the visual
 * language stays consistent across the eval section. Engineers reading either
 * page see the same field layout.
 */
function RunSummaryPanel({
  label,
  run,
}: {
  label: "Run A" | "Run B";
  run: {
    run_id: string;
    config_name: string;
    started_at: string;
    n_questions: number;
    n_errors: number;
  };
}) {
  const fields: Array<[string, React.ReactNode]> = [
    [
      "Run ID",
      <span className="font-mono text-xs" title={run.run_id}>
        {shortId(run.run_id)}
      </span>,
    ],
    ["Config", run.config_name],
    ["Started", formatDate(run.started_at)],
    ["Questions", run.n_questions],
    [
      "Errors",
      <span
        className={
          run.n_errors > 0 ? "text-destructive font-medium" : undefined
        }
      >
        {run.n_errors}
      </span>,
    ],
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm">
          {fields.map(([field, value]) => (
            <div key={field} className="flex gap-2">
              <dt className="text-muted-foreground shrink-0 w-20">{field}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

/**
 * DiffCard — renders either the top wins or top regressions table.
 *
 * WHY separate component: the wins and regressions cards have identical
 * structure — only the title and the filtered/sorted rows differ. Extracting
 * avoids duplicating 30 lines of JSX.
 */
function DiffCard({
  title,
  rows,
}: {
  title: string;
  rows: PerQuestionDiff[];
}) {
  if (rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b text-muted-foreground">
              <th className="py-2 pl-4 pr-2 text-left font-medium">Question ID</th>
              <th className="py-2 px-2 text-left font-medium">Dataset</th>
              <th className="py-2 pl-2 pr-4 text-right font-medium tabular-nums">
                A → B (Δ)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.question_id}-${row.dataset}`} className="border-b last:border-0">
                <td
                  className="py-1.5 pl-4 pr-2 font-mono"
                  title={row.question_id}
                >
                  {shortId(row.question_id)}
                </td>
                <td className="py-1.5 px-2 text-muted-foreground">
                  {row.dataset}
                </td>
                <td className="py-1.5 pl-2 pr-4 text-right tabular-nums">
                  {/*
                   * PATTERN: Show raw scores for transparency so readers know
                   * whether a +0.20 delta is 0.60→0.80 or 0.10→0.30 — the
                   * absolute values matter for interpretation.
                   */}
                  {row.a_score.toFixed(2)} → {row.b_score.toFixed(2)}{" "}
                  <span
                    className={
                      row.delta >= 0 ? "text-green-600" : "text-destructive"
                    }
                  >
                    ({fmtDelta(row.delta)})
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * CompareView renders a side-by-side comparison of two eval runs.
 *
 * URL contract: `/eval/compare?a=<run_id>&b=<run_id>`.
 * Both params must be present; missing either shows a friendly prompt.
 *
 * Data flow:
 *   URL params a, b
 *     → useCompareRuns(a, b)        (single API call)
 *     → CompareResult { run_a, run_b, deltas, per_question_diff }
 *     → mapAggregatedFromDeltas     (derives AggregatedMetric[] for each side)
 *     → MetricBars (comparison mode)
 *     → DiffCard × 2               (wins / regressions)
 */
export function CompareView(): React.JSX.Element {
  const [searchParams] = useSearchParams();
  const a = searchParams.get("a") ?? undefined;
  const b = searchParams.get("b") ?? undefined;

  // -------------------------------------------------------------------------
  // Missing-params guard — shown before the hook fires
  // -------------------------------------------------------------------------

  if (!a || !b) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-12 text-center">
        <p className="text-muted-foreground">
          Select two runs from the runs list to compare.
        </p>
        <Link
          to="/eval"
          className="text-sm text-primary underline-offset-4 hover:underline"
        >
          Go to runs list
        </Link>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Data fetch — useCompareRuns is enabled only when both IDs are defined
  // -------------------------------------------------------------------------

  // IMPORTANT: hooks must be called unconditionally in React. The early return
  // above only fires when a or b is undefined, so by this point both are
  // strings — the hook's `enabled: !!idA && !!idB` will be true.
  //
  // WHY not call the hook before the guard: it would still be enabled=false and
  // return { data: undefined, isLoading: false }, so this ordering is safe.
  // Hooks are called on every render regardless; the guard just short-circuits
  // the JSX, not the hook call itself.

  return <CompareViewInner a={a} b={b} />;
}

/**
 * CompareViewInner — rendered only when both run IDs are present.
 *
 * WHY split into an inner component:
 *   React's rules of hooks prohibit calling hooks conditionally. The outer
 *   CompareView must return early for the missing-params case. Moving the hook
 *   call here lets the guard live at the top level without violating the rules.
 *
 * PATTERN: This is a standard React "wrapper with guard → inner with hook"
 *   split. RunDetail uses the same pattern for runId from useParams.
 */
function CompareViewInner({ a, b }: { a: string; b: string }) {
  const { data, isLoading, isError, error } = useCompareRuns(a, b);

  // -------------------------------------------------------------------------
  // Loading state
  // -------------------------------------------------------------------------

  if (isLoading) return <CompareSkeleton />;

  // -------------------------------------------------------------------------
  // Error state — 409 (version mismatch) gets a distinct banner
  // -------------------------------------------------------------------------

  if (isError) {
    const msg =
      error instanceof Error ? error.message : "Unknown error";

    // WHY check "mismatch": the backend detail string is
    //   "eval set version mismatch between runs" (compare.py, line 140).
    // Matching "mismatch" is narrower than "version" and won't false-positive
    // on other errors that might mention "version".
    const isVersionMismatch =
      msg.toLowerCase().includes("mismatch");

    if (isVersionMismatch) {
      return (
        <div className="flex flex-col gap-4 p-6">
          <div className="rounded-md border border-yellow-400/50 bg-yellow-50 px-4 py-3 text-sm text-yellow-800 dark:border-yellow-500/30 dark:bg-yellow-950/30 dark:text-yellow-300">
            These runs cannot be compared because they used different eval sets.
          </div>
          <Link
            to="/eval"
            className="self-start text-sm text-primary underline-offset-4 hover:underline"
          >
            ← Back to runs list
          </Link>
        </div>
      );
    }

    // Generic error (404, 500, network failure)
    return (
      <div className="flex flex-col gap-4 p-6">
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Failed to load comparison: {msg}
        </div>
        <Link
          to="/eval"
          className="self-start text-sm text-primary underline-offset-4 hover:underline"
        >
          ← Back to runs list
        </Link>
      </div>
    );
  }

  if (!data) return <CompareSkeleton />;

  // -------------------------------------------------------------------------
  // Derived state
  // -------------------------------------------------------------------------

  const { run_a, run_b, deltas, per_question_diff } = data;

  // Cast per_question_diff to our typed interface once. The backend schema
  // guarantees this shape (see src/eval/compare.py), but the wire type is
  // Array<Record<string, unknown>> to avoid coupling the TS types to the
  // Python DTO in full detail.
  // WHY double cast (unknown → PerQuestionDiff[]): TypeScript won't allow a
  // direct cast from Record<string, unknown>[] to a named interface because the
  // two types don't share an overlap that TS can verify statically. Casting
  // through unknown is the idiomatic escape hatch when the shape is guaranteed
  // by a runtime contract we trust (the backend schema).
  const diffs = per_question_diff as unknown as PerQuestionDiff[];

  // Wins: positive deltas, largest first, max 5.
  const wins = diffs
    .filter((d) => d.delta > 0)
    .sort((x, y) => y.delta - x.delta)
    .slice(0, 5);

  // Regressions: negative deltas, largest magnitude first (most negative → first), max 5.
  const regressions = diffs
    .filter((d) => d.delta < 0)
    .sort((x, y) => x.delta - y.delta)
    .slice(0, 5);

  // Rebuild AggregatedMetric arrays for MetricBars comparison mode.
  const metricsA = mapAggregatedFromDeltas(deltas, "a");
  const metricsB = mapAggregatedFromDeltas(deltas, "b");

  // -------------------------------------------------------------------------
  // Render — happy path
  // -------------------------------------------------------------------------

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Back link */}
      <Link
        to="/eval"
        className="self-start text-sm text-primary underline-offset-4 hover:underline"
      >
        ← All runs
      </Link>

      {/* Section 1: Run summaries — two-column grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <RunSummaryPanel label="Run A" run={run_a} />
        <RunSummaryPanel label="Run B" run={run_b} />
      </div>

      {/* Section 2: Metric comparison chart */}
      <Card>
        <CardHeader>
          <CardTitle>Metric Comparison</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          <MetricBars
            metrics={metricsA}
            comparison={{ b: metricsB, deltas }}
            height={300}
          />
          {/*
           * WHY a separate caption here despite MetricBars rendering its own
           * significance list: MetricBars' line names *which* metrics are
           * significant. This caption explains the test method so readers know
           * what "significant" means without digging into the source.
           *
           * TRADE-OFF: mild duplication of context. The alternative — removing
           * this caption — leaves the ★ symbol unexplained at the chart level.
           */}
          <p className="text-xs text-muted-foreground">
            ★ marks differences with p &lt; 0.05 (paired permutation test, n=10 000).
          </p>
        </CardContent>
      </Card>

      {/* Sections 3 & 4: Top wins and top regressions cards */}
      {(wins.length > 0 || regressions.length > 0) && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {wins.length > 0 && (
            <DiffCard title="Top Wins (Run B > Run A)" rows={wins} />
          )}
          {regressions.length > 0 && (
            <DiffCard title="Top Regressions (Run B < Run A)" rows={regressions} />
          )}
        </div>
      )}
    </div>
  );
}
