/**
 * MetricBars — recharts BarChart for AggregatedMetric arrays with CI whiskers.
 *
 * RAG Pipeline Position (Evaluation Layer):
 *   Run → AggregatedMetric[] → [METRIC-BARS] → Visual bar chart
 *                                    ^^^
 *   This component sits at the DISPLAY step of evaluation: it takes
 *   pre-aggregated per-metric statistics (mean, CI bounds) and renders
 *   them as a bar chart so engineers can compare quality at a glance.
 *
 * WHY recharts over a custom SVG:
 *   recharts gives us ErrorBar (CI whiskers), ResponsiveContainer, and
 *   animated bars without any D3 boilerplate. The trade-off is a heavier
 *   bundle, but recharts is already installed for this project.
 *
 * Two modes:
 *   - Single-run: one bar per (metric_name, dataset) row + ErrorBar whiskers.
 *   - Comparison: two bars per row (Run A / Run B). Statistically significant
 *     deltas are called out below the chart in a ★-prefixed list.
 *
 * TRADE-OFF: The ★ significant-delta annotation is rendered as text below the
 *   chart rather than as an overlay on the bars. Overlaying labels on grouped
 *   bars in recharts requires a custom <Customized> renderer that re-computes
 *   bar x/y positions — fragile across bar widths. The text list is simpler,
 *   equally informative, and survives recharts version changes.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AggregatedMetric, MetricDelta } from "@/api/eval";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface MetricBarsProps {
  metrics: AggregatedMetric[];
  comparison?: {
    b: AggregatedMetric[];
    deltas: MetricDelta[];
  };
  /** Chart height in px. Default 300. */
  height?: number;
  className?: string;
}

// ---------------------------------------------------------------------------
// Internal row shape consumed by recharts <BarChart data={rows}>
// ---------------------------------------------------------------------------

interface ChartRow {
  /** X-axis label: "{metric_name} ({dataset || 'all'})" */
  label: string;
  /** Run A mean */
  a_mean: number;
  /**
   * ErrorBar offset pair: [distance below mean, distance above mean].
   * recharts ErrorBar interprets a 2-element array as [below, above].
   */
  a_err: [number, number];
  /** Run B mean (comparison mode only) */
  b_mean?: number;
  b_err?: [number, number];
  /** True when the matching MetricDelta.significant is true */
  significant?: boolean;
}

// ---------------------------------------------------------------------------
// buildRows — transforms AggregatedMetric arrays into chart-ready rows
// ---------------------------------------------------------------------------

/**
 * Build ChartRow array from one or two sets of AggregatedMetric.
 *
 * Single-run: every metric in `a` becomes a row.
 * Compare: only metrics present in BOTH `a` and `b` are included — partial
 * matches would render an empty bar for the missing side, which is confusing.
 *
 * WHY offset-based ErrorBar:
 *   recharts ErrorBar for a single <Bar> expects the dataKey to point at a
 *   2-element array [belowOffset, aboveOffset], NOT absolute CI values.
 *   So we convert: below = mean - ci_low, above = ci_high - mean.
 */
function buildRows(
  a: AggregatedMetric[],
  b?: AggregatedMetric[],
  deltas?: MetricDelta[],
): ChartRow[] {
  // Build a lookup for Run B keyed by "metric_name|dataset" for O(1) access.
  const bByKey = new Map<string, AggregatedMetric>();
  if (b) {
    for (const m of b) {
      bByKey.set(`${m.metric_name}|${m.dataset ?? ""}`, m);
    }
  }

  // Build a significance lookup keyed the same way.
  const sigByKey = new Map<string, boolean>();
  if (deltas) {
    for (const d of deltas) {
      sigByKey.set(`${d.metric_name}|${d.dataset ?? ""}`, d.significant);
    }
  }

  const rows: ChartRow[] = [];

  for (const am of a) {
    const key = `${am.metric_name}|${am.dataset ?? ""}`;
    const dataset = am.dataset ?? "all";
    const label = `${am.metric_name} (${dataset})`;

    const a_err: [number, number] = [
      am.mean - am.ci_low,
      am.ci_high - am.mean,
    ];

    if (!b) {
      // Single-run mode — no Run B needed.
      rows.push({ label, a_mean: am.mean, a_err });
      continue;
    }

    // Comparison mode — skip metrics missing from Run B.
    const bm = bByKey.get(key);
    if (!bm) continue;

    const b_err: [number, number] = [
      bm.mean - bm.ci_low,
      bm.ci_high - bm.mean,
    ];

    rows.push({
      label,
      a_mean: am.mean,
      a_err,
      b_mean: bm.mean,
      b_err,
      significant: sigByKey.get(key) ?? false,
    });
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

interface TooltipPayloadEntry {
  name: string;
  value: number;
  payload: ChartRow;
  dataKey: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

/**
 * CustomTooltip — shows "mean (± half-CI-width)" formatted to 4 decimals.
 *
 * WHY custom tooltip instead of recharts default:
 *   The default tooltip would show raw a_err/b_err arrays, which are offsets
 *   not intuitive values. We convert back to ± half-CI for readability.
 */
function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-medium">{label}</p>
      {payload.map((entry) => {
        // entry.dataKey is "a_mean" or "b_mean"; the matching err key is "a_err" or "b_err".
        const errKey = entry.dataKey === "a_mean" ? "a_err" : "b_err";
        const err = entry.payload[errKey as "a_err" | "b_err"];
        // Half-CI-width is the average of [below, above] offsets.
        const halfCI = err ? ((err[0] + err[1]) / 2).toFixed(4) : "—";
        return (
          <p key={entry.name} style={{ color: entry.name === "Run A" ? "#4a90e2" : "#f39c12" }}>
            {entry.name}: {entry.value.toFixed(4)} (±{halfCI})
          </p>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricBars — public export
// ---------------------------------------------------------------------------

/**
 * MetricBars renders AggregatedMetric arrays as a bar chart.
 *
 * @example Single-run
 *   <MetricBars metrics={run.aggregated} />
 *
 * @example Comparison
 *   <MetricBars metrics={runA.aggregated} comparison={{ b: runB.aggregated, deltas }} />
 */
export function MetricBars({
  metrics,
  comparison,
  height = 300,
  className,
}: MetricBarsProps) {
  // Empty state — return a muted paragraph so the parent doesn't render
  // dead whitespace or an empty chart frame.
  if (metrics.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No metrics yet.</p>
    );
  }

  const rows = buildRows(metrics, comparison?.b, comparison?.deltas);

  // Y-axis: use [0, 1] for standard 0-1 metrics. If any mean exceeds 1,
  // let recharts auto-scale (pass undefined to let it decide).
  // WHY: Most RAG metrics (recall@k, faithfulness, NDCG) live in [0, 1].
  //      Auto-scaling for edge cases (e.g., raw token counts) prevents clipping.
  const allMeans = rows.flatMap((r) =>
    r.b_mean !== undefined ? [r.a_mean, r.b_mean] : [r.a_mean],
  );
  const maxMean = Math.max(...allMeans);
  const yDomain: [number | string, number | string] =
    maxMean <= 1.0 ? [0, 1] : [0, "auto"];

  // Collect labels of significant deltas for the annotation list.
  const significantLabels = rows
    .filter((r) => r.significant)
    .map((r) => r.label);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={rows}
          margin={{ top: 8, right: 16, bottom: 48, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          {/*
           * XAxis angle -30 with textAnchor="end" and extra height prevents
           * long labels from overlapping. This mirrors the pattern used in
           * evaluation dashboards where metric names can be verbose.
           */}
          <XAxis
            dataKey="label"
            angle={-30}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 11 }}
          />
          <YAxis domain={yDomain} tick={{ fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend verticalAlign="top" height={36} />

          {/* Run A bar — always rendered */}
          <Bar dataKey="a_mean" name="Run A" fill="#4a90e2" maxBarSize={48}>
            {/*
             * PATTERN: ErrorBar dataKey points at the [below, above] offset
             * array on each row. recharts uses these to draw whiskers relative
             * to the bar top, not absolute SVG coordinates.
             */}
            <ErrorBar
              dataKey="a_err"
              width={4}
              strokeWidth={1.5}
              stroke="#4a90e2"
            />
          </Bar>

          {/* Run B bar — only in comparison mode */}
          {comparison ? (
            <Bar dataKey="b_mean" name="Run B" fill="#f39c12" maxBarSize={48}>
              <ErrorBar
                dataKey="b_err"
                width={4}
                strokeWidth={1.5}
                stroke="#f39c12"
              />
            </Bar>
          ) : null}
        </BarChart>
      </ResponsiveContainer>

      {/*
       * Significant delta annotations — rendered as text below the chart.
       * WHY text list over bar overlay: overlaying labels on grouped bars
       * requires re-computing bar x/y positions via recharts internals,
       * which is fragile. The list is simpler and equally informative.
       */}
      {comparison && significantLabels.length > 0 && (
        <div className="mt-2 text-xs text-muted-foreground">
          <span className="font-medium">★ Statistically significant (p &lt; 0.05):</span>{" "}
          {significantLabels.join(", ")}
        </div>
      )}

      {comparison && significantLabels.length === 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          No statistically significant differences between runs.
        </p>
      )}
    </div>
  );
}
