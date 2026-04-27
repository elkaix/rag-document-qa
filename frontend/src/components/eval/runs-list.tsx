/**
 * RunsList — table of eval runs with sort, filter, and multi-select compare.
 *
 * Frontend Position:
 *   /eval (index route) → [RunsList] → row click → /eval/runs/:runId
 *                                    → "Compare" → /eval/compare?a=&b=
 *                                    → "New Run" → NewEvalRunDialog (Task 10)
 *
 * WHY this file exists:
 *   The eval dashboard entry point. Engineers need to see all past runs at a
 *   glance, filter by config, sort by any column, and quickly navigate to
 *   detailed views or side-by-side comparisons.
 *
 * PATTERN: Filter → sort → render is a classic derived-state pipeline.
 *   Both transformations are memoised with useMemo so they don't re-run on
 *   unrelated parent re-renders.
 */

import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useDebounce } from "@/hooks/use-debounce";
import { useRunsList } from "@/api/eval";
import type { RunSummary } from "@/api/eval";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
// Types
// ---------------------------------------------------------------------------

type SortKey = keyof Pick<
  RunSummary,
  "config_name" | "started_at" | "n_questions" | "n_errors" | "headline_metric"
>;

type SortDir = "asc" | "desc";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format an ISO timestamp as a short locale date + time string.
 *
 * WHY Intl.DateTimeFormat over date-fns:
 *   No extra dependency; the browser-native formatter handles locale/timezone
 *   automatically and is already used in doc-table for date-only formatting.
 */
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
 * Format headline_metric to 3 decimal places, or "—" if null.
 *
 * WHY 3 decimals: standard precision for recall/NDCG metrics reported to
 * engineers — enough signal without false precision.
 */
function formatMetric(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

/**
 * Truncate a run_id UUID to the first 8 chars for display.
 *
 * WHY: Full UUIDs overflow narrow columns and are unreadable at a glance.
 *      8 chars is enough to disambiguate within a typical run list.
 */
function shortId(id: string): string {
  return id.length > 8 ? id.slice(0, 8) + "…" : id;
}

/**
 * Generic comparator that handles null values, always sorting them last
 * regardless of sort direction.
 *
 * WHY nulls-last: a null headline_metric means the run has no results yet,
 *   which is less informative than a real value. Keeping nulls at the bottom
 *   regardless of direction avoids them dominating the first rows on asc.
 */
function compare(a: RunSummary, b: RunSummary, key: SortKey, dir: SortDir): number {
  const va = a[key];
  const vb = b[key];

  // Nulls always sink to the bottom.
  if (va === null && vb === null) return 0;
  if (va === null) return 1;
  if (vb === null) return -1;

  // ISO strings sort correctly as strings (lexicographic == chronological).
  // Numbers sort numerically. Both comparisons are uniform here.
  const cmp = va < vb ? -1 : va > vb ? 1 : 0;
  return dir === "asc" ? cmp : -cmp;
}

// ---------------------------------------------------------------------------
// Sort header button
// ---------------------------------------------------------------------------

/** Renders a clickable column header with an ↑/↓ glyph when active. */
function SortHead({
  label,
  sortKey,
  currentKey,
  currentDir,
  onSort,
  className,
}: {
  label: string;
  sortKey: SortKey;
  currentKey: SortKey;
  currentDir: SortDir;
  onSort: (key: SortKey) => void;
  className?: string;
}) {
  const isActive = currentKey === sortKey;
  return (
    <TableHead className={className}>
      <button
        className="flex items-center gap-1 font-medium hover:text-foreground transition-colors"
        onClick={() => onSort(sortKey)}
        type="button"
      >
        {label}
        {isActive && (
          <span className="text-xs">{currentDir === "asc" ? "↑" : "↓"}</span>
        )}
      </button>
    </TableHead>
  );
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

/** Renders 5 placeholder rows while the API request is in-flight. */
function RunsListSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-9 w-48" />
        <div className="flex gap-2">
          <Skeleton className="h-9 w-32" />
          <Skeleton className="h-9 w-24" />
        </div>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            {["w-6", "w-24", "w-40", "w-36", "w-20", "w-16", "w-28"].map(
              (w, i) => (
                <TableHead key={i} className={w}>
                  <Skeleton className="h-4 w-full" />
                </TableHead>
              ),
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array.from({ length: 5 }).map((_, i) => (
            <TableRow key={i}>
              {Array.from({ length: 7 }).map((__, j) => (
                <TableCell key={j}>
                  <Skeleton className="h-4 w-full" />
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * RunsList renders the full list of eval runs.
 *
 * State machines:
 *   - Loading  → skeleton
 *   - Error    → error banner
 *   - Empty    → "No eval runs yet" callout
 *   - Loaded   → sorted, filtered, paginated table
 *
 * Multi-select behaviour:
 *   Selected IDs are kept in a Set<string>. Filtering/sorting does NOT clear
 *   the selection — selected IDs that scroll off the visible (filtered) list
 *   remain selected and still count toward the "Compare Selected" threshold.
 *   This avoids surprising selection resets while the user is typing a filter.
 */
export function RunsList() {
  const navigate = useNavigate();
  const { data, isLoading, isError, error } = useRunsList();

  // Search input (raw) and debounced query fed into useMemo.
  const [searchRaw, setSearchRaw] = useState("");
  const query = useDebounce(searchRaw, 300);

  // Sort state — default: started_at descending (most recent first).
  const [sortKey, setSortKey] = useState<SortKey>("started_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // Multi-select: a Set of run_ids that are checked.
  // WHY Set: O(1) has/toggle, serialises cheaply to an array for the URL.
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // -------------------------------------------------------------------------
  // Derived state: filter then sort
  // -------------------------------------------------------------------------

  const filtered = useMemo(() => {
    if (!data) return [];
    if (!query) return data;
    const q = query.toLowerCase();
    return data.filter((r) => r.config_name.toLowerCase().includes(q));
  }, [data, query]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => compare(a, b, sortKey, sortDir));
  }, [filtered, sortKey, sortDir]);

  // -------------------------------------------------------------------------
  // Handlers
  // -------------------------------------------------------------------------

  function handleSortClick(key: SortKey) {
    if (key === sortKey) {
      // Same column → toggle direction.
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      // New column → default to descending so largest/latest shows first.
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function handleCompare() {
    // Guard: only proceed when exactly 2 are selected (belt-and-suspenders
    // since the button is also disabled otherwise).
    const ids = Array.from(selected);
    if (ids.length !== 2) return;
    navigate(`/eval/compare?a=${encodeURIComponent(ids[0])}&b=${encodeURIComponent(ids[1])}`);
  }

  function handleRowClick(runId: string) {
    navigate(`/eval/runs/${runId}`);
  }

  // -------------------------------------------------------------------------
  // Render: loading
  // -------------------------------------------------------------------------

  if (isLoading) return <RunsListSkeleton />;

  // -------------------------------------------------------------------------
  // Render: error
  // -------------------------------------------------------------------------

  if (isError) {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
        Failed to load eval runs:{" "}
        {error instanceof Error ? error.message : "Unknown error"}
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: empty (no runs at all, not just filtered out)
  // -------------------------------------------------------------------------

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-end">
          <NewRunButton />
        </div>
        <div className="rounded-md border border-dashed px-6 py-12 text-center text-muted-foreground">
          <p className="text-sm">No eval runs yet.</p>
          <p className="mt-1 text-xs">
            Click "New Run" to kick off your first evaluation.
          </p>
        </div>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: table
  // -------------------------------------------------------------------------

  const sortHeadProps = { currentKey: sortKey, currentDir: sortDir, onSort: handleSortClick };

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar: search left, action buttons right */}
      <div className="flex items-center gap-2">
        <Input
          placeholder="Filter by config name…"
          value={searchRaw}
          onChange={(e) => setSearchRaw(e.target.value)}
          className="max-w-xs"
        />

        <div className="ml-auto flex gap-2">
          {/*
           * WHY exactly-2 gate: comparing 1 or 3+ runs has no defined meaning
           * in the current CompareView (it accepts exactly two run IDs).
           */}
          <Button
            variant="outline"
            size="sm"
            disabled={selected.size !== 2}
            onClick={handleCompare}
          >
            Compare Selected
          </Button>

          <NewRunButton />
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            {/* Checkbox header — no sort, just labels the column */}
            <TableHead className="w-8" />

            {/* Run ID — not sortable; truncated for display only */}
            <TableHead className="w-28 font-medium">Run ID</TableHead>

            <SortHead
              label="Config"
              sortKey="config_name"
              {...sortHeadProps}
            />
            <SortHead
              label="Started"
              sortKey="started_at"
              {...sortHeadProps}
            />
            <SortHead
              label="Questions"
              sortKey="n_questions"
              {...sortHeadProps}
              className="text-right"
            />
            <SortHead
              label="Errors"
              sortKey="n_errors"
              {...sortHeadProps}
              className="text-right"
            />
            <SortHead
              label="Recall@5"
              sortKey="headline_metric"
              {...sortHeadProps}
              className="text-right"
            />
          </TableRow>
        </TableHeader>

        <TableBody>
          {sorted.map((run) => (
            <RunRow
              key={run.run_id}
              run={run}
              isSelected={selected.has(run.run_id)}
              onToggle={() => toggleSelect(run.run_id)}
              onRowClick={() => handleRowClick(run.run_id)}
            />
          ))}

          {/* Filter produced no results, but runs exist */}
          {sorted.length === 0 && (
            <TableRow>
              <TableCell
                colSpan={7}
                className="py-8 text-center text-muted-foreground"
              >
                No runs match your filter.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row sub-component — keeps the main render readable
// ---------------------------------------------------------------------------

function RunRow({
  run,
  isSelected,
  onToggle,
  onRowClick,
}: {
  run: RunSummary;
  isSelected: boolean;
  onToggle: () => void;
  onRowClick: () => void;
}) {
  return (
    <TableRow
      className="cursor-pointer"
      onClick={onRowClick}
      data-state={isSelected ? "selected" : undefined}
    >
      {/*
       * PATTERN: stopPropagation on the checkbox cell, not just the input,
       *   so the row click does not fire when the user is toggling selection.
       *   Mirrors the doc-table pattern for delete/expand buttons.
       */}
      <TableCell
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
        className="pr-0"
      >
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          aria-label={`Select run ${run.run_id}`}
          className="size-4 cursor-pointer accent-primary"
          // WHY onClick noop: onChange fires on keyboard; onClick is a safety
          // net to prevent the cell's stopPropagation from consuming the event
          // before React's synthetic onChange fires. Both are needed.
          onClick={(e) => e.stopPropagation()}
        />
      </TableCell>

      <TableCell className="font-mono text-xs" title={run.run_id}>
        {shortId(run.run_id)}
      </TableCell>

      <TableCell>{run.config_name}</TableCell>

      <TableCell>{formatDate(run.started_at)}</TableCell>

      <TableCell className="text-right">{run.n_questions}</TableCell>

      <TableCell className="text-right">
        {/* Non-zero errors are highlighted so they stand out at a glance. */}
        <span className={run.n_errors > 0 ? "text-destructive font-medium" : undefined}>
          {run.n_errors}
        </span>
      </TableCell>

      <TableCell className="text-right tabular-nums">
        {formatMetric(run.headline_metric)}
      </TableCell>
    </TableRow>
  );
}

// ---------------------------------------------------------------------------
// New Run button — placeholder until Task 10 (NewEvalRunDialog)
// ---------------------------------------------------------------------------

/**
 * NewRunButton — triggers the new-eval-run flow.
 *
 * TODO (Task 10): Replace the alert with <NewEvalRunDialog /> once that
 *   component is implemented. Options:
 *     1. Lift `open` state to RunsList and render <NewEvalRunDialog open={open} onOpenChange={setOpen} />
 *     2. Dispatch CustomEvent("eval:new-run") and let a top-level listener open the dialog.
 *   Option 1 is simpler and preferred for a single-page feature.
 */
function NewRunButton() {
  return (
    <Button
      size="sm"
      // TODO (Task 10): replace with dialog open handler once NewEvalRunDialog lands.
      onClick={() => alert("TODO: open NewEvalRunDialog (Task 10)")}
    >
      New Run
    </Button>
  );
}
