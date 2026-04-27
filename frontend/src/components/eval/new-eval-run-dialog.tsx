/**
 * NewEvalRunDialog — config picker + run submit + status-poll toast.
 *
 * Frontend Position:
 *   RunsList "New Run" button → [Dialog] → POST /api/eval/run
 *                                       → toast → poll /status → "View Run"
 *
 * WHY this approach (Approach B — self-contained toast):
 *   The toast lifecycle is independent of the dialog: after submit the dialog
 *   closes but the toast must live on until the run finishes. We manage both
 *   in one file by keeping `activeRunId` state here. The toast renders as a
 *   fixed-position card outside the dialog markup, so it persists even after
 *   the dialog closes.
 *
 * PATTERN: Controlled dialog — the `open` / `onOpenChange` props come from
 *   RunsList, which renders <NewEvalRunDialog> at a stable JSX position (not
 *   inside a conditional branch). This guarantees React never remounts the
 *   component (and loses `activeRunId`) when the runs list transitions from
 *   "empty" to "loaded" after the first run is submitted.
 */

import React, { useState } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { getRunStatus, useConfigs, useSubmitRun } from "@/api/eval";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface NewEvalRunDialogProps {
  open: boolean;
  /** Called by Base UI Dialog when it wants to open/close (dismiss on Escape,
   *  backdrop click). We also call it explicitly after a successful submit. */
  onOpenChange: (open: boolean) => void;
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

/**
 * NewEvalRunDialog renders two co-located UIs:
 *   1. A modal dialog for picking a config and submitting a run.
 *   2. A fixed-position toast card that appears after submit and polls the
 *      run status until completion or failure.
 *
 * WHY co-located: only one component tracks `activeRunId`, keeping the state
 * in one place and avoiding cross-component event buses.
 */
export function NewEvalRunDialog({
  open,
  onOpenChange,
}: NewEvalRunDialogProps): React.JSX.Element {
  const [selectedConfig, setSelectedConfig] = useState<string>("");
  // activeRunId is set on submit success and cleared when the toast is dismissed.
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  const { data: configs, isLoading: configsLoading } = useConfigs();
  const submitRun = useSubmitRun();

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  function handleSubmit() {
    if (!selectedConfig) return;

    submitRun.mutate(selectedConfig, {
      onSuccess(result) {
        // Close the dialog and start tracking the new run.
        onOpenChange(false);
        setSelectedConfig("");
        setActiveRunId(result.run_id);
      },
    });
  }

  // Base UI's onOpenChange passes (open, eventDetails) — we only need `open`.
  // When the dialog closes (Escape or backdrop), reset form state but keep the
  // active toast if one is running.
  function handleOpenChange(next: boolean) {
    if (!next) {
      // Reset form when dialog closes (cancel or dismiss).
      setSelectedConfig("");
      submitRun.reset();
    }
    onOpenChange(next);
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <>
      {/* ------------------------------------------------------------------ */}
      {/* Modal dialog                                                         */}
      {/* ------------------------------------------------------------------ */}
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Eval Run</DialogTitle>
          </DialogHeader>

          <div className="flex flex-col gap-3 py-2">
            <label
              htmlFor="eval-config-select"
              className="text-sm font-medium"
            >
              Config
            </label>

            {configsLoading ? (
              /* Loading skeleton — shown while /api/eval/configs is in flight */
              <Skeleton className="h-9 w-full" />
            ) : (
              /*
               * WHY native <select> over a custom Select primitive:
               *   No shadcn Select component exists in this repo. Adding one
               *   just for this dialog would be scope creep. A styled native
               *   <select> is accessible, keyboard-navigable, and renders
               *   identically in all browsers — the right default here.
               */
              <select
                id="eval-config-select"
                value={selectedConfig}
                onChange={(e) => setSelectedConfig(e.target.value)}
                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50"
              >
                <option value="" disabled>
                  {configs && configs.length > 0
                    ? "Select a config…"
                    : "No configs available"}
                </option>
                {configs?.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            )}

            {/* Mutation error — shown if the submit POST fails */}
            {submitRun.isError && (
              <p className="text-sm text-destructive">
                {submitRun.error instanceof Error
                  ? submitRun.error.message
                  : "Failed to start run"}
              </p>
            )}
          </div>

          <DialogFooter showCloseButton>
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={!selectedConfig || submitRun.isPending}
            >
              {submitRun.isPending ? "Starting…" : "Start Run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ------------------------------------------------------------------ */}
      {/* Status toast — mounted after a successful submit, lives independently */}
      {/* of the dialog's open state.                                          */}
      {/* ------------------------------------------------------------------ */}
      {activeRunId !== null && (
        <RunStatusToast
          runId={activeRunId}
          onDismiss={() => setActiveRunId(null)}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// RunStatusToast — polls /status and renders a fixed bottom-right card
// ---------------------------------------------------------------------------

/**
 * RunStatusToast displays live progress for a single eval run.
 *
 * RAG Pipeline Position:
 *   POST /api/eval/run → [RunStatusToast] → GET /api/eval/runs/:id/status (polling)
 *                                         → "View Run" → /eval/runs/:id
 *
 * WHY fixed-position card instead of the global sonner toast:
 *   The standard sonner toast API doesn't support updating existing toasts with
 *   rich interactive elements like a progress bar, a "View Run" button, or an
 *   error message. A fixed card gives us full control over the content lifecycle
 *   while still appearing "toast-like" at the bottom-right corner.
 *
 * PATTERN: refetchInterval stops polling once status reaches a terminal state.
 *   We pass `false` for completed/failed runs, preventing unnecessary requests
 *   after the run is done.
 */
function RunStatusToast({
  runId,
  onDismiss,
}: {
  runId: string;
  onDismiss: () => void;
}) {
  const navigate = useNavigate();

   /*
   * WHY useQuery directly instead of useRunStatus:
   *   The `useRunStatus` hook accepts only `number | false` for refetchInterval.
   *   To avoid circular-reference TypeScript errors (referencing `data` before
   *   it is declared in the same statement) and the linter's no-setState-in-effect
   *   rule, we call useQuery directly and pass the TanStack v5 function form of
   *   `refetchInterval`. The function receives the query object which carries the
   *   latest data via `query.state.data`, so we can read the status without any
   *   extra state variable.
   *
   * PATTERN: refetchInterval: (query) => { ... } — evaluated after each fetch.
   *   Returning `false` stops future polls; returning a number schedules the next.
   */
  const { data } = useQuery({
    queryKey: ["eval-run-status", runId],
    queryFn: () => getRunStatus(runId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "completed" || s === "failed" ? false : 1000;
    },
  });

  const status = data?.status ?? "queued";
  const isTerminal = status === "completed" || status === "failed";

  // Short run_id for display (first 8 chars of UUID).
  const shortRunId = runId.length > 8 ? runId.slice(0, 8) + "…" : runId;

  return (
    /*
     * WHY z-[100]: dialog overlay uses z-50, so the toast needs a higher
     * z-index to remain visible over the overlay when both appear simultaneously
     * (unlikely but defensive).
     */
    <div
      role="status"
      aria-live="polite"
      aria-label={`Eval run ${shortRunId} status`}
      className="fixed bottom-4 right-4 z-[100] w-80 rounded-xl border bg-popover p-4 text-sm text-popover-foreground shadow-lg ring-1 ring-foreground/10"
    >
      {/* Header row: run ID + status badge */}
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium">
          Run{" "}
          <span className="font-mono text-xs" title={runId}>
            {shortRunId}
          </span>
        </span>
        <StatusBadge status={status} />
      </div>

      {/* Progress bar — shown while running or queued */}
      {!isTerminal && (
        <div className="mt-3">
          <Progress
            value={
              data
                ? data.n_total > 0
                  ? (data.n_completed / data.n_total) * 100
                  : 0
                : 0
            }
          />
          {data && data.n_total > 0 && (
            <p className="mt-1.5 text-xs text-muted-foreground">
              {data.n_completed} / {data.n_total} questions
            </p>
          )}
        </div>
      )}

      {/* Completed: replace progress bar with "View Run" + dismiss */}
      {status === "completed" && (
        <div className="mt-3 flex gap-2">
          <Button
            size="sm"
            onClick={() => {
              navigate(`/eval/runs/${runId}`);
              onDismiss();
            }}
          >
            View Run
          </Button>
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Failed: error message + dismiss */}
      {status === "failed" && (
        <div className="mt-3 flex flex-col gap-2">
          {data?.error_message && (
            <p className="text-xs text-destructive">{data.error_message}</p>
          )}
          <Button
            size="sm"
            variant="destructive"
            className="self-start"
            onClick={onDismiss}
          >
            Dismiss
          </Button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatusBadge helper
// ---------------------------------------------------------------------------

/**
 * Small inline badge coloured by run status.
 *
 * WHY inline status-to-colour map: it's used only in this one component.
 * Pulling it into a shared constant would be premature abstraction.
 */
function StatusBadge({ status }: { status: string }) {
  const colours: Record<string, string> = {
    queued: "bg-muted text-muted-foreground",
    running: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    completed: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    failed: "bg-destructive/10 text-destructive",
  };
  const cls = colours[status] ?? colours["queued"];
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}
