/**
 * TelemetryFooter — per-message timing and cost summary bar.
 *
 * RAG Pipeline Position:
 *   Query → Retrieval → Generation → Response → [TELEMETRY FOOTER]
 *                                                      ^^^
 *   This component sits at the DISPLAY step, after a full assistant turn
 *   completes. It surfaces backend timing (retrieve_ms, generate_ms),
 *   token counts (prompt + completion), and cost so the developer can spot
 *   slow retrievals, verbose generations, or unexpectedly expensive calls
 *   without leaving the chat window.
 *
 * WHY a dedicated component:
 *   The telemetry data arrives as a separate WebSocket event ("telemetry")
 *   after the "done" event. Keeping its rendering isolated means the parent
 *   (ChatMessage) never needs to know about formatting logic — it just
 *   passes `message.telemetry` through.
 *
 * PATTERN: Tooltip-wrapped trigger — the one-liner summary is always visible;
 *   a hover tooltip surfaces the full prompt/completion split without
 *   cluttering the message layout.
 */

import { Fragment } from "react";

import type { TelemetryPayload } from "@/api/types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface TelemetryFooterProps {
  telemetry: TelemetryPayload;
  className?: string;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format retrieval latency as an integer millisecond value.
 * Retrieval is fast enough that sub-second display is always the right unit.
 */
function formatRetrieve(ms: number): string {
  return `Retrieve ${Math.round(ms)}ms`;
}

/**
 * Format generation latency: seconds for ≥1 s, milliseconds otherwise.
 *
 * WHY dual-unit: generation times span 100 ms (fast local model) to 30 s
 * (slow hosted model). Showing "30,000ms" is less readable than "30.0s".
 */
function formatGenerate(ms: number): string {
  if (ms >= 1000) {
    return `Generate ${(ms / 1000).toFixed(1)}s`;
  }
  return `Generate ${Math.round(ms)}ms`;
}

/**
 * Format total token count with locale-aware thousands separators.
 * e.g. 4217 → "4,217 tok"
 */
function formatTokens(prompt: number, completion: number): string {
  const total = prompt + completion;
  return `${total.toLocaleString()} tok`;
}

/**
 * Format cost in USD to 4 decimal places.
 * Always renders, even when cost is exactly 0 ($0.0000).
 *
 * WHY always render zero: hiding a zero cost would make users think the
 * telemetry event didn't arrive, rather than that the call was free (e.g.
 * when a local Ollama model is used).
 */
function formatCost(cost: number): string {
  return `$${cost.toFixed(4)}`;
}

// ---------------------------------------------------------------------------
// Tooltip content — token breakdown table
// ---------------------------------------------------------------------------

/**
 * TooltipBreakdown — rendered inside the shadcn Tooltip popup.
 *
 * WHY tabular: the four rows (Prompt, Completion, Total, Cost) need alignment
 * to be scannable. A two-column grid achieves this without a <table> element.
 */
function TooltipBreakdown({ telemetry }: { telemetry: TelemetryPayload }) {
  const { prompt_tokens, completion_tokens, cost_usd } = telemetry;
  const total = prompt_tokens + completion_tokens;

  const rows: [string, string][] = [
    ["Prompt", `${prompt_tokens.toLocaleString()} tokens`],
    ["Completion", `${completion_tokens.toLocaleString()} tokens`],
    ["Total", `${total.toLocaleString()} tokens`],
    ["Cost USD", cost_usd.toFixed(4)],
  ];

  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 tabular-nums">
      {rows.map(([label, value]) => (
        // WHY Fragment with key: short-form <>...</> can't carry a key, and
        //     React warns about missing keys on array children.
        <Fragment key={label}>
          <span className="text-right opacity-70">{label}:</span>
          <span>{value}</span>
        </Fragment>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TelemetryFooter — public export
// ---------------------------------------------------------------------------

/**
 * TelemetryFooter renders a muted one-liner beneath an assistant message
 * showing retrieve latency, generate latency, token count, and cost.
 *
 * Usage:
 *   {message.telemetry ? <TelemetryFooter telemetry={message.telemetry} /> : null}
 *
 * @param telemetry - Backend timing and token-cost payload from the
 *   "telemetry" WebSocket event. See TelemetryPayload in api/types.ts.
 * @param className - Optional extra Tailwind classes for the container.
 */
export function TelemetryFooter({
  telemetry,
  className,
}: TelemetryFooterProps): React.JSX.Element {
  const segments = [
    formatRetrieve(telemetry.retrieve_ms),
    formatGenerate(telemetry.generate_ms),
    formatTokens(telemetry.prompt_tokens, telemetry.completion_tokens),
    formatCost(telemetry.cost_usd),
  ];

  // Dot separator — aria-hidden so screen readers don't announce bullet chars.
  const Dot = () => (
    <span aria-hidden="true" className="opacity-40">
      ·
    </span>
  );

  return (
    // PATTERN: TooltipProvider is already mounted at the App root, so we only
    //          need Tooltip > TooltipTrigger > TooltipContent here.
    <Tooltip>
      {/*
       * WHY render prop: @base-ui/react TooltipTrigger renders a <button> by
       * default. We use the `render` prop to substitute a <div> so this purely
       * informational row doesn't carry an implicit interactive button role.
       */}
      <TooltipTrigger
        render={
          <div
            className={cn(
              "mt-1 flex items-center gap-1.5 cursor-default select-none",
              "text-xs text-muted-foreground tabular-nums",
              className,
            )}
          />
        }
      >
        {segments.map((seg, i) => (
          // WHY Fragment with key: the per-iteration wrapper needs a stable
          //     identity to satisfy React's array-children key requirement.
          <Fragment key={seg}>
            {i > 0 && <Dot />}
            <span>{seg}</span>
          </Fragment>
        ))}
      </TooltipTrigger>

      <TooltipContent side="top" align="start">
        <TooltipBreakdown telemetry={telemetry} />
      </TooltipContent>
    </Tooltip>
  );
}
