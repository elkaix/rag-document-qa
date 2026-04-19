import { useEffect, useState } from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThinkingPanelProps {
  /** Programmatic status lines emitted during retrieval. */
  statusLog?: string[];
  /** Accumulated reasoning tokens from the LLM chain-of-thought pass. */
  reasoning?: string;
  /** Seconds the CoT reasoning pass took. Undefined while reasoning is
   *  still streaming (header shows shimmering "Thinking"). Once set, the
   *  header switches to "Thought for N.Ns". */
  thinkingSeconds?: number;
  /** True once the whole stream is complete. Drives the auto-collapse —
   *  the panel stays OPEN through the full stream so the user can watch
   *  reasoning + answer simultaneously, then collapses once done. */
  streamDone?: boolean;
}

/**
 * ChatGPT-style collapsible "Thinking" panel.
 *
 * Lifecycle:
 *   1. Reasoning streaming    (thinkingSeconds undefined)
 *      → panel open, shimmering "Thinking" header with brain icon, status
 *        bullets + italic reasoning text accumulate live.
 *   2. Answer streaming       (thinkingSeconds set, streamDone false)
 *      → header switches to "Thought for N.Ns" (no shimmer), panel STAYS
 *        open so the full reasoning trace remains visible next to the
 *        streaming answer bubble.
 *   3. Done                   (streamDone true)
 *      → auto-collapses. User can still click to re-open and inspect.
 *
 * PATTERN: The panel owns only the open/closed UI state. Everything else
 *          (status log, reasoning text, timing signals) streams in via
 *          props — re-renders happen naturally as parent state updates
 *          token-by-token.
 */
export function ThinkingPanel({
  statusLog,
  reasoning,
  thinkingSeconds,
  streamDone,
}: ThinkingPanelProps) {
  const reasoningActive = thinkingSeconds === undefined;
  const [open, setOpen] = useState(true);

  // WHY collapse on streamDone (not on thinkingSeconds): Collapsing as soon
  //     as the first answer token arrives would hide reasoning the user may
  //     still be reading. Keeping the panel open through the entire answer
  //     stream lets reasoning and answer be compared side by side, then
  //     auto-collapses only once everything has finished.
  useEffect(() => {
    if (streamDone) setOpen(false);
  }, [streamDone]);

  const hasStatus = (statusLog?.length ?? 0) > 0;
  const hasReasoning = (reasoning?.length ?? 0) > 0;
  // Completed historical messages (loaded from DB) carry no trace — don't
  // render a ghost "Thinking..." panel for them.
  if (streamDone && !hasStatus && !hasReasoning) return null;

  const headerText = reasoningActive
    ? "Thinking"
    : `Thought for ${thinkingSeconds!.toFixed(1)}s`;

  return (
    <div className="mb-2 w-full max-w-[85%] rounded-xl border border-[#E5E7EB] bg-[#FAFAFA]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
      >
        <Brain
          className={cn(
            "size-3.5 shrink-0",
            reasoningActive ? "text-[#0d74e7]" : "text-[#6B7280]"
          )}
        />
        <span
          className={cn(
            "text-xs font-medium",
            reasoningActive ? "thinking-shimmer" : "text-[#6B7280]"
          )}
        >
          {headerText}
        </span>
        <span className="ml-auto text-[#9CA3AF]">
          {open ? (
            <ChevronDown className="size-3.5" />
          ) : (
            <ChevronRight className="size-3.5" />
          )}
        </span>
      </button>

      {open && (hasStatus || hasReasoning) && (
        <div className="space-y-2 border-t border-[#E5E7EB] px-3 py-2 text-xs leading-relaxed text-[#4B5563]">
          {hasStatus && (
            <ul className="space-y-1 font-mono text-[11px] text-[#6B7280]">
              {statusLog!.map((line, i) => {
                const isLast =
                  i === statusLog!.length - 1 && reasoningActive && !hasReasoning;
                return (
                  <li key={i} className="flex items-start gap-1.5">
                    <span
                      className={cn(
                        "mt-1 size-1 shrink-0 rounded-full",
                        isLast ? "bg-[#0d74e7]" : "bg-[#D1D5DB]"
                      )}
                    />
                    <span>{line}</span>
                  </li>
                );
              })}
            </ul>
          )}
          {hasReasoning && (
            <div className="whitespace-pre-wrap italic text-[#374151]">
              {reasoning}
              {reasoningActive && (
                <span className="ml-0.5 inline-block h-3 w-[2px] translate-y-0.5 animate-pulse bg-[#0d74e7]" />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
