/**
 * EvaluationSection — displays LLM-as-judge scores in the Sources panel.
 *
 * RAG Pipeline Position:
 *   Query → Retrieve → Generate → [EVALUATION] → Score
 *                                      ^^^
 *   After an answer is generated, a judge model scores it against the
 *   retrieved chunks on metrics like faithfulness, answer relevancy, and
 *   context precision. This component renders those scores.
 *
 * WHY: Showing evaluation results alongside sources lets users quickly
 *      judge whether the answer is reliable without reading the full
 *      reasoning. The expandable "View Details" section surfaces the
 *      judge's chain-of-thought for power users.
 *
 * PATTERN: Co-located helper components (ScoreBar, ClaimBreakdown) are
 *          kept in this file because they are tightly coupled display
 *          primitives with no reuse outside EvaluationSection.
 */

import { useState } from "react";
import type { EvaluationScore } from "@/api/types";

// ---------------------------------------------------------------------------
// ScoreBar — a thin progress bar showing one metric's score with color coding
// ---------------------------------------------------------------------------

function ScoreBar({ score }: { score: EvaluationScore }) {
  const pct = Math.round(score.score * 100);

  // WHY: Three-tier color system maps naturally to traffic-light semantics.
  //      >= 0.8 → green (reliable), >= 0.5 → amber (acceptable), < 0.5 → red (poor).
  const color =
    score.score >= 0.8
      ? "bg-emerald-500"
      : score.score >= 0.5
        ? "bg-amber-500"
        : "bg-red-500";

  // Convert snake_case metric names to Title Case for display
  const label = score.metric
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-muted-foreground mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      {/* PATTERN: Thin h-1.5 bars keep the panel compact; bg-[#F3F4F6] is
                   the same neutral track color used elsewhere in the UI. */}
      <div className="h-1.5 bg-[#F3F4F6] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ClaimBreakdown — parses faithfulness `details` JSON and lists claims
// ---------------------------------------------------------------------------

function ClaimBreakdown({ details }: { details: string }) {
  // WHY: The judge model returns a JSON string so it can be stored as a
  //      plain scalar in the DB. We parse it here, closest to render, and
  //      bail out silently if the format is unexpected to avoid crashes.
  try {
    const parsed = JSON.parse(details) as { claims?: unknown };
    const claims = parsed.claims;
    if (!Array.isArray(claims)) return null;

    return (
      <div className="mt-2 space-y-1">
        {claims.map(
          (
            claim: { claim: string; supported: boolean },
            i: number
          ) => (
            <div key={i} className="flex items-start gap-2 text-xs">
              {/* Green dot = supported by retrieved context; red = hallucinated */}
              <span
                className={`mt-1 size-2 rounded-full shrink-0 ${
                  claim.supported ? "bg-emerald-500" : "bg-red-500"
                }`}
              />
              <span className="text-muted-foreground">{claim.claim}</span>
            </div>
          )
        )}
      </div>
    );
  } catch {
    // Malformed JSON — render nothing rather than crashing the panel
    return null;
  }
}

// ---------------------------------------------------------------------------
// EvaluationSection — public export, composed from the helpers above
// ---------------------------------------------------------------------------

interface EvaluationSectionProps {
  evaluation: EvaluationScore[];
}

export function EvaluationSection({ evaluation }: EvaluationSectionProps) {
  const [showDetails, setShowDetails] = useState(false);

  // WHY: Return null (not an empty placeholder) so the parent panel can
  //      decide how to handle the no-evaluation state without rendering
  //      any dead whitespace.
  if (evaluation.length === 0) return null;

  return (
    <div className="p-4 border-b border-[#E5E7EB]">
      {/* Header — matches SourcesPanel section label style */}
      <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-3">
        Evaluation
      </h3>

      {/* Score bars — one per metric */}
      {evaluation.map((score) => (
        <ScoreBar key={score.metric} score={score} />
      ))}

      {/* View Details toggle — link-style, no visual weight */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-xs text-[#0d74e7] hover:underline mt-2"
      >
        {showDetails ? "Hide Details" : "View Details"}
      </button>

      {/* Expanded per-metric reasoning + faithfulness claim breakdown */}
      {showDetails && (
        <div className="mt-3 space-y-3">
          {evaluation.map((score) => (
            <div
              key={score.metric}
              className="bg-[#F9FAFB] rounded-lg p-3 text-xs"
            >
              <p className="font-medium text-muted-foreground mb-1">
                {score.metric
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (l) => l.toUpperCase())}
              </p>
              <p className="text-muted-foreground leading-relaxed">
                {score.reasoning}
              </p>
              {/* PATTERN: ClaimBreakdown only renders for faithfulness because
                           that is the only metric whose details JSON contains
                           a `claims` array. Other metrics may use `details`
                           for raw scores or structured outputs in the future. */}
              {score.metric === "faithfulness" && score.details && (
                <ClaimBreakdown details={score.details} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
