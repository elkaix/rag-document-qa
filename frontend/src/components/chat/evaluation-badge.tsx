/**
 * EvaluationBadge — inline faithfulness score chip + on-demand evaluator.
 *
 * RAG Pipeline Position:
 *   Document → Chunks → Embeddings → Vector Store → Retrieval → Generator → Answer → [EVALUATION]
 *                                                                                          ^^^
 * This component sits below every assistant answer bubble. It shows the
 * faithfulness score that arrives in real-time via WebSocket (streamed as an
 * "evaluation" event), and provides a manual "Evaluate" button to trigger the
 * full 3-metric LLM-as-judge evaluation when it hasn't run yet.
 *
 * WHY: Faithfulness is the most critical RAG quality signal — it tells you
 * whether the answer is grounded in the retrieved chunks or hallucinated.
 * Surfacing it inline, right next to the answer, makes quality visible without
 * requiring the user to navigate to a separate evaluation view.
 *
 * TRADE-OFF: We only show faithfulness in the badge (not all three metrics)
 * because it's the primary hallucination guard. Relevancy and context precision
 * are secondary and belong in a detailed panel.
 */

import { useState } from "react";
import { Loader2, Shield, ShieldAlert, ShieldCheck } from "lucide-react";
import { api } from "@/api/client";
import type { EvaluationScore } from "@/api/types";
import { Button } from "@/components/ui/button";

interface EvaluationBadgeProps {
  messageId: string;
  evaluation?: EvaluationScore[];
  onEvaluationComplete?: (scores: EvaluationScore[]) => void;
}

/**
 * Renders the right shield icon and color based on the faithfulness score.
 *
 * WHY: Three-tier coloring (green / amber / red) maps to intuitive traffic-light
 * semantics. 0.8+ = trustworthy, 0.5–0.8 = uncertain, <0.5 = likely hallucinated.
 * Using filled/outlined shield variants reinforces the safety metaphor.
 */
function FaithfulnessIcon({ score }: { score: number }) {
  if (score >= 0.8) {
    return <ShieldCheck className="size-3.5 text-emerald-600 shrink-0" />;
  }
  if (score >= 0.5) {
    return <Shield className="size-3.5 text-amber-500 shrink-0" />;
  }
  return <ShieldAlert className="size-3.5 text-red-500 shrink-0" />;
}

export function EvaluationBadge({
  messageId,
  evaluation,
  onEvaluationComplete,
}: EvaluationBadgeProps) {
  const [loading, setLoading] = useState(false);

  // PATTERN: Find the faithfulness score from the evaluation array.
  // The full 3-metric evaluation includes "faithfulness", "answer_relevancy",
  // and "context_precision". We key off evaluation.length < 3 to decide
  // whether to show the Evaluate button — fewer than 3 means the full run
  // hasn't completed yet (we may have only a streaming faithfulness score).
  const faithfulness = evaluation?.find((e) => e.metric === "faithfulness");
  const hasFullEvaluation = (evaluation?.length ?? 0) >= 3;

  async function handleEvaluate() {
    setLoading(true);
    try {
      const scores = await api.evaluateMessage(messageId);
      onEvaluationComplete?.(scores);
    } catch (err) {
      // WHY: Log the error but don't surface a toast — evaluation is non-critical.
      // The button will re-enable so the user can retry manually.
      console.error("Evaluation failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Nothing to show yet — no partial score and full eval hasn't run
  if (!faithfulness && hasFullEvaluation) return null;

  return (
    <div className="flex items-center gap-2 mt-1">
      {/* Faithfulness score chip — shown as soon as we have any score */}
      {faithfulness && (
        <div className="flex items-center gap-1">
          <FaithfulnessIcon score={faithfulness.score} />
          <span className="text-xs text-muted-foreground">
            {Math.round(faithfulness.score * 100)}%
          </span>
        </div>
      )}

      {/* Evaluate button — only visible when the full evaluation hasn't run */}
      {!hasFullEvaluation && (
        <Button
          variant="ghost"
          size="xs"
          onClick={handleEvaluate}
          disabled={loading}
          className="text-xs text-muted-foreground h-5 px-1.5"
        >
          {loading ? (
            // PATTERN: Spinner replaces button text during async call so the
            // user gets immediate visual feedback without layout shift.
            <Loader2 className="size-3 animate-spin" />
          ) : (
            "Evaluate"
          )}
        </Button>
      )}
    </div>
  );
}
