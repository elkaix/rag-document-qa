import { useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import type { ChatMessage as ChatMessageType, EvaluationScore } from "@/api/types";
import { cn } from "@/lib/utils";
import { ThinkingPanel } from "./thinking-panel";
import { MarkdownRenderer } from "./markdown-renderer";
import { EvaluationBadge } from "./evaluation-badge";

// WHY removed TypingIndicator: The bouncing-dots placeholder used to show
// for the brief window between sending a query and the first server event.
// That visual slot is now owned by the ThinkingPanel — as soon as an
// assistant message is created, the panel renders a shimmering "Thinking"
// header that morphs into the live reasoning stream. The reasoning IS the
// loading indicator now.

interface ChatMessageProps {
  message: ChatMessageType;
  onEvaluate?: (messageId: string, scores: EvaluationScore[]) => void;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);
  // BUG FIX: Previously held the timeout handle in a useState tuple and
  //          mutated tuple[0] directly, which lint flags as invalid state
  //          mutation. A ref is the right primitive for "mutable, does
  //          not trigger render" values like timeout handles.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleCopy() {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function handleMouseEnter() {
    if (timerRef.current) clearTimeout(timerRef.current);
    setHovered(true);
  }

  function handleMouseLeave() {
    timerRef.current = setTimeout(() => setHovered(false), 2500);
  }

  return (
    <button
      onClick={handleCopy}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={cn(
        "p-1.5 rounded-md transition-all duration-200 border border-[#E5E7EB] text-muted-foreground hover:text-[#24292d] hover:bg-[#F3F4F6]",
        hovered || copied ? "opacity-100" : "opacity-0 group-hover/msg:opacity-100"
      )}
      title="Copy to clipboard"
    >
      {copied ? (
        <Check className="size-3.5 text-[#2fbb4f]" />
      ) : (
        <Copy className="size-3.5" />
      )}
    </button>
  );
}

export function ChatMessage({ message, onEvaluate }: ChatMessageProps) {
  const isUser = message.role === "user";
  // A freshly spawned assistant message has statusLog=[] (defined) even
  // before any event arrives. Persisted messages loaded from history have
  // statusLog=undefined — we use that distinction to decide whether to show
  // the live Thinking panel as the loading indicator vs. skipping it
  // entirely for a completed historical answer.
  const isLiveAssistant = !isUser && message.statusLog !== undefined;
  const hasAnswer = message.content !== "";

  return (
    <div
      className={cn("group/msg flex flex-col gap-1", isUser ? "items-end" : "items-start")}
    >
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {isUser ? "You" : "Assistant"}
      </span>

      {(isUser || hasAnswer || isLiveAssistant) && (
        <div
          className={cn(
            "relative",
            isUser ? "max-w-[75%]" : "max-w-[97%]"
          )}
        >
          {/* Thinking panel — same width container as the answer bubble */}
          {isLiveAssistant && (
            <ThinkingPanel
              statusLog={message.statusLog}
              reasoning={message.reasoning}
              thinkingSeconds={message.thinkingSeconds}
              streamDone={message.streamDone}
            />
          )}

          {/* Answer bubble */}
          {(hasAnswer || isUser) && (
            <div
              className={cn(
                "relative rounded-xl px-4 py-3",
                isUser
                  ? "bg-[#0d74e7] text-sm leading-relaxed text-white rounded-br-sm"
                  : "bg-white text-[#24292d] border border-[#E5E7EB] rounded-bl-sm shadow-sm"
              )}
            >
              {!isUser && hasAnswer && (
                <div className="absolute top-2 right-2">
                  <CopyButton text={message.content} />
                </div>
              )}
              {isUser ? (
                <span className="whitespace-pre-wrap">{message.content}</span>
              ) : (
                <MarkdownRenderer content={message.content} />
              )}
            </div>
          )}
          {isUser && hasAnswer && (
            <div className="mt-1 flex justify-end">
              <CopyButton text={message.content} />
            </div>
          )}
        </div>
      )}

      {/* PATTERN: EvaluationBadge is shown only after streaming completes
          (streamDone=true) so it doesn't flash up mid-response. The badge
          triggers the LLM-as-judge evaluation call on first render and
          surfaces faithfulness/relevancy/precision scores. */}
      {!isUser && hasAnswer && message.streamDone && (
        <EvaluationBadge
          messageId={message.id}
          evaluation={message.evaluation}
          onEvaluationComplete={(scores) => onEvaluate?.(message.id, scores)}
        />
      )}
    </div>
  );
}
