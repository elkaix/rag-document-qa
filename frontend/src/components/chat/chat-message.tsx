import { useState } from "react";
import Markdown from "react-markdown";
import { Check, Copy } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/api/types";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: ChatMessageType;
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-0.5">
      <span className="size-1.5 rounded-full bg-[#6B7280] animate-[typing-bounce_1.4s_ease-in-out_infinite]" />
      <span className="size-1.5 rounded-full bg-[#6B7280] animate-[typing-bounce_1.4s_ease-in-out_0.2s_infinite]" />
      <span className="size-1.5 rounded-full bg-[#6B7280] animate-[typing-bounce_1.4s_ease-in-out_0.4s_infinite]" />
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);
  const timerRef = useState<ReturnType<typeof setTimeout> | null>(null);

  function handleCopy() {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function handleMouseEnter() {
    if (timerRef[0]) clearTimeout(timerRef[0]);
    setHovered(true);
  }

  function handleMouseLeave() {
    const id = setTimeout(() => setHovered(false), 2500);
    timerRef[0] = id;
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

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isLoading = !isUser && message.content === "";

  return (
    <div
      className={cn("group/msg flex flex-col gap-1", isUser ? "items-end" : "items-start")}
    >
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {isUser ? "You" : "Assistant"}
        </span>
        {!isUser && !isLoading && <CopyButton text={message.content} />}
      </div>
      <div
        className={cn(
          "relative max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-[#0d74e7] text-white rounded-br-sm"
            : "bg-white text-[#24292d] border border-[#E5E7EB] rounded-bl-sm shadow-sm"
        )}
      >
        {isLoading ? (
          <TypingIndicator />
        ) : isUser ? (
          <span className="whitespace-pre-wrap">{message.content}</span>
        ) : (
          <div className="prose-chat">
            <Markdown>{message.content}</Markdown>
          </div>
        )}
      </div>
    </div>
  );
}
