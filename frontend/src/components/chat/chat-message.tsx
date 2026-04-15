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

  function handleCopy() {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 p-1 rounded-md opacity-0 group-hover/msg:opacity-100 transition-opacity bg-white/80 hover:bg-white border border-[#E5E7EB] text-muted-foreground hover:text-[#24292d]"
      title="Copy to clipboard"
    >
      {copied ? (
        <Check className="size-3 text-[#2fbb4f]" />
      ) : (
        <Copy className="size-3" />
      )}
    </button>
  );
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isLoading = !isUser && message.content === "";

  return (
    <div
      className={cn("flex flex-col gap-1", isUser ? "items-end" : "items-start")}
    >
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {isUser ? "You" : "Assistant"}
      </span>
      <div
        className={cn(
          "group/msg relative max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed",
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
          <>
            <div className="prose-chat">
              <Markdown>{message.content}</Markdown>
            </div>
            <CopyButton text={message.content} />
          </>
        )}
      </div>
    </div>
  );
}
