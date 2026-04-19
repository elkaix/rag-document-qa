import { useEffect, useRef } from "react";
import { MessageSquare } from "lucide-react";
import type { ChatMessage as ChatMessageType, EvaluationScore } from "@/api/types";
import { ChatMessage } from "./chat-message";

interface ChatThreadProps {
  messages: ChatMessageType[];
  onEvaluate?: (messageId: string, scores: EvaluationScore[]) => void;
}

export function ChatThread({ messages, onEvaluate }: ChatThreadProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      {messages.length === 0 ? (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
          <MessageSquare className="size-12 opacity-30" />
          <p className="text-sm">No messages yet. Start a conversation.</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4 p-4">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} onEvaluate={onEvaluate} />
          ))}
        </div>
      )}
    </div>
  );
}
