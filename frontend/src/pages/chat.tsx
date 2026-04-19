/**
 * Chat page — main conversation interface with streaming responses.
 *
 * RAG Pipeline Position:
 *   Document -> Chunks -> Embeddings -> Vector Store -> RETRIEVAL -> GENERATION -> [CHAT PAGE]
 *                                                                                    ^^^
 *   This page is the user-facing entry point. It sends queries via WebSocket,
 *   receives streamed tokens, and displays source citations alongside answers.
 *
 * WHY: The page uses an optional :conversationId URL param to support both
 *      new chats (no ID) and resuming existing conversations (with ID).
 *      When an ID is present, we fetch the conversation history and populate
 *      the chat thread before allowing new messages.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatThread } from "@/components/chat/chat-thread";
import { ChatInput } from "@/components/chat/chat-input";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { useChat } from "@/hooks/use-chat";
import { useSettings } from "@/hooks/use-settings";
import { api } from "@/api/client";
import type { ChatMessage, EvaluationScore } from "@/api/types";

const MIN_SOURCES_W = 200;
const MAX_SOURCES_W = 600;
const DEFAULT_SOURCES_W = 288;

export default function ChatPage() {
  // WHY: Optional conversationId from URL — undefined means "new chat",
  //      a value means "resume existing conversation".
  const { conversationId } = useParams<{ conversationId: string }>();
  const { messages, sources, isStreaming, sendMessage, clearChat, loadMessages, updateEvaluation } = useChat();
  const { settings } = useSettings();
  const [sourcesWidth, setSourcesWidth] = useState(DEFAULT_SOURCES_W);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // PATTERN: Conditional query — only fetch conversation data when we have
  //          a conversationId. This avoids a wasted API call on the "new chat" page.
  const { data: conversationData } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api.getConversation(conversationId!),
    enabled: !!conversationId,
  });

  // WHY: Track which conversation data we've already loaded into the chat
  //      thread to avoid infinite re-render loops. Without this, clearChat
  //      and loadMessages ping-pong setState calls.
  const loadedDataRef = useRef<string | null>(null);

  // Clear chat when navigating to a different conversation
  useEffect(() => {
    loadedDataRef.current = null;
    clearChat();
  }, [conversationId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load conversation messages once when data arrives
  if (
    conversationData &&
    conversationData.id === conversationId &&
    loadedDataRef.current !== conversationData.id
  ) {
    loadedDataRef.current = conversationData.id;
    const msgs: ChatMessage[] = conversationData.messages.map((m) => ({
      id: m.id,
      role: m.role as "user" | "assistant",
      content: m.content,
      sources: m.sources,
    }));
    loadMessages(msgs);
  }

  // PATTERN: Pass conversationId to sendMessage so the backend knows which
  //          conversation to append the new message pair to.
  function handleSend(query: string) {
    sendMessage(query, settings.model, settings.topK, conversationId);
  }

  // WHY: Delegate evaluation score persistence to the useChat hook so the
  //      message state stays the single source of truth for evaluation data.
  const handleEvaluate = useCallback(
    (messageId: string, scores: EvaluationScore[]) => {
      updateEvaluation(messageId, scores);
    },
    [updateEvaluation]
  );

  // PATTERN: Derive the most recent evaluation from the message list so the
  //          SourcesPanel always reflects the last completed answer — no
  //          separate state needed.
  const latestEvaluation = [...messages].reverse().find(
    (m) => m.role === "assistant" && m.evaluation && m.evaluation.length > 0
  )?.evaluation;

  // --- Resizable sources panel drag handling ---
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const newWidth = rect.right - e.clientX;
    setSourcesWidth(Math.max(MIN_SOURCES_W, Math.min(MAX_SOURCES_W, newWidth)));
  }, []);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  return (
    <div
      ref={containerRef}
      className="grid h-full"
      style={{ gridTemplateColumns: `1fr 3px ${sourcesWidth}px`, gridTemplateRows: "auto 1fr" }}
    >
      {/* Row 1: Headers */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#E5E7EB]">
        <h1 className="text-lg font-semibold">Chat</h1>
        <Button
          variant="ghost"
          size="sm"
          onClick={clearChat}
          disabled={messages.length === 0}
        >
          <Trash2 className="mr-1.5 size-3.5" />
          Clear
        </Button>
      </div>

      {/* Divider header spacer */}
      <div className="border-b border-[#E5E7EB] bg-[#E5E7EB]" />

      <div className="px-4 py-3 border-b border-[#E5E7EB]">
        <h2 className="text-lg font-semibold">Sources</h2>
      </div>

      {/* Row 2: Content */}
      <div className="flex flex-col overflow-hidden">
        <ChatThread messages={messages} onEvaluate={handleEvaluate} />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>

      {/* Draggable divider — full height */}
      <div
        className="cursor-col-resize bg-[#E5E7EB] hover:bg-[#0d74e7]/50 active:bg-[#0d74e7] transition-colors"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />

      {/* Sources panel — contained, scrolls independently */}
      <div className="overflow-hidden">
        <SourcesPanel sources={sources} evaluation={latestEvaluation} />
      </div>
    </div>
  );
}
