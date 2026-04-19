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
import { PanelRightClose, PanelRightOpen, Trash2 } from "lucide-react";
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
  const [sourcesOpen, setSourcesOpen] = useState(true);
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
    <div ref={containerRef} className="flex h-full">
      {/* Chat column — grows to fill */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Chat header */}
        <div className="flex h-[50px] items-center justify-between px-4 border-b border-[#E5E7EB] bg-white">
          <h1 className="text-lg font-semibold">Chat</h1>
          <div className="flex items-center gap-1">
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearChat}
                className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
              >
                <Trash2 className="mr-1.5 size-3.5" />
                Clear
              </Button>
            )}
            {!sourcesOpen && (
              <button
                className="rounded-md p-1 transition-colors"
                style={{ color: "#9CA3AF" }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "#F3F4F6"; e.currentTarget.style.backgroundColor = "#3a3f44"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "#9CA3AF"; e.currentTarget.style.backgroundColor = "transparent"; }}
                onClick={() => setSourcesOpen(true)}
                title="Show sources"
              >
                <PanelRightOpen className="size-4" />
              </button>
            )}
          </div>
        </div>
        {/* Chat content */}
        <ChatThread messages={messages} onEvaluate={handleEvaluate} />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>

      {/* Draggable divider */}
      {sourcesOpen && (
        <div
          className="w-[3px] cursor-col-resize bg-[#E5E7EB] hover:bg-[#0d74e7]/50 active:bg-[#0d74e7] transition-colors"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
      )}

      {/* Sources column — animated width */}
      <div
        className="flex flex-col overflow-hidden bg-[#24292d] transition-all duration-300 ease-in-out"
        style={{ width: sourcesOpen ? sourcesWidth : 0, minWidth: sourcesOpen ? sourcesWidth : 0 }}
      >
        {/* Sources header */}
        <div className="flex h-[50px] items-center justify-between px-4 border-b border-[#3a3f44] shrink-0">
          <h2 className="text-lg font-semibold whitespace-nowrap" style={{ color: "#F3F4F6" }}>Sources</h2>
          <button
            onClick={() => setSourcesOpen(false)}
            className="rounded-md p-1 transition-colors"
            style={{ color: "#9CA3AF" }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "#F3F4F6"; e.currentTarget.style.backgroundColor = "#3a3f44"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "#9CA3AF"; e.currentTarget.style.backgroundColor = "transparent"; }}
            title="Close sources"
          >
            <PanelRightClose className="size-4" />
          </button>
        </div>
        {/* Sources content */}
        <div className="flex flex-1 overflow-hidden">
          <SourcesPanel sources={sources} evaluation={latestEvaluation} />
        </div>
      </div>
    </div>
  );
}
