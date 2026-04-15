import { useCallback, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatThread } from "@/components/chat/chat-thread";
import { ChatInput } from "@/components/chat/chat-input";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { useChat } from "@/hooks/use-chat";
import { useSettings } from "@/hooks/use-settings";

const MIN_SOURCES_W = 200;
const MAX_SOURCES_W = 600;
const DEFAULT_SOURCES_W = 288;

export default function ChatPage() {
  const { messages, sources, isStreaming, sendMessage, clearChat } = useChat();
  const { settings } = useSettings();
  const [sourcesWidth, setSourcesWidth] = useState(DEFAULT_SOURCES_W);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  function handleSend(query: string) {
    sendMessage(query, settings.model, settings.topK);
  }

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
        <ChatThread messages={messages} />
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
        <SourcesPanel sources={sources} />
      </div>
    </div>
  );
}
