import { useCallback, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { wsUrl } from "@/api/client";
import type { ChatMessage, SourceInfo, WsMessage } from "@/api/types";

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}-${Date.now()}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const assistantIdRef = useRef<string>("");
  const qc = useQueryClient();

  const sendMessage = useCallback(
    (query: string, model: string, topK: number, conversationId?: string) => {
      const userMsg: ChatMessage = { id: nextId(), role: "user", content: query };
      const asstId = nextId();
      assistantIdRef.current = asstId;
      const asstMsg: ChatMessage = { id: asstId, role: "assistant", content: "" };
      setMessages((prev) => [...prev, userMsg, asstMsg]);
      setSources([]);
      setIsStreaming(true);

      wsRef.current?.close();
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({
          query,
          top_k: topK,
          model,
          conversation_id: conversationId ?? null,
        }));
      };

      ws.onmessage = (event) => {
        const data: WsMessage = JSON.parse(event.data);
        if (data.type === "token") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, content: m.content + data.content }
                : m
            )
          );
        } else if (data.type === "done") {
          setSources(data.sources);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, sources: data.sources }
                : m
            )
          );
          setIsStreaming(false);
          qc.invalidateQueries({ queryKey: ["conversations"] });
          ws.close();
        } else if (data.type === "error") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, content: `Error: ${data.content}` }
                : m
            )
          );
          setIsStreaming(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantIdRef.current
              ? { ...m, content: "Connection error. Is the backend running?" }
              : m
          )
        );
        setIsStreaming(false);
      };
    },
    [qc]
  );

  const clearChat = useCallback(() => {
    wsRef.current?.close();
    setMessages([]);
    setSources([]);
    setIsStreaming(false);
  }, []);

  const loadMessages = useCallback((msgs: ChatMessage[]) => {
    setMessages(msgs);
  }, []);

  return { messages, sources, isStreaming, sendMessage, clearChat, loadMessages };
}
