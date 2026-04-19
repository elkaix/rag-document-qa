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
  // WHY: Track when the CoT reasoning pass started so we can report elapsed
  //      "thought for N sec" once the answer begins — matches ChatGPT's UX.
  const thinkingStartRef = useRef<number | null>(null);
  // WHY a dedicated ref flag: Earlier versions detected "first token" by
  //      inspecting m.content === "" inside the setState updater. Under
  //      React 19 StrictMode (or any case where the updater is invoked more
  //      than once for the same logical event) the updater saw a non-empty
  //      content on the "first" token and silently skipped the stamp. A ref
  //      guard is immune to updater replay — it flips atomically on the
  //      first `token` frame the onmessage callback handles.
  const thinkingStampedRef = useRef<boolean>(false);
  const qc = useQueryClient();

  const sendMessage = useCallback(
    (query: string, model: string, topK: number, conversationId?: string) => {
      const userMsg: ChatMessage = { id: nextId(), role: "user", content: query };
      const asstId = nextId();
      assistantIdRef.current = asstId;
      const asstMsg: ChatMessage = {
        id: asstId,
        role: "assistant",
        content: "",
        reasoning: "",
        statusLog: [],
      };
      setMessages((prev) => [...prev, userMsg, asstMsg]);
      setSources([]);
      setIsStreaming(true);
      thinkingStartRef.current = performance.now();
      thinkingStampedRef.current = false;

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
        if (data.type === "status") {
          // PATTERN: Append each status line to the message's statusLog so the
          //          UI can render a lightweight timeline of what the agent is
          //          doing (searching, retrieving, analyzing, composing).
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, statusLog: [...(m.statusLog ?? []), data.content] }
                : m
            )
          );
        } else if (data.type === "reasoning") {
          // PATTERN: Accumulate reasoning tokens separately from the final
          //          answer so the "Thinking" panel can show them without
          //          interleaving with the answer bubble.
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, reasoning: (m.reasoning ?? "") + data.content }
                : m
            )
          );
        } else if (data.type === "token") {
          // WHY: The first answer token marks the end of the CoT phase —
          //      record how long thinking took so the UI can display
          //      "Thought for 3.2s" like ChatGPT's reasoning models.
          //
          // PATTERN: We decide "is this the first token?" via a ref flag
          //          OUTSIDE the React updater, so a strict-mode replay of
          //          the updater can't skip the stamp.
          let stampSeconds: number | undefined;
          if (!thinkingStampedRef.current && thinkingStartRef.current !== null) {
            thinkingStampedRef.current = true;
            stampSeconds =
              (performance.now() - thinkingStartRef.current) / 1000;
          }
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantIdRef.current) return m;
              return {
                ...m,
                content: m.content + data.content,
                ...(stampSeconds !== undefined
                  ? { thinkingSeconds: stampSeconds }
                  : {}),
              };
            })
          );
        } else if (data.type === "done") {
          setSources(data.sources);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, sources: data.sources, streamDone: true }
                : m
            )
          );
          setIsStreaming(false);
          thinkingStartRef.current = null;
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
          thinkingStartRef.current = null;
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
        thinkingStartRef.current = null;
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
