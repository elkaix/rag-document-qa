/**
 * Shared conversation page — read-only public view.
 *
 * RAG Pipeline Position:
 *   This is a PRESENTATION-ONLY page. It fetches a previously shared
 *   conversation via a unique token and renders the messages in a
 *   read-only ChatThread. No query/retrieval happens here.
 *
 * WHY: Share tokens allow users to create public links to conversations
 *      without exposing internal conversation IDs. The token-based URL
 *      sits outside the AppLayout (no sidebar) for a clean read-only view.
 */

import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ChatThread } from "@/components/chat/chat-thread";
import type { ChatMessage } from "@/api/types";

export default function SharedPage() {
  const { token } = useParams<{ token: string }>();

  // PATTERN: Conditional query — only fires when we have a valid token
  //          from the URL params. Prevents a request to /api/shared/undefined.
  const { data, isLoading, error } = useQuery({
    queryKey: ["shared", token],
    queryFn: () => api.getShared(token!),
    enabled: !!token,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-500">
        Loading...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-500">
        Conversation not found.
      </div>
    );
  }

  // WHY: Map MessageInfo (from API) to ChatMessage (for ChatThread).
  //      The ChatThread component expects a simpler shape with just
  //      id, role, content, and optional sources.
  const messages: ChatMessage[] = data.messages.map((m) => ({
    id: m.id,
    role: m.role as "user" | "assistant",
    content: m.content,
    sources: m.sources,
  }));

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="text-2xl font-semibold mb-1">{data.title}</h1>
      <p className="text-sm text-gray-500 mb-6">
        Shared conversation (read-only)
      </p>
      <div className="border rounded-lg overflow-hidden">
        <ChatThread messages={messages} />
      </div>
    </div>
  );
}
