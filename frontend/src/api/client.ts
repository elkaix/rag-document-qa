import type {
  ConversationDetail,
  ConversationSummary,
  DocumentChunksResponse,
  DocumentInfo,
  QueryRequest,
  QueryResponse,
  UploadResponse,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  uploadFile: async (file: File): Promise<UploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/api/upload", {
      method: "POST",
      body: form,
    });
  },

  query: (body: QueryRequest) =>
    request<QueryResponse>("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  listDocuments: () => request<DocumentInfo[]>("/api/documents"),

  deleteDocument: (docId: string) =>
    request<{ doc_id: string; chunks_deleted: number; status: string }>(
      `/api/documents/${docId}`,
      { method: "DELETE" }
    ),

  getDocumentChunks: (docId: string) =>
    request<DocumentChunksResponse>(`/api/documents/${docId}/chunks`),

  listConversations: () =>
    request<ConversationSummary[]>("/api/conversations"),

  createConversation: (title = "New Chat") =>
    request<{ id: string; title: string; created_at: string }>("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),

  getConversation: (id: string) =>
    request<ConversationDetail>(`/api/conversations/${id}`),

  updateConversation: (id: string, patch: { title?: string; pinned?: boolean }) =>
    request<{ id: string; title: string; pinned: boolean }>(
      `/api/conversations/${id}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      }
    ),

  deleteConversation: (id: string) =>
    request<{ status: string }>(`/api/conversations/${id}`, { method: "DELETE" }),

  searchConversations: (q: string) =>
    request<ConversationSummary[]>(`/api/conversations/search?q=${encodeURIComponent(q)}`),

  exportConversation: (id: string) =>
    fetch(`${BASE_URL}/api/conversations/${id}/export`).then((r) => r.text()),

  shareConversation: (id: string) =>
    request<{ share_token: string; share_url: string }>(
      `/api/conversations/${id}/share`,
      { method: "POST" }
    ),

  getShared: (token: string) =>
    request<ConversationDetail>(`/api/shared/${token}`),
};

export function wsUrl(): string {
  const base = BASE_URL || window.location.origin;
  const url = new URL("/api/chat", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
