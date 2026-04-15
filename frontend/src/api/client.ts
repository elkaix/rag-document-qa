import type {
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
};

export function wsUrl(): string {
  const base = BASE_URL || window.location.origin;
  const url = new URL("/api/chat", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
