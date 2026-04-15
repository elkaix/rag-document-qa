export interface SourceInfo {
  doc_id: string;
  chunk_id: string;
  filename: string | null;
  score: number;
  excerpt: string;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  strategy?: string;
  model?: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceInfo[];
  confidence: number;
  latency_ms: number;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  chunks_count: number;
  status: "success" | "error";
  message?: string;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunks: number;
  upload_date: string;
  file_type: string | null;
  file_size_bytes: number | null;
}

export interface ChunkInfo {
  chunk_id: string;
  excerpt: string;
  metadata: Record<string, unknown>;
}

export interface DocumentChunksResponse {
  doc_id: string;
  filename: string;
  chunk_count: number;
  chunks: ChunkInfo[];
}

export interface WsTokenMessage {
  type: "token";
  content: string;
}

export interface WsDoneMessage {
  type: "done";
  sources: SourceInfo[];
  message_id: string | null;
  conversation_id: string | null;
}

export interface WsErrorMessage {
  type: "error";
  content: string;
}

export type WsMessage = WsTokenMessage | WsDoneMessage | WsErrorMessage;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  pinned: boolean;
  created_at: string;
  updated_at: string;
  share_token: string | null;
}

export interface MessageInfo {
  id: string;
  role: "user" | "assistant";
  content: string;
  model: string | null;
  created_at: string;
  sources: SourceInfo[];
}

export interface ConversationDetail extends ConversationSummary {
  messages: MessageInfo[];
}
