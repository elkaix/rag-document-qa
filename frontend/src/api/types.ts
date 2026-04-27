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

export interface WsReasoningMessage {
  type: "reasoning";
  content: string;
}

export interface WsStatusMessage {
  type: "status";
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

/**
 * Per-request timing and token-cost summary emitted by the backend after
 * the "done" event. Mirrors the backend `StageTelemetry` dataclass.
 *
 * WHY a dedicated interface: keeps the telemetry shape self-documenting and
 * lets the UI consume it without casting or runtime duck-typing.
 */
export interface TelemetryPayload {
  retrieve_ms: number;
  generate_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
}

export interface WsTelemetryMessage {
  type: "telemetry";
  content: TelemetryPayload;
}

export interface EvaluationScore {
  metric: string;
  score: number;
  reasoning: string;
  details?: string;
  judge_model?: string;
  evaluated_at?: string;
}

export interface WsEvaluationMessage {
  type: "evaluation";
  content: EvaluationScore;
}

export type WsMessage =
  | WsTokenMessage
  | WsReasoningMessage
  | WsStatusMessage
  | WsDoneMessage
  | WsErrorMessage
  | WsEvaluationMessage
  | WsTelemetryMessage;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  /** LLM chain-of-thought tokens streamed before the answer. */
  reasoning?: string;
  /** Programmatic status events describing retrieval progress. */
  statusLog?: string[];
  /** Seconds spent on the CoT reasoning pass. Set when answer starts streaming. */
  thinkingSeconds?: number;
  /** True once the final `done` event has been received — used by the UI
   *  to decide when to collapse the Thinking panel. Kept open through the
   *  whole answer stream so reasoning stays visible in real time. */
  streamDone?: boolean;
  /** LLM-as-judge evaluation scores (faithfulness, relevancy, precision). */
  evaluation?: EvaluationScore[];
  /** Per-request timing and cost from the backend telemetry event. */
  telemetry?: TelemetryPayload;
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
