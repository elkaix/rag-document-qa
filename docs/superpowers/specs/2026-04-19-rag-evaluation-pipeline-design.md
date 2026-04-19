# RAG Evaluation Pipeline — Design Spec

## Overview

A custom LLM-as-judge evaluation system that scores RAG answers across three metrics: faithfulness, answer relevancy, and context precision. No external evaluation libraries — built entirely on the existing `LLMHandler` with structured evaluation prompts.

Two modes of operation:
- **Real-time**: Automatic faithfulness check on every answer (fires after streaming completes)
- **On-demand**: Full 3-metric evaluation triggered by user action

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implementation | Custom LLM-as-judge | Zero new deps, portfolio-friendly, full control over prompts |
| Timing | Real-time faithfulness + on-demand full suite | Catches hallucinations immediately, deep analysis when needed |
| UI placement | Inline badge + Sources panel detail | Immediate signal without clutter, natural expansion point |
| Judge model | Dedicated `EVAL_MODEL` (gpt-4.1-mini) | Avoids self-evaluation bias, tunable cost/quality |
| Persistence | New `MessageEvaluation` table | Full audit trail, queryable, stores judge reasoning |

## Metrics

### Faithfulness (real-time, every answer)

Detects hallucination by checking if every claim in the answer is supported by the retrieved context.

**Algorithm:**
1. Extract individual claims from the answer
2. For each claim, check if it's supported by the retrieved context
3. Score = (supported claims) / (total claims)
4. Return score (0.0-1.0) + reasoning + claim-level breakdown

**Thresholds:**
- Green (>= 0.8): Well-grounded
- Yellow (>= 0.5): Some unsupported claims
- Red (< 0.5): Significant hallucination

### Answer Relevancy (on-demand)

Checks if the answer addresses the question asked. Judge receives question + answer (no context needed) and scores directness of response on a 0-1 scale. Penalizes tangential or generic responses.

### Context Precision (on-demand)

Checks if retrieved chunks are relevant to the question. For each chunk, the judge scores relevance (0 or 1). Context Precision = (relevant chunks) / (total chunks). Identifies noisy retrieval.

## Data Model

### New table: `MessageEvaluation`

```
message_evaluations
    id:           int (auto-increment PK)
    message_id:   str (FK → messages.id, ON DELETE CASCADE)
    metric:       str ("faithfulness" | "answer_relevancy" | "context_precision")
    score:        float (0.0 to 1.0)
    reasoning:    str (judge's explanation)
    details:      str | None (JSON — claim-level or chunk-level breakdown)
    judge_model:  str (e.g. "gpt-4.1-mini")
    evaluated_at: datetime
```

The `details` field stores structured JSON:
- Faithfulness: `{"claims": [{"claim": "...", "supported": true/false}]}`
- Context Precision: `{"chunks": [{"chunk_id": "...", "relevant": true/false}]}`
- Answer Relevancy: `null` (score + reasoning is sufficient)

## API

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/messages/{id}/evaluate` | Trigger full on-demand evaluation (3 metrics) |
| `GET` | `/api/messages/{id}/evaluation` | Fetch existing evaluation scores |

### WebSocket Extension

New event type after the `done` event:

```json
{"type": "evaluation", "content": {"metric": "faithfulness", "score": 0.92, "reasoning": "All claims supported by context."}}
```

Fires automatically after real-time faithfulness check completes. Frontend receives scores without a separate API call.

## Backend Implementation

### File changes

**`src/config.py`**
- Add `EVAL_MODEL = "gpt-4.1-mini"`

**`src/models/evaluation.py`** (new)
- `MessageEvaluation` SQLModel class

**`src/evaluation.py`** (replace stub)
- `evaluate_faithfulness(answer, contexts, llm) -> (score, reasoning, details)`
- `evaluate_answer_relevancy(question, answer, llm) -> (score, reasoning)`
- `evaluate_context_precision(question, contexts, llm) -> (score, reasoning, details)`

Each function:
1. Constructs a structured prompt requesting strict JSON output
2. Calls `llm.generate()` (single-shot, not streaming)
3. Parses the JSON response
4. Returns typed results

**`src/backend.py`**
- `evaluate_faithfulness_realtime(message_id, answer, contexts)` — internal, called after stream_query completes
- `evaluate_message(message_id)` — called by API, runs all 3 metrics, persists results

**`src/api/routes/evaluation.py`** (new)
- `POST /api/messages/{id}/evaluate`
- `GET /api/messages/{id}/evaluation`

**`src/api/routes/query.py`**
- After the streaming loop's done event, fire faithfulness check via `run_in_executor`
- Send `evaluation` WebSocket event with the score

### Evaluation Prompt Design

Each prompt instructs the judge to return strict JSON. Example faithfulness prompt:

```
System: You are an evaluation judge for a retrieval-augmented Q&A system.
Your task is to assess whether the answer is faithful to the provided
context — meaning every claim in the answer can be traced back to the
context. Return ONLY valid JSON, no markdown, no preamble.

User:
Context:
{chunks joined by separator}

Answer:
{generated answer}

Evaluate faithfulness. Return JSON with this exact schema:
{
  "claims": [
    {"claim": "extracted claim text", "supported": true|false, "evidence": "quote or null"}
  ],
  "score": float between 0.0 and 1.0 (supported_claims / total_claims),
  "reasoning": "one paragraph explaining the score"
}
```

## Frontend Implementation

### Chat Message — Inline Faithfulness Badge

After the answer bubble, a small colored dot + score appears when the `evaluation` WebSocket event arrives:

- Green dot + score for faithful (>= 0.8)
- Yellow dot + score for borderline (>= 0.5)
- Red dot + score for hallucinated (< 0.5)

Clicking the badge scrolls to the evaluation section in the Sources panel. An "Evaluate" button next to the badge triggers the full on-demand evaluation.

### Sources Panel — Evaluation Section

New section at the top of the Sources panel (above source cards):

```
Evaluation
  Faithfulness      [======    ] 0.92
  Answer Relevancy  [========  ] 0.97
  Context Precision [====      ] 0.60

  > View Details
```

"View Details" expands to show:
- Judge reasoning for each metric
- Faithfulness: claim-by-claim breakdown (supported = green, unsupported = red)
- Context Precision: per-chunk relevance verdicts

### State Changes

**`hooks/use-chat.ts`** — Handle `evaluation` WebSocket event, store on message object

**`api/types.ts`** — Add:
```typescript
interface EvaluationScore {
  metric: string;
  score: number;
  reasoning: string;
  details?: unknown;
  judge_model: string;
  evaluated_at: string;
}
```
Add `evaluation?: EvaluationScore[]` to `ChatMessage`

**`api/client.ts`** — Add `evaluateMessage(id)` and `getEvaluation(id)` API functions

**`components/chat/sources-panel.tsx`** — Add evaluation section component

**`components/chat/chat-message.tsx`** — Add inline faithfulness badge + "Evaluate" button

## Testing

### Backend tests (`tests/test_evaluation.py`)

- Test each evaluator function with mocked LLM responses (deterministic JSON)
- Test score calculation: all claims supported = 1.0, none = 0.0, partial = ratio
- Test JSON parse error handling (graceful fallback when judge returns malformed JSON)
- Test `MessageEvaluation` persistence and cascade delete
- Test real-time faithfulness fires after stream_query

### Frontend

- Verify faithfulness badge renders with correct color for each threshold
- Verify "Evaluate" button triggers API call and displays results
- Verify evaluation section appears in Sources panel with scores

## Quality Thresholds (Production Guidelines)

| Metric | Good | Needs Review | Problem |
|--------|------|-------------|---------|
| Faithfulness | >= 0.8 | 0.5 - 0.8 | < 0.5 |
| Answer Relevancy | >= 0.8 | 0.5 - 0.8 | < 0.5 |
| Context Precision | >= 0.6 | 0.3 - 0.6 | < 0.3 |
