# RAG Eval Harness — Sub-plan 1D: Observability + Telemetry Footer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Sub-plans 1A + 1B + 1C complete (eval system is end-to-end usable).

**Goal:** Make every RAG query observable. Per-stage spans (retrieve, generate) get token counts and cost attached, exported via OpenTelemetry to Arize Phoenix at `localhost:6006`. The same per-stage timings + tokens + cost are returned in the API response and rendered as a small footer under each chat answer in the React UI. Phoenix is optional (docker-compose profile-gated); the system works without it.

**Architecture:** A new `src/observability.py` initializes an OpenTelemetry `TracerProvider` with an OTLP HTTP exporter pointing to Phoenix. A `@traced_stage(name)` decorator wraps the retriever and generator stages in `RAGBackend`; span attributes capture `top_k`, `chunk_count`, `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`. A new `StageTelemetry` Pydantic model is returned from `/api/query` and pushed as a final WebSocket event. Frontend renders a muted footer line under each assistant bubble.

**Tech Stack:** OpenTelemetry SDK (`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`), Arize Phoenix (`arize-phoenix` python SDK + Docker image), existing FastAPI backend, existing React chat UI.

**Spec:** [`docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md`](../specs/2026-04-26-rag-eval-harness-phase-1-design.md) §9.

---

## File Structure

**New files:**

| Path | Responsibility |
|------|----------------|
| `src/observability.py` | OTel tracer init, OTLP exporter setup, `@traced_stage` decorator, `record_span_attrs()` helper. |
| `src/api/schemas/telemetry.py` | `StageTelemetry` Pydantic DTO returned in API responses. |
| `frontend/src/components/chat/TelemetryFooter.tsx` | Renders the muted "Retrieve · Generate · Tokens · Cost" line. |

**Modified files:**

| Path | Change |
|------|--------|
| `requirements.txt` | Add `opentelemetry-sdk>=1.25`, `opentelemetry-exporter-otlp-proto-http>=1.25`, `arize-phoenix>=4.0` (optional dep group). |
| `src/api/main.py` | Call `init_observability()` on lifespan startup. |
| `src/backend.py` | Wrap `_retrieve` and `_generate` (or equivalent) call sites with `@traced_stage`; collect telemetry into the return value. |
| `src/api/routes/query.py` | Include `telemetry` in REST response; emit final `telemetry` WebSocket event. |
| `frontend/src/api/types.ts` | Add `TelemetryPayload` type and add `telemetry?` to `ChatMessage`. |
| `frontend/src/hooks/use-chat.ts` | Handle `telemetry` event; store on the message. |
| `frontend/src/components/chat/chat-message.tsx` | Render `<TelemetryFooter />` below the answer when `message.telemetry` is set. |
| `docker-compose.yml` | Add `phoenix` service under a new `observability` profile. |
| `Architecture.md` | Add an "Observability" subsection. |
| `README.md` | Document running with traces (`docker compose --profile observability up`). |

**Tests:**
`tests/test_observability.py`, `tests/test_api_query_telemetry.py` (extension to existing query tests), plus a frontend test for `TelemetryFooter`.

---

## Task 1 — Add observability dependencies

**Files:** `requirements.txt`

- [ ] Add (alphabetically):
  - `arize-phoenix>=4.0`
  - `opentelemetry-exporter-otlp-proto-http>=1.25`
  - `opentelemetry-sdk>=1.25`
- [ ] `pip install -r requirements.txt`; verify `python -c "import phoenix, opentelemetry; print('ok')"`.
- [ ] Commit: `chore(obs): add OpenTelemetry SDK and Arize Phoenix deps`.

---

## Task 2 — `src/observability.py`

**Files:** `tests/test_observability.py`, `src/observability.py`

**API:**
```python
TRACER_NAME = "rag-qa"

def init_observability(otlp_endpoint: str | None = None) -> None:
    """Initialize global TracerProvider + OTLP exporter.

    Idempotent — safe to call multiple times.
    `otlp_endpoint` defaults to `http://localhost:6006/v1/traces` if None.
    On import errors or connection failures, fails QUIETLY (logs warning,
    spans become no-ops). The system never crashes due to OTel.
    """

def get_tracer() -> Tracer:
    """Return the rag-qa tracer."""

def traced_stage(name: str):
    """Decorator that opens a span around the wrapped function and
    records `result_attrs` (a dict returned in the function's return
    value alongside the actual payload)."""
```

**Decorator contract:** the wrapped function must return `(payload, attrs_dict)`. The decorator opens a span, calls the function, sets each `attrs_dict[k]` as a span attribute via `span.set_attribute(k, v)`, and returns just `payload` (callers see the same shape as before instrumentation).

**Test cases (use `opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter`):**
- `init_observability()` is idempotent (call twice, only one provider).
- `@traced_stage("rag.retrieve")` on a function returning `(["chunk"], {"top_k": 5})` records a span named `rag.retrieve` with attribute `top_k=5`.
- A function that raises propagates the exception AND records the span as ERROR status.
- When OTLP endpoint is unreachable (point at `http://127.0.0.1:1`), `init_observability` does not raise; subsequent spans become no-ops gracefully.

Commit: `feat(obs): add OpenTelemetry tracer init and traced_stage decorator`.

---

## Task 3 — `src/api/schemas/telemetry.py`

**Files:** `src/api/schemas/telemetry.py`

```python
class StageTelemetry(BaseModel):
    retrieve_ms: float
    generate_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
```

(Tested implicitly by the route tests in Task 5.)

Commit: `feat(api): add StageTelemetry DTO for per-stage timings`.

---

## Task 4 — Instrument `RAGBackend.query` / `stream_query`

**Files:** `tests/test_backend_telemetry.py`, `src/backend.py` (modify)

**Approach:**
- Refactor `query` so that `_retrieve` and `_generate` (call them what the existing code calls them — read first) each return `(payload, attrs)` and are wrapped with `@traced_stage("rag.retrieve")` / `@traced_stage("rag.generate")`.
- `query` collects timings via `time.perf_counter()` around each call, builds a `StageTelemetry` object, and returns it alongside the existing answer + sources. Update the type signature accordingly (e.g. `tuple[str, list[Source], StageTelemetry]`).
- Token counting: use `tiktoken` if available (`from tiktoken import encoding_for_model`); fall back to `len(text.split()) * 1.3` heuristic with a logged warning the first time.
- Cost computed via `eval.pricing.cost_usd(model, prompt_tokens, completion_tokens)`.
- For `stream_query`: collect tokens during streaming; emit telemetry after the final token via the new event type added in Task 5.

**Test cases:**
- After `backend.query(...)`, returned `StageTelemetry` has all five fields populated with non-negative values.
- Span attributes recorded match the telemetry fields (use InMemorySpanExporter from Task 2).
- `stream_query` yields a final `("telemetry", StageTelemetry)` tuple after the answer completes.

Commit: `feat(backend): instrument RAGBackend with traced stages and StageTelemetry`.

---

## Task 5 — Update REST + WebSocket query routes

**Files:** `tests/test_api_query_telemetry.py`, `src/api/routes/query.py` (modify)

**Changes:**
- REST `/api/query` response model gains `telemetry: StageTelemetry`.
- WebSocket protocol: after the existing `done` event, emit:
  ```json
  {"type": "telemetry", "content": {"retrieve_ms": 142.0, "generate_ms": 2103.0,
                                    "prompt_tokens": 3417, "completion_tokens": 800,
                                    "cost_usd": 0.0083}}
  ```
- `done` event remains unchanged for backward compatibility.

**Test cases:**
- POST `/api/query` returns `telemetry` with all expected fields.
- WebSocket: collect events, last is `telemetry` with valid payload.
- Token counts match prompt/completion lengths within ±10% (heuristic-tolerant).

Commit: `feat(api): emit StageTelemetry in REST and WebSocket query responses`.

---

## Task 6 — Frontend types + `useChat` handler

**Files:** `frontend/src/api/types.ts` (modify), `frontend/src/hooks/use-chat.ts` (modify), test for the hook.

**Changes:**
- `types.ts`: add
  ```ts
  export interface TelemetryPayload {
    retrieve_ms: number;
    generate_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number;
  }
  ```
  and add `telemetry?: TelemetryPayload` to `ChatMessage`.
- `use-chat.ts`: in the WebSocket message handler switch, add a `case "telemetry":` branch that sets `telemetry` on the most recent assistant message via the existing reducer/state-update pattern.

**Test:** existing `use-chat` test pattern — feed a sequence of `status / reasoning / token / done / telemetry` events, assert the final assistant message has both `content` and `telemetry`.

Commit: `feat(frontend): handle telemetry event and expose on chat message`.

---

## Task 7 — `TelemetryFooter.tsx` + render under chat message

**Files:** `frontend/src/components/chat/TelemetryFooter.tsx`, `frontend/src/components/chat/chat-message.tsx` (modify)

**`TelemetryFooter`:**
- Props: `telemetry: TelemetryPayload`.
- Renders a single muted line:
  > `Retrieve {ms} · Generate {seconds-with-1-decimal-if->1s-else-ms} · {tokens.toLocaleString()} tok · ${cost_usd.toFixed(4)}`
- Tailwind: `text-xs text-muted-foreground mt-1 flex gap-2 items-center`.
- Each segment separated by a small dot (`·`).
- Tooltip on hover (shadcn/ui `<Tooltip>`) shows breakdown: prompt vs completion tokens.

**`chat-message.tsx`:** After the answer markdown render, add `{message.role === "assistant" && message.telemetry && <TelemetryFooter telemetry={message.telemetry} />}`.

**Test:** RTL test — render with sample telemetry, assert all fields present in DOM, assert tooltip appears on hover.

Commit: `feat(frontend): add TelemetryFooter under each assistant chat message`.

---

## Task 8 — `init_observability` in API lifespan; Phoenix in docker-compose

**Files:** `src/api/main.py` (modify), `docker-compose.yml` (modify)

**`src/api/main.py`:** in the existing `lifespan` async context manager, after engine + Chroma init, call:
```python
init_observability(otlp_endpoint=os.getenv("OTLP_ENDPOINT"))
```
(env override so prod can point elsewhere; default points at Phoenix on localhost.)

**`docker-compose.yml`:** add a service:
```yaml
phoenix:
  image: arizephoenix/phoenix:latest
  ports:
    - "6006:6006"   # Phoenix UI + OTLP HTTP receiver
  profiles: ["observability"]
  restart: unless-stopped
```

Document in README:
> To run with traces: `docker compose --profile observability up`. Phoenix UI at http://localhost:6006.

Commit: `feat(obs): wire init_observability in lifespan; add Phoenix service profile`.

---

## Task 9 — Update `Architecture.md` and final smoke test

**Files:** `Architecture.md` (modify)

**Add an "Observability" subsection:**
- Brief: OTel SDK exports spans to Phoenix; falls back to no-op if endpoint unreachable.
- The `rag.retrieve` span carries `top_k`, `chunk_count`; the `rag.generate` span carries `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`.
- Telemetry is also surfaced inline in the chat UI under each answer.

**Manual smoke test:**
1. `docker compose --profile observability up -d`.
2. `python -m src.api.main` and `cd frontend && bun run dev`.
3. Send a chat query.
4. Verify the chat answer shows the telemetry footer (e.g. "Retrieve 142ms · Generate 2.1s · 4,217 tok · $0.0083").
5. Open Phoenix at http://localhost:6006; verify the trace appears with both `rag.retrieve` and `rag.generate` spans, attributes populated.
6. Stop Phoenix (`docker compose stop phoenix`); send another chat query; verify the chat still works (telemetry footer still appears using server-computed values; only the trace export degrades gracefully).

Commit: `docs(arch): document observability layer; verify end-to-end traces`.

---

## Sub-plan 1D Completion Checklist

- [ ] All `tests/test_observability.py` and `test_api_query_telemetry.py` pass.
- [ ] Backend test suite green.
- [ ] Frontend test suite green.
- [ ] Manual smoke test (Task 9 step list) succeeds end-to-end.
- [ ] Phoenix-down case still works (no crashes; chat continues).
- [ ] `Architecture.md` documents the observability layer.
- [ ] README has the `--profile observability` instructions.
- [ ] `git status` clean.

---

# Phase 1 Complete — End-to-End Verification

After 1A + 1B + 1C + 1D are merged, perform the spec's Phase 1 acceptance checks (spec §17):

1. `python -m pytest tests/ -v` — all green.
2. `python -m src.eval.cli run --config configs/eval/baseline.yaml` — completes; valid run dir created.
3. Open `/eval` in browser — run appears; detail view renders with all metric columns populated.
4. Launch a second run with a different `top_k`; open `/eval/compare` — deltas non-zero, significance markers visible.
5. `docker compose --profile observability up`; open `localhost:6006`; send a query in chat; trace appears with both spans.
6. Chat-answer footer shows non-zero numbers for retrieve/generate/tokens/cost.
7. `Architecture.md` reflects the new layer.

Phase 1 of the **Applied/GenAI + Research/Modeling** portfolio strategy is then complete. Move on to Phase 2 (hybrid search, reranking, query rewriting) as scoped in the design spec.
