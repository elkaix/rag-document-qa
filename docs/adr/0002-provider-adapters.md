# ADR 0002 — LLM provider adapters with injected clients

- **Status:** Accepted
- **Sequencing:** Step 2 of the RAG architecture deepening spec ([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)); resolves [#10](https://github.com/elkaix/rag-document-qa/issues/10).
- **Date:** 2026-07-12

> ADR numbering follows issue #16's sequencing steps. Step 1 (deletions +
> vector-store lookup, PR #17) landed without a written ADR; this is the first.

## Context

`src/llm_handler.py` was an 812-line class that repeated the same three-branch
provider dispatch **four times** (`generate`, `stream_response`,
`generate_messages`, `stream_messages`) and built every provider client from
module globals. Two consequences:

1. **Untestable providers.** The only way to fake a provider was to monkeypatch a
   module global (`openai.OpenAI`). That worked for OpenAI in CI but left
   **Anthropic and Ollama with zero coverage** — there was no seam to inject a
   fake at.
2. **No usage at the source.** Generation returned only `str`, so callers that
   needed token counts (cost telemetry) rebuilt the prompt and re-tokenized it —
   counting a *reconstruction*, not what the provider actually billed.

## Decision

Introduce a **`ProviderAdapter` Protocol** (`generate` -> `GenerationResult`,
`stream` -> text chunks then a terminal `Usage`) with one adapter per provider:

- **`OpenAICompatibleAdapter`** serves both OpenAI and GLM. The token-parameter
  and temperature quirks live here; GLM differs only by the injected client's
  base URL, so it is not a second code path.
- **`AnthropicAdapter`** owns the system-message split.
- **`OllamaAdapter`** owns the `/api/chat` payload shape.
- **`DummyAdapter`** is the always-available fallback.

**Clients are injected** via a zero-arg `client_factory` per adapter, so every
provider is unit-testable with a fake (see `tests/test_llm_adapters.py`) — no
module-global monkeypatching. `LLMHandler` selects one adapter at construction;
the four dispatch tables collapse to that single selection. Its public interface
is unchanged; prompt variants translate to the messages form the adapters speak.

**Usage is provider-reported where available, adapter-counted as fallback**
(`base.counted_usage`). Streaming reports usage in its terminal `Usage` event.

## Consequences

- Anthropic and Ollama gain their first real unit coverage; OpenAI/GLM quirks are
  asserted directly against a fake client.
- **Preserved behavior (deliberately):** only `ProviderUnavailableError` (missing
  SDK / missing GLM key / Ollama connection refused) triggers the dummy fallback;
  real API errors (auth, quota, malformed) still propagate. A missing *OpenAI*
  key is left to the SDK to reject, exactly as before — the asymmetry with GLM's
  explicit pre-check is intentional and preserved.
- **Consolidation:** the single-prompt Ollama `/api/generate` endpoint is gone —
  every call is translated to a messages list first, so `/api/chat` covers both.
  Semantically equivalent; the wire format differs.
- In this step, `LLMHandler`'s public `stream_*` methods still yield text only
  (they drop the adapter's terminal `Usage`), keeping the backend untouched.
  Step 3 surfaces that usage to the telemetry layer.
- The injected-client type is intentionally loose (`object`) at the SDK seam —
  the adapters call methods on external SDK objects we do not own; no `Any` and
  no `# type: ignore` are used.
